#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <fstream>
#include <random>

/* g++ main.cpp -o app.exe `pkg-config --cflags --libs opencv4` */

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

enum ColorSpace
{
	BGR,
	HSV,
	LAB,
	GRAY
};

enum ColorSpaceChannelHSV
{
	H,
	S,
	V
};

enum ColorSpaceChannelLAB
{
	L,
	A,
	B
};

struct Label
{
	int dx;
	int dy;
	double angle;
	double scale;
};

struct ORBFeature
{
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

void computeORB(const cv::Mat& gray, ORBFeature& feature)
{
	static auto orb = cv::ORB::create(1000);
	orb->detectAndCompute(gray, cv::noArray(), feature.keypoints, feature.descriptors);
}

unordered_map<string, Label> readCSV(string filePath)
{
	unordered_map<string, Label> labels;

	ifstream file(filePath);
	if (!file.is_open())
	{
		cout << "Cannot open lables.csv" << endl;
		return;
	}
	string line;
	/* Skip header line */
	getline(file, line);
	while (getline(file, line))
	{
		stringstream ss(line);

		string filename;
		string dx;
		string dy;
		string angle;
		string scale;

		getline(ss, filename, ',');
		getline(ss, dx, ',');
		getline(ss, dy, ',');
		getline(ss, angle, ',');
		getline(ss, scale, ',');

		labels.emplace(filename, Label{stoi(dx), stoi(dy), stod(angle), stod(scale)});
	}

	file.close();
	return labels;
}

vector<string> getImagePaths(string folderName)
{
	vector<string> imagePaths;

	/*Load Image path*/
	for (auto& entry : fs::directory_iterator(folderName))
	{
		string ext = entry.path().extension().string();
		transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return tolower(c); });
		if (ext == ".png" || ext == ".jpg" || ext == ".bmp" || ext == ".jpeg")
		{
			imagePaths.push_back(entry.path().string());
		}
	}
	return imagePaths;
}

int main(int argc, char const* argv[])
{
	/* Get all images path*/
	string goldenImgFolder = "./Images/golden";
	string datasetImgFolder = "./Images/dataset";
	string csvPath = "./labels.csv";

	unordered_map<string, Label> labels;
	vector<string> goldenImagePath = getImagePaths(goldenImgFolder);
	vector<string> datasetImagePath = getImagePaths(datasetImgFolder);

	labels = readCSV(csvPath);
	if (goldenImagePath.empty())
	{
		cout << "Golden Image Not Found" << endl;
		return -1;
	}

	if (datasetImagePath.empty())
	{
		cout << "Dataset Image Not Found" << endl;
		return -1;
	}
	/* Load Image */
	int index = 0;

	Mat goldenImg = imread(goldenImagePath[0]);
	/* Check Image */
	if (goldenImg.empty())
	{
		cout << "Cannot load golden image" << endl;
		return -1;
	}
	else
	{
		Ptr<CLAHE> clahe = createCLAHE();
		Mat grayGoldenImg, blurGrayGoldenImg;
		ORBFeature goldenImgORB;
		BFMatcher matcher(NORM_HAMMING);

		cvtColor(goldenImg, grayGoldenImg, cv::COLOR_BGR2GRAY);
		GaussianBlur(grayGoldenImg, blurGrayGoldenImg, Size(3, 3), 0);
		clahe->apply(blurGrayGoldenImg, blurGrayGoldenImg);
		computeORB(blurGrayGoldenImg, goldenImgORB);

		while (index < datasetImagePath.size())
		{
			Mat datasetImg = imread(datasetImagePath[index]);
			if (datasetImg.empty())
			{
				cout << "Cannot load dataset Img" << endl;
				continue;
			}
			else
			{
				Mat grayDatasetImg, blurGrayDatasetImg;
				ORBFeature datasetImgORB;
				cvtColor(datasetImg, grayDatasetImg, cv::COLOR_BGR2GRAY);
				GaussianBlur(grayDatasetImg, blurGrayDatasetImg, Size(3, 3), 0);
				clahe->apply(blurGrayDatasetImg, blurGrayDatasetImg);
				computeORB(blurGrayDatasetImg, datasetImgORB);

				vector<vector<DMatch>> knnMatches;
				vector<DMatch> goodMatches;
				matcher.knnMatch(goldenImgORB.descriptors, datasetImgORB.descriptors, knnMatches,
								 2);
				for (auto& m : knnMatches)
				{
					if (m.size() < 2)
					{
						continue;
					}
					if (m[0].distance < 0.75 * m[1].distance)
					{
						goodMatches.push_back(m[0]);
					}
				}

				vector<Point2f> srcPoint;
				vector<Point2f> dstPoint;
				for (auto& m : goodMatches)
				{
					srcPoint.push_back(goldenImgORB.keypoints[m.queryIdx].pt);
					dstPoint.push_back(datasetImgORB.keypoints[m.trainIdx].pt);
				}

				Mat inlierMask;
				Mat affine = estimateAffinePartial2D(dstPoint, srcPoint, inlierMask, RANSAC);

				Mat aligned;
				warpAffine(datasetImg, aligned, affine, goldenImg.size());

				string imagePath = datasetImagePath[index];
				string fileName = fs::path(imagePath).filename().string();
				string stem = fs::path(imagePath).stem().string();
				string ext = fs::path(imagePath).extension().string();
				imwrite("./Images/aligned/" + fileName, aligned);

				/* Extract ground truth */
				auto it = labels.find(fileName);
				if (it == labels.end())
				{
					cout << "Cannot find label: " << fileName << endl;
				}
				else
				{
					/* Evaluation */
					Label groundTruth = it->second;
					double a = affine.at<double>(0, 0);
					double b = affine.at<double>(0, 1);
					double c = affine.at<double>(1, 0);
					double d = affine.at<double>(1, 1);

					double tx = affine.at<double>(0, 2);
					double ty = affine.at<double>(1, 2);

					double scale = sqrt(a * a + c * c);
					double angle = atan2(c, a);

					angle = angle * 180.0 / CV_PI;
					double pred_dx = -tx;
					double pred_dy = -ty;

					double pred_angle = -angle;

					double pred_scale = 1.0 / scale;

					double error_dx = abs(pred_dx - groundTruth.dx);

					double error_dy = abs(pred_dy - groundTruth.dy);

					double error_angle = abs(pred_angle - groundTruth.angle);

					double error_scale = abs(pred_scale - groundTruth.scale);
				}
				index++;
			}
		}
	}
	return 0;
}
