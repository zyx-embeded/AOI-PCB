#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>

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

struct Pad
{
	int id;
	Point2d center;
	double area;
	Rect boundaryBox;
	double width;
	double height;
	double aspectRatio;
	double rectangularity;
	double circularity;
	int componentId;
	double confidence;
};

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

Mat thresholdImg(Mat img, Mat kernelMorphology, ColorSpace colorSpace, int colorSpaceChannel,
				 int propertyMin, int propertyMax)
{
	Mat blurImg, cvtImg, thresholdResult;
	vector<Mat> channels;

	// blurImg = img.clone();
	GaussianBlur(img, blurImg, Size(3, 3), 0);
	// medianBlur(img, blurImg, 5);
	// bilateralFilter(img, blurImg, 9, 75, 75);
	switch (colorSpace)
	{
	case BGR:
		/* Nothing to do */
		break;
	case HSV:
		cvtColor(blurImg, cvtImg, COLOR_BGR2HSV);
		break;
	case LAB:
		cvtColor(blurImg, cvtImg, COLOR_BGR2Lab);
		break;
	case GRAY:
		/* Nothing to do */
		break;
	default:
		break;
	}
	split(cvtImg, channels);

	if (colorSpaceChannel < channels.size() and colorSpaceChannel >= 0)
	{
		threshold(channels[colorSpaceChannel], thresholdResult, propertyMin, propertyMax,
				  THRESH_BINARY_INV);
		morphologyEx(thresholdResult, thresholdResult, MORPH_OPEN, kernelMorphology);
	}
	else
	{
		thresholdResult = img.clone();
		cout << "thresholdImg, parameter invalid";
	}
	return thresholdResult;
}

Mat createPadMask(const Mat rawImg, int aMin, int aMax, int sMin, int sMax)
{
	Mat backgroundMask, padRawMask;
	Mat labels, stats, centroids;

	int pcbLabel = -1;
	int maxArea = 0;

	Mat kernelMorphology = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));

	backgroundMask = thresholdImg(rawImg, kernelMorphology, LAB, A, aMin, aMax);
	padRawMask = thresholdImg(rawImg, kernelMorphology, HSV, S, sMin, sMax);

	/* Find PCB area */
	int nLabels = connectedComponentsWithStats(backgroundMask, labels, stats, centroids);
	for (int i = 0; i < nLabels; i++)
	{
		int area = stats.at<int>(i, CC_STAT_AREA);
		;
		if (area > maxArea)
		{
			pcbLabel = i;
			maxArea = area;
		}
	}

	/* Suppress Background if it exists */
	double ratio = (double)maxArea / (rawImg.cols * rawImg.rows);
	Mat boardMask = (labels == pcbLabel);
	if ((ratio <= 0.95))
	{
		Mat invBoardMask;
		bitwise_not(boardMask, invBoardMask);
		nLabels = connectedComponentsWithStats(invBoardMask, labels, stats, centroids, 8, CV_32S);
		for (int i = 0; i < nLabels; i++)
		{
			/* Convert Pad to white color */

			double width = (double)stats.at<int>(i, CC_STAT_WIDTH);
			double height = (double)stats.at<int>(i, CC_STAT_HEIGHT);

			double wRatio = (double)width / rawImg.cols;
			double hRatio = (double)height / rawImg.rows;
			double aspect = (double)width / height;
			bool isBackground = (wRatio >= 0.95) || (hRatio >= 0.95);
			if (!isBackground)
			{
				boardMask.setTo(255, labels == i);
			}
		}
		bitwise_and(padRawMask, boardMask, padRawMask, padRawMask);
	}
	return padRawMask;
}

bool exportBoardFeaturesToJson(const vector<Pad>& pads, const string& filePath)
{
	ofstream file(filePath + "/golden.json");
	nlohmann::json j;

	if (!file.is_open())
	{
		cerr << "ERROR: Cannot open file: " << filePath << endl;

		return false;
	}

	// Extract json data
	j["pads"] = nlohmann::json::array();
	for (auto& pad : pads)
	{
		nlohmann::json p;
		p["id"] = pad.id;
		p["center"] = {pad.center.x, pad.center.y};
		p["boundary_box"] = {pad.boundaryBox.x, pad.boundaryBox.y, pad.width, pad.height};
		p["area"] = pad.area;
		j["pads"].push_back(p);
	}

	file << j.dump(4);
	file.close();

	if (file.fail())
	{
		cerr << "ERROR: Failed while writing file: " << filePath << endl;

		return false;
	}

	return true;
}

int main(int argc, char const* argv[])
{
	/* Get all images path*/
	string goldenImgFolder = "./Images/golden";
	string datasetImgFolder = "./Images/predict";

	vector<string> goldenImagePath = getImagePaths(goldenImgFolder);
	vector<string> datasetImagePath = getImagePaths(datasetImgFolder);

	if (goldenImagePath.empty())
	{
		cout << "Golden Image Not Found" << endl;
		return -1;
	}

	// if (datasetImagePath.empty())
	// {
	// 	cout << "Predict Image Not Found" << endl;
	// 	return -1;
	// }

	/* Load Image */
	int index = 0;
	int sMin = 75, sMax = 255, aMin = 100, aMax = 255, sMinROI = 128, sMaxROI = 255;
	namedWindow("Trackbar");

	createTrackbar("S Min ROI", "Trackbar", &sMinROI, 255);
	createTrackbar("S Max ROI", "Trackbar", &sMaxROI, 255);
	createTrackbar("A Min", "Trackbar", &aMin, 255);
	createTrackbar("A Max", "Trackbar", &aMax, 255);
	createTrackbar("S Min", "Trackbar", &sMin, 255);
	createTrackbar("S Max", "Trackbar", &sMax, 255);

	Mat goldenImg = imread(goldenImagePath[0]);
	/* Check Image */
	if (goldenImg.empty())
	{
		cout << "Cannot load golden image" << endl;
		return -1;
	}
	else
	{
		// Golden Image Extractor
		Mat padBinary;
		Mat kernelMorphology = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
		Mat labels, stats, centroids;

		padBinary = createPadMask(goldenImg, aMin, aMax, sMin, sMax);
		int numLabels = cv::connectedComponentsWithStats(padBinary, labels, stats, centroids, 8,
														 CV_32S);
		vector<Pad> pads;

		for (int i = 1; i < numLabels; i++)
		{
			Pad p;
			p.componentId = i;
			p.area = stats.at<int>(i, CC_STAT_AREA);
			int x = stats.at<int>(i, CC_STAT_LEFT);
			int y = stats.at<int>(i, CC_STAT_TOP);
			int w = stats.at<int>(i, CC_STAT_WIDTH);
			int h = stats.at<int>(i, CC_STAT_HEIGHT);

			p.boundaryBox = Rect(x, y, w, h);
			p.width = w;
			p.height = h;
			p.center = Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
			p.aspectRatio = static_cast<double>(w) / h;
			p.rectangularity = p.area / static_cast<double>(w * h);
			pads.push_back(p);
		}

		sort(pads.begin(), pads.end(),
			 [](const Pad& a, const Pad& b)
			 {
				 if (abs(a.center.y - b.center.y) > 10.0)
				 {
					 return a.center.y < b.center.y;
				 }
				 return a.center.x < b.center.x;
			 });
		for (int i = 0; i < pads.size(); i++)
		{
			pads[i].id = i + 1;
		}
		bool exportSuccess = exportBoardFeaturesToJson(pads,goldenImgFolder);
		imshow("Raw Image", goldenImg);
		imshow("Threshold Pad", padBinary);
	}

	waitKey(0);
	destroyAllWindows();

	return 0;
}
