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

struct PadTraceCandidate
{
	Point padPoint;
	Point tracePoint;

	float distance;

	int traceLabel;
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

vector<Pad> importBoardFeatureFromJson(const string& filePath)
{
	ifstream file(filePath + "/golden.json");
	nlohmann::json data;
	vector<Pad> pads;
	Pad pad;
	if (!file.is_open())
	{
		cerr << "ERROR: Cannot open file: " << filePath << endl;
		/* File cannot be opened */
		return pads;
	}

	// Read json data
	file >> data;
	for (const auto& item : data["pads"])
	{
		pad.id = item["id"];

		pad.width = item["boundary_box"][2];
		pad.height = item["boundary_box"][3];

		pad.boundaryBox.x = item["boundary_box"][0];
		pad.boundaryBox.y = item["boundary_box"][1];
		pad.boundaryBox.width = item["boundary_box"][2];
		pad.boundaryBox.height = item["boundary_box"][3];

		pad.area = item["area"];
		pad.center.x = item["center"][0];
		pad.center.y = item["center"][1];

		pads.push_back(pad);
	}

	file.close();

	return pads;
}

PadTraceCandidate findNearestPadPoint(const Mat& dist, const Mat& labels,
									  const unordered_map<int, Point>& labelToPoint,
									  const Mat& boundaryROI, const Rect& box)
{
	PadTraceCandidate candidate{};

	float minDist = FLT_MAX;

	Point nearestPadPoint;
	Point nearestTracePoint;

	int nearestTraceLabel = -1;

	for (int y = 0; y < boundaryROI.rows; y++)
	{
		for (int x = 0; x < boundaryROI.cols; x++)
		{
			if (boundaryROI.at<uchar>(y, x) == 0)
				continue;

			int gx = x + box.x;
			int gy = y + box.y;

			float d = dist.at<float>(gy, gx);

			if (d < minDist)
			{
				minDist = d;

				nearestPadPoint = Point(gx, gy);

				nearestTraceLabel = labels.at<int>(gy, gx);

				auto it = labelToPoint.find(nearestTraceLabel);

				if (it != labelToPoint.end())
				{
					nearestTracePoint = it->second;
				}
			}
		}
	}

	candidate.padPoint = nearestPadPoint;
	candidate.tracePoint = nearestTracePoint;
	candidate.distance = minDist;
	candidate.traceLabel = nearestTraceLabel;

	return candidate;
}

int main(int argc, char const* argv[])
{
	/* Get all images path*/
	string goldenFolder = "./Images/golden";
	string datasetImgFolder = "./Images/predict";

	vector<string> datasetImagePath = getImagePaths(datasetImgFolder);

	// if (datasetImagePath.empty())
	// {
	// 	cout << "Predict Image Not Found" << endl;
	// 	return -1;
	// }

	/* Load Image */
	int index = 0;

	Mat goldenImg = imread(goldenFolder + "/golden.jpeg");
	Mat padMaskGoldenImg = imread(goldenFolder + "/golden_pad_mask.jpeg");
	Mat goldenNoPad = goldenImg.clone();
	Mat padMask;

	cvtColor(padMaskGoldenImg, padMask, COLOR_BGR2GRAY);
	/* Check Image */
	if (goldenImg.empty())
	{
		cout << "Cannot load golden image" << endl;
		return -1;
	}
	else
	{
		// Golden Image Extractor
		Mat keepMask;
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
		bitwise_not(padMaskGoldenImg, keepMask);
		goldenNoPad.setTo(cv::Scalar(0, 0, 0), padMaskGoldenImg);

		int lMin = 116, lMax = 255;
		Mat connectivityMask;
		Mat copperClean;
		Mat traceMask = thresholdImg(goldenNoPad, kernel, LAB, L, lMin, lMax);

		morphologyEx(traceMask, copperClean, MORPH_OPEN, kernel, Point(-1, -1), 2);
		bitwise_not(copperClean, copperClean);
		bitwise_or(copperClean, padMask, connectivityMask);

		// Boundary Pad
		Mat erodedPad, padBoundaryMask;
		erode(padMask, erodedPad, kernel);
		padBoundaryMask = padMask - erodedPad;

		// Gather all pads properties
		Mat labels, stats, centroids;
		int numComponents = cv::connectedComponentsWithStats(padBoundaryMask, labels, stats,
															 centroids, 8, CV_32S);

		vector<Pad> pads = importBoardFeatureFromJson("./Images/golden");
		Mat traceDist, traceLabels;

		distanceTransform(traceMask, traceDist, traceLabels, DIST_L2, 5, DIST_LABEL_PIXEL);

		unordered_map<int, Point> labelToPoint;

		for (int y = 0; y < traceMask.rows; y++)
		{
			for (int x = 0; x < traceMask.cols; x++)
			{
				if (traceMask.at<uchar>(y, x) == 0)
				{
					int label = traceLabels.at<int>(y, x);
					labelToPoint[label] = Point(x, y);
				}
			}
		}
		Mat debug = goldenImg.clone();
		if (!pads.empty())
		{
			for (const auto& pad : pads)
			{
				cv::Rect box = pad.boundaryBox;

				cv::Mat boundaryROI = padBoundaryMask(box);

				int pixels = cv::countNonZero(boundaryROI);

				if (pixels == 0)
				{
					std::cout << "Pad " << pad.id << ": NO BOUNDARY" << std::endl;

					continue;
				}
				PadTraceCandidate candidate = findNearestPadPoint(traceDist, traceLabels,
																  labelToPoint, boundaryROI, box);
				cout << "Pad " << pad.id << " -> "
					 << "PadPoint: (" << candidate.padPoint.x << ", " << candidate.padPoint.y
					 << ") "
					 << "TracePoint: (" << candidate.tracePoint.x << ", " << candidate.tracePoint.y
					 << ") "
					 << "Gap: " << candidate.distance << " Label: " << candidate.traceLabel << endl;
				// debug
				if (candidate.distance <= (float)4)
				{
					circle(debug, candidate.padPoint, 3, Scalar(0, 255, 0), -1);

					circle(debug, candidate.tracePoint, 3, Scalar(0, 0, 255), -1);

					line(debug, candidate.padPoint, candidate.tracePoint, Scalar(255, 0, 0), 1);
				}
			}
		}
		else
		{
			cout << "Cannot export json data" << endl;
		}

		imshow("Raw Image", goldenImg);
		imshow("Connectivity Mask", connectivityMask);
		imshow("Visualize", debug);
	}
	waitKey(0);
	destroyAllWindows();

	return 0;
}
