#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <windows.h>

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
void drawRectangle(Mat src, Mat dst, vector<Point> element)
{
	Rect box = boundingRect(element);
	rectangle(src, box, Scalar(0, 0, 255), 2);
	// rectangle(dst, box, Scalar(0, 0, 255), 2);
}

void drawCircle(Mat src, Mat dst, vector<Point> element)
{
	Point2f center;
	float radius;
	minEnclosingCircle(element, center, radius);
	circle(src, center, (int)radius, Scalar(0, 0, 255), 2);
}

void annotateObject(Mat img, vector<Point> element, string content)
{
	Rect r = boundingRect(element);
	Point textPos(r.x, r.y - 10);
	putText(img, content, textPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 144, 145), 1);
}

vector<string> getAllImagePaths(string folderName)
{
	vector<string> imagePaths;
	string folder = "./Images";

	/*Load Image path*/
	for (auto& entry : fs::directory_iterator(folder))
	{
		string ext = entry.path().extension().string();
		if (ext == ".png" || ext == ".jpg" || ext == ".bmp")
		{
			imagePaths.push_back(entry.path().string());
		}
	}
	return imagePaths;
}

bool isDetectedPadMissingHole(Mat rawImg, vector<Point> contoursRawImg, int sMin, int sMax,
							  Mat kernelMorphology)
{
	vector<Point> localContour;
	vector<vector<Point>> contourList;
	vector<Mat> roiChannels;
	vector<vector<Point>> contoursROI;

	Mat padROI, hsvROI, thresholdROI;
	Mat labels, stats, centroids;

	Rect roiRect = boundingRect(contoursRawImg);
	Mat roi = rawImg(roiRect).clone();

	for (auto& p : contoursRawImg)
	{
		localContour.push_back(Point(p.x - roiRect.x, p.y - roiRect.y));
	}

	Mat padMask = Mat::zeros(roiRect.size(), CV_8UC1);
	contourList.push_back(localContour);
	drawContours(padMask, contourList, 0, Scalar(255), FILLED);
	bitwise_and(roi, roi, padROI, padMask);

	GaussianBlur(padROI, padROI, Size(3, 3), 0);
	cvtColor(padROI, hsvROI, COLOR_BGR2HSV);
	split(hsvROI, roiChannels);

	threshold(roiChannels[1], thresholdROI, sMin, sMax, THRESH_BINARY_INV);
	morphologyEx(thresholdROI, thresholdROI, MORPH_OPEN, kernelMorphology);
	bitwise_not(thresholdROI, thresholdROI);
	int nLabels = connectedComponentsWithStats(thresholdROI, labels, stats, centroids);

	if (nLabels > 1)
	{
		int maxArea = 0;
		for (int i = 0; i < nLabels; i++)
		{
			int area = stats.at<int>(i, CC_STAT_AREA);
			maxArea = max(maxArea, area);
		}
		for (int i = 0; i < nLabels; i++)
		{
			int area = stats.at<int>(i, CC_STAT_AREA);
			if (area < maxArea and area > 20)
			{
				return true;
			}
		}
	}
	return false;
}

Mat thresholdImg(Mat img, Mat kernelMorphology, ColorSpace colorSpace, int colorSpaceChannel,
				 int propertyMin, int propertyMax)
{
	Mat blurImg, cvtImg, thresholdResult;
	vector<Mat> channels;

	GaussianBlur(img, blurImg, Size(3, 3), 0);
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

int main(int argc, char const* argv[])
{
	/* Get all images path*/
	string folderName = "./Images";
	vector<string> imagePaths = getAllImagePaths(folderName);
	if (imagePaths.empty())
	{
		cout << "No image found" << endl;
		return -1;
	}
	/* Load Image */
	int index = 0;
	int sMin = 75, sMax = 255, aMin = 115, aMax = 255, lMin = 180, lMax = 255;
	// namedWindow("Trackbar");

	// createTrackbar("L Min", "Trackbar", &lMin, 255);
	// createTrackbar("L Max", "Trackbar", &lMax, 255);
	// createTrackbar("A Min", "Trackbar", &aMin, 255);
	// createTrackbar("A Max", "Trackbar", &aMax, 255);
	// createTrackbar("S Min", "Trackbar", &sMin, 255);
	// createTrackbar("S Max", "Trackbar", &sMax, 255);
	while (true)
	{
		Mat img = imread(imagePaths[index]);
		/* Check Image */
		if (img.empty())
		{
			continue;
		}
		else
		{
			vector<Mat> channelsBGR, channelsHSV, channelsLAB, channelsYCrCb;
			Mat gray, hsv, lab, ycrcb, h, s, v, l, a, b, y, cr, cb;
			Mat firstRow, secondRow, thirdRow, allRows;
			while (true)
			{
				cvtColor(img, gray, COLOR_BGR2GRAY);
				cvtColor(img, hsv, COLOR_BGR2HSV);
				cvtColor(img, lab, COLOR_BGR2Lab);
				cvtColor(img, ycrcb, 	 COLOR_BGR2YCrCb);

				split(img, channelsBGR);
				split(hsv, channelsHSV);
				split(lab, channelsLAB);
				split(ycrcb, channelsYCrCb);

				h = channelsHSV[0];
				s = channelsHSV[1];
				v = channelsHSV[2];
				l = channelsLAB[0];
				a = channelsLAB[1];
				b = channelsLAB[2];
				y = channelsYCrCb[0];
				cr = channelsYCrCb[1];
				cb = channelsYCrCb[2];

				hconcat(vector<Mat>{h, s, v}, firstRow);
				hconcat(vector<Mat>{l, a, b}, secondRow);
				hconcat(vector<Mat>{y, cr, cb}, thirdRow);
				vconcat(vector<Mat>{firstRow, secondRow, thirdRow}, allRows);
				imshow("Raw Image", img);
				// imshow("Gray", gray);
				imshow("HSV", firstRow);
				imshow("LAB", secondRow);
				// imshow("YCrCb", thirdRow);

				// imshow("Channels",allRows);

				/* Wait key */

				int key = waitKeyEx(10);
				if (key == 18)
				{
					// cout << "Missing Hole: " << contoursROI.size() << endl;
				}
				// Right Arrow
				if (key == VK_RIGHT)
				{
					index++;

					if (index >= imagePaths.size())
					{
						index = 0;
					}
					break;
				}

				// Left Arrow
				else if (key == VK_LEFT)
				{
					index--;

					if (index < 0)
					{
						index = imagePaths.size() - 1;
					}
					break;
				}
				// Esc
				else if (key == 27)
				{
					return 0;
				}
			}
		}
	}
	waitKey(0);
	return 0;
}
