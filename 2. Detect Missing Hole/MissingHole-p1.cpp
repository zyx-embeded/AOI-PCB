#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <windows.h>

/* g++ main.cpp -o app.exe `pkg-config --cflags --libs opencv4` */

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

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

int distinguishShape(vector<Point> element)
{
	Rect r = boundingRect(element);
	double area = contourArea(element);
	float ratio = (float)r.width / r.height;
	double perimeter = arcLength(element, true);
	double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);

	vector<Point> approx;
	approxPolyDP(element, approx, 0.02 * perimeter, true);
	int corners = approx.size();

	if (ratio < 1.2 and circularity >= 0.85)
	{
		return 1; /* Circle pad */
	}
	else if (ratio < 5 and circularity > 0.45 and corners > 5)
	{
		return 2; /* Capsule/Ellipse Pad*/
	}
	else if (corners == 4)
	{
		return 3; /* Rectangle */
	}
	else if (ratio > 6)
	{
		return 4; /* Trace copper */
	}
	return -1;
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
	int sMin = 130, sMax = 255;
	namedWindow("Trackbar");

	createTrackbar("S Min", "Trackbar", &sMin, 255);
	createTrackbar("S Max", "Trackbar", &sMax, 255);
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
			Mat blurImg;
			Mat hsv;
			Mat mask;
			Mat display = img.clone();
			Mat kernelMorphology = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));

			while (true)
			{
				vector<Mat> hsvChannels;
				double maxArea = 0;
				int maxAreaIndex = -1;
				/* Firstly, Blur image. Secondly, convert channels RGB to HSV channels */
				GaussianBlur(display, blurImg, Size(5, 5), 0);
				cvtColor(blurImg, hsv, COLOR_BGR2HSV);
				split(hsv, hsvChannels);

				/* Threshold HSV image base on Green color 's properties */
				threshold(hsvChannels[1], mask, sMin, sMax, THRESH_BINARY);
				// inRange(hsv, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), mask);
				bitwise_not(mask, mask);
				morphologyEx(mask, mask, MORPH_OPEN, kernelMorphology);

				/* Find contour max Area */
				vector<vector<Point>> contoursRawImg;
				findContours(mask, contoursRawImg, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
				for (int i = 0; i < contoursRawImg.size(); i++)
				{
					double area = contourArea(contoursRawImg[i]);
					if (area > maxArea)
					{
						maxArea = area;
						maxAreaIndex = i;
					}
				}

				/* Initialize Canvas then refill area equivalent to largest contour */
				Mat padMask = Mat::zeros(mask.size(), CV_8UC1);
				drawContours(padMask, contoursRawImg, maxAreaIndex, Scalar(255), FILLED);

				/* Img with largest contour (without hierarchy component)*/
				Mat roi;
				bitwise_and(display, display, roi, padMask);

				Mat hsvROI;
				cvtColor(roi, hsvROI, COLOR_BGR2HSV);
				split(hsvROI, hsvChannels);
				/* Threshold missing hole */
				Mat hsvMissingHoleROI;
				// inRange(hsvROI, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax),
				// 		hsvMissingHoleROI);
				threshold(hsvChannels[1], hsvMissingHoleROI, sMin, sMax, THRESH_BINARY);
				morphologyEx(hsvMissingHoleROI, hsvMissingHoleROI, MORPH_OPEN, kernelMorphology);

				vector<vector<Point>> contoursROI;
				findContours(hsvMissingHoleROI, contoursROI, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

				imshow("PCB Raw", img);
				imshow("PCB Canvas", padMask);
				imshow("PCB Threshold", mask);
				imshow("PCB Hole", hsvMissingHoleROI);

				/* Wait key */

				int key = waitKeyEx(10);
				if (key == 18)
				{
					// cout << "Max index: " << maxAreaIndex << endl;
					// cout << "Contour size: " << contoursRawImg.size() << endl;
					cout << "Missing Hole: " << contoursROI.size() << endl;
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
	return 0;
}
