#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <opencv2/ximgproc.hpp>

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

struct SinglePad
{
	Mat padMask;
	Mat holeMask;
	Rect roiRect;
	vector<Point> padLocalContour;
	bool missingHole = false;
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

Mat fillHoles(const Mat& binaryROI)
{
	Mat holes, padded;
	int pad = 10;

	copyMakeBorder(binaryROI, padded, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));

	Mat flood = padded.clone();
	floodFill(flood, Point(0, 0), Scalar(255));
	bitwise_not(flood, holes);

	Mat filled = padded | holes;
	return filled(Rect(pad, pad, binaryROI.cols, binaryROI.rows)).clone();
}

vector<SinglePad> splitBridgedPads(const Mat& padMaskROI, const Mat& holeMaskROI,
								   const Rect& roiRect)
{
	Mat dist, dist8u, marker, distShow;
	Mat wsImg = padMaskROI.clone();

	Mat filledPadMask = fillHoles(padMaskROI);
	Mat filledHoleMask = fillHoles(holeMaskROI);

	vector<SinglePad> discretePads;

	distanceTransform(filledPadMask, dist, DIST_L2, 5);
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	threshold(dist, dist, 0.75, 1.0, THRESH_BINARY);
	dist.convertTo(dist8u, CV_8U, 255);

	int nums = connectedComponents(dist8u, marker);

	int peakCounts = nums - 1;
	if (peakCounts <= 1)
	{
		vector<vector<Point>> contours;
		Mat separatedHoleMask = holeMaskROI.clone();
		Mat separatedPadMask = padMaskROI.clone();
		SinglePad pad;

		pad.padMask = separatedPadMask;
		bitwise_and(separatedPadMask, separatedHoleMask, pad.holeMask, separatedPadMask);
		findContours(separatedPadMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (!contours.empty())
		{
			int maxIdx = 0;
			double maxArea = 0;

			for (int i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);

				if (area > maxArea)
				{
					maxArea = area;
					maxIdx = i;
				}
			}

			pad.padLocalContour = contours[maxIdx];
		}
		pad.roiRect = roiRect;
		discretePads.push_back(pad);
	}
	else
	{
		cvtColor(wsImg, wsImg, COLOR_GRAY2BGR);
		watershed(wsImg, marker);

		for (int label = 1; label < nums; label++)
		{
			vector<vector<Point>> contours;
			SinglePad pad;
			Mat separatedHoleMask, separatedPadMask;

			Mat labelMask = (marker == label);

			labelMask.convertTo(labelMask, CV_8U, 255);
			bitwise_and(holeMaskROI, labelMask, separatedHoleMask);
			bitwise_and(padMaskROI, labelMask, separatedPadMask);
			findContours(separatedPadMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			if (!contours.empty())
			{
				pad.padLocalContour = contours[0];
			}
			pad.padMask = separatedPadMask;
			bitwise_and(separatedPadMask, separatedHoleMask, pad.holeMask, separatedPadMask);
			pad.roiRect = roiRect;
			discretePads.push_back(pad);
		}
	}

	return discretePads;
}

Mat createPadMaskROI(Mat rawImg, vector<Point> contoursRawImg, int sMin, int sMax)
{
	Mat binaryROI, padROI, hsvROI;

	vector<Mat> ROIChannels;
	vector<Point> localContour;
	vector<vector<Point>> contourList;

	Rect roiRect = boundingRect(contoursRawImg);
	Mat ROI = rawImg(roiRect).clone();

	for (auto& p : contoursRawImg)
	{
		localContour.push_back(Point(p.x - roiRect.x, p.y - roiRect.y));
	}

	Mat padMask = Mat::zeros(roiRect.size(), CV_8UC1);
	contourList.push_back(localContour);

	drawContours(padMask, contourList, 0, Scalar(255), FILLED);
	binaryROI = padMask;

	return binaryROI;
}

Mat createHoleMaskROI(Mat rawImg, vector<Point> contoursRawImg, int sMin, int sMax)
{
	Mat binaryROI, padROI, hsvROI;

	vector<Mat> ROIChannels;
	vector<Point> localContour;
	vector<vector<Point>> contourList;

	Rect roiRect = boundingRect(contoursRawImg);
	Mat ROI = rawImg(roiRect).clone();

	for (auto& p : contoursRawImg)
	{
		localContour.push_back(Point(p.x - roiRect.x, p.y - roiRect.y));
	}

	Mat holeMask = Mat::zeros(roiRect.size(), CV_8UC1);
	contourList.push_back(localContour);

	drawContours(holeMask, contourList, 0, Scalar(255), FILLED);
	bitwise_and(ROI, ROI, padROI, holeMask);

	GaussianBlur(padROI, padROI, Size(5, 5), 0);
	cvtColor(padROI, hsvROI, COLOR_BGR2HSV);
	split(hsvROI, ROIChannels);

	threshold(ROIChannels[1], binaryROI, sMin, sMax, THRESH_BINARY);

	return binaryROI;
}

void isDetectedPadMissingHole(SinglePad& singlePad)
{
	Mat labels, stats, centroids;
	int nLabels = connectedComponentsWithStats(singlePad.holeMask, labels, stats, centroids);

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
			if (area < maxArea and area > 5)
			{
				singlePad.missingHole = true;
				return;
			}
		}
	}

	singlePad.missingHole = false;
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

Mat createPadMask(const Mat& rawImg, int aMin, int aMax, int sMin, int sMax)
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
	int sMin = 75, sMax = 255, aMin = 100, aMax = 255, sMinROI = 128, sMaxROI = 255;
	namedWindow("Trackbar");

	createTrackbar("S Min ROI", "Trackbar", &sMinROI, 255);
	createTrackbar("S Max ROI", "Trackbar", &sMaxROI, 255);
	createTrackbar("A Min", "Trackbar", &aMin, 255);
	createTrackbar("A Max", "Trackbar", &aMax, 255);
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
			Mat kernelMorphology = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
			while (true)
			{
				Mat display = img.clone();
				Mat boardOnly;
				Mat suppressBackground;
				Mat padRawMask;
				Mat labels, stats, centroids;
				bool isBackgroundExist = false;
				vector<Mat> channels;
				vector<vector<Point>> highlightMissingHolePad;
				vector<vector<Point>> boardContour;
				int pcbLabel = -1;
				int maxArea = 0;

				/* Filter contour on PCB (Without background object)  */
				padRawMask = createPadMask(display, aMin, aMax, sMin, sMax);

				findContours(padRawMask, boardContour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
				for (int i = 0; i < boardContour.size(); i++)
				{
					double area = contourArea(boardContour[i]);
					if (area < 50)
					{
						continue;
					}
					else
					{
						Rect roiRect = boundingRect(boardContour[i]);
						Mat holeMaskROI = createHoleMaskROI(display, boardContour[i], sMinROI,
															sMaxROI);
						Mat padMaskROI = createPadMaskROI(display, boardContour[i], sMinROI,
														  sMaxROI);
						vector<SinglePad> splitPads = splitBridgedPads(padMaskROI, holeMaskROI,
																	   roiRect);

						if (splitPads.size() > 1)
						{
							/* Bridged Pad */
							for (auto& singlePad : splitPads)
							{
								isDetectedPadMissingHole(singlePad);
								if (singlePad.missingHole)
								{
									vector<Point> contourGlobal;
									for (auto& p : singlePad.padLocalContour)
									{
										contourGlobal.push_back(Point(p.x + singlePad.roiRect.x,
																	  p.y + singlePad.roiRect.y));
									}
									highlightMissingHolePad.push_back(contourGlobal);
								}
							}
						}
						else
						{
							isDetectedPadMissingHole(splitPads[0]);
							if (splitPads[0].missingHole)
							{
								vector<Point> contourGlobal;
								for (auto& p : splitPads[0].padLocalContour)
								{
									contourGlobal.push_back(Point(p.x + splitPads[0].roiRect.x,
																  p.y + splitPads[0].roiRect.y));
								}

								highlightMissingHolePad.push_back(contourGlobal);
							}
						}
					}
				}
				if (!highlightMissingHolePad.empty())
				{
					drawContours(display, highlightMissingHolePad, -1, Scalar(0, 0, 255), 2);
				}
				imshow("Raw Image", img);
				imshow("Threshold Pad", padRawMask);
				imshow("Processed Image", display);
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
