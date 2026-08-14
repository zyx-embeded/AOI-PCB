#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

	if (ratio < 1.2 and circularity >= 0.9)
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

int main(int argc, char const* argv[])
{
	/* Load Image */
	int grayScale = -1;
	Mat img = imread("./Images/l_light_01_missing_hole_01_1_600.jpg");
	/* Check Image */
	if (img.empty())
	{
		cout << "Cannot load Image" << endl;
		return -1;
	}
	else
	{
		cout << "Load Successful" << endl;
		/* Convert BGR Image to Grayscale Image */
		Mat imgGrayScale;
		Mat imgGaussianBlur;
		Mat imgBilateralFilter;
		Mat imgBilateralMorphology;
		Mat kernelMorphology = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		Mat kernelDilate = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		vector<vector<Point>> contoursGauss;
		vector<vector<Point>> contoursBilateral;

		// resize(img, img, Size(840, 840), INTER_AREA);
		cvtColor(img, imgGrayScale, COLOR_BGR2GRAY);
		/* Show Image */
		GaussianBlur(imgGrayScale, imgGaussianBlur, cv::Size(5, 5), 0);
		bilateralFilter(imgGrayScale, imgBilateralFilter, 11, 20, 25);
		// adaptiveThreshold(imgGaussianBlur, imgGaussianBlur, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
		// 				  THRESH_BINARY, 15, 4);
		// adaptiveThreshold(imgBilateralFilter, imgBilateralFilter, 255,
		// ADAPTIVE_THRESH_GAUSSIAN_C, 				  THRESH_BINARY, 15, 4);
		threshold(imgGaussianBlur, imgGaussianBlur, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
		threshold(imgBilateralFilter, imgBilateralFilter, 0, 255, THRESH_BINARY | THRESH_OTSU);
		morphologyEx(imgBilateralFilter, imgBilateralMorphology, MORPH_OPEN, kernelMorphology,
					 Point(-1, -1), 2);
		dilate(imgBilateralMorphology,imgBilateralMorphology,kernelDilate,Point(-1, -1), 1);
		findContours(imgGaussianBlur, contoursGauss, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		findContours(imgBilateralMorphology, contoursBilateral, RETR_TREE, CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contoursBilateral.size(); i++)
		{
			double area = contourArea(contoursBilateral[i]);
			// if (area > 30)
			// {
				/* Contour Img */
				// int shapeContour = distinguishShape(contoursBilateral[i]);
				// switch (shapeContour)
				// {
				// case 1: /* Circle */
				// case 2: /* Capsule/Ellipse */
				// case 3: /* Rectangle */
				// 	drawContours(img, contoursBilateral, i, Scalar(0, 0, 255), 2);
				// 	break;
				// case 4:
					drawContours(img, contoursBilateral, i, Scalar(0, 0, 255), 2);

				// 	break;
				// default:
				// 	break;
				// }
			// }
		}

		// imshow("Fault PCB Gray Scale", imgGrayScale);
		// imshow("Fault Gauss Blur", imgGaussianBlur);
		imshow("Fault Bilateral Blur", imgBilateralFilter);
		imshow("Fault Bilateral Blur Morp", imgBilateralMorphology);
		imshow("Fault PCB ", img);

		/* Wait key */
		waitKey(0);
		destroyAllWindows();
	}
	return 0;
}
