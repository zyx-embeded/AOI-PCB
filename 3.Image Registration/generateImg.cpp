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

double randDouble(double min, double max)
{
	static random_device rd;
	static mt19937 gen(rd());

	uniform_real_distribution<> dis(min, max);

	return dis(gen);
}

int randInt(int min, int max)
{
	static random_device rd;
	static mt19937 gen(rd());

	uniform_int_distribution<> dis(min, max);

	return dis(gen);
}

vector<string> getAllImagePaths(string folderName)
{
	vector<string> imagePaths;
	string folder = "./Images";

	/*Load Image path*/
	for (auto& entry : fs::directory_iterator(folder))
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
	string folderName = "./Images";
	vector<string> imagePaths = getAllImagePaths(folderName);
	if (imagePaths.empty())
	{
		cout << "No image found" << endl;
		return -1;
	}
	/* Load Image */
	int index = 0;

	Mat img = imread(imagePaths[index]);
	/* Check Image */
	if (img.empty())
	{
	}
	else
	{
		Mat goldenImg = img.clone();
		Mat testImg;
		ofstream csv("labels.csv");
		int numSamples = 1000;
		for (int i = 0; i < numSamples; i++)
		{
			/* Random parameters */
			double angle = randDouble(-30, 30);

			double scale = randDouble(0.9, 1.1);

			int dx = randInt(-50, 50);
			int dy = randInt(-50, 50);

			/* Affine transform */
			Point2f center(goldenImg.cols / 2.0f, goldenImg.rows / 2.0f);

			Mat M = getRotationMatrix2D(center, angle, scale);

			M.at<double>(0, 2) += dx;
			M.at<double>(1, 2) += dy;

			/* Warp */
			Mat output;
			warpAffine(goldenImg, output, M, goldenImg.size(), INTER_LINEAR, BORDER_CONSTANT,
					   Scalar(0));

			/* Optional Noise */
			double alpha = randDouble(0.8, 1.2);
			int beta = randInt(-20, 20);
			output.convertTo(output, -1, alpha, beta);

			/* Save Image */
			char fileName[100];

			sprintf(fileName, "Images/SPI-AOI-TEST-%04d.jpeg", i);
			imwrite(fileName, output);
			csv << fileName << "," << dx << "," << dy << "," << angle << "," << scale << "\n";
		}
		csv.close();
	}
	cout << "Done\n";
	return 0;
}
