#include <iostream>
#include <cmath>
#include <numeric>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "Region.h"

Region::Region(cv::Mat image, int row, int column, int xSize, int ySize)
{
	c = column;
	r = row;
	x = xSize;
	y = ySize;

	for (int i = -1 * x; i <= x; i++)
	{
		for (int j = -1 * y; j <= y; j++)
		{
			pixelValues.push_back((int)image.at<uchar>(r + i, c + j));
		}
	}
	if ((int)image.at<uchar>(r, c) == 0)
	{
		mod2 = 0;
		mod = 0.0f;
	}
	else
	{
		mod2 = modulo2();
		mod = modulo();
	}
}

cv::Point2f Region::center()
{
	return cv::Point2f(c, r);
}

int Region::dotProduct(Region A)
{
	int init = 0;
	for (int i = 0; i < pixelValues.size(); i++)
	{
		init += pixelValues.at(i)*A.pixelValues.at(i);
	}
	return init;
}

int Region::modulo2()
{
	int init = 0;
	for (int i = 0; i < pixelValues.size(); i++)
	{
		init += pixelValues.at(i)*pixelValues.at(i);
	}
	return init;
}

double Region::modulo()
{
	double mod = 0;
	mod = std::sqrt((double)modulo2());
	return mod;
}

double Region::nCorr(Region A)
{
	double nCorr = dotProduct(A) / (modulo()*A.modulo());
	return nCorr;
}

void Region::printPixelValues()
{
	for each (int a in pixelValues)
	{
		std::cout << a << "\t";
	}
	std::cout << "\n";
}

void Region::draw(cv::Mat &image, cv::Scalar color)
{
	cv::Rect2d reg = cv::Rect2d(c - x, r - y, 2 * x + 1, 2 * y + 1);
	cv::rectangle(image, reg, color, -1);
}

double Region::getMod()
{
	return mod;
}

int Region::getMod2()
{
	return mod2;
}