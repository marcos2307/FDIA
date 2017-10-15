#include <iostream>
#include <cmath>
#include <numeric>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <fstream>
#include "Region.h"

Region::Region(cv::Mat image, int row, int column, int xSize, int ySize)
{
	c = column;
	r = row;
	for (int i = -1 * xSize; i <= xSize; i++)
	{
		for (int j = -1 * ySize; j <= ySize; j++)
		{
			pixelValues.push_back((int)image.at<uchar>(row + i, column + j));
		}
	}
}

int Region::dotProduct(Region A)
{
	int init = 0;
	std::inner_product(pixelValues.begin(), pixelValues.end(), A.pixelValues.begin(), init);
	return init;
}

int Region::modulo()
{
	int mod = 0;
	std::inner_product(pixelValues.begin(), pixelValues.end(), pixelValues.begin(), mod);
	mod = std::sqrt(mod);
	return mod;
}

cv::Point2f Region::center()
{
	return cv::Point2f(c, r);
}