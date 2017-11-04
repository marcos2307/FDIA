#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <fstream>
#include "densificationUtils.h"

float ZNCC(cv::Point2f point1, cv::Point2f point2, std::vector<cv::Mat> grayImages)
{
	cv::Mat result(1,1,CV_32F);
	cv::Rect r1(point1.x - 2, point1.y - 2, 5, 5);
	cv::Rect r2(point2.x - 2, point2.y - 2, 5, 5);
	cv::matchTemplate(grayImages[0](r1), grayImages[1](r2), result, cv::TM_CCOEFF_NORMED);
	return result.at<float>(0,0);
}

double s(cv::Point2f x, cv::Mat M)
{
	cv::Point2f up(x), down(x), left(x), right(x);
	up.y++;
	down.y--;
	left.x--;
	right.x++;
	double d[4] = { 0 };

	d[0] = (double)abs(M.at<char>(up) - M.at<char>(x)) / 255;
	d[1] = (double)abs(M.at<char>(down) - M.at<char>(x)) / 255;
	d[2] = (double)abs(M.at<char>(left) - M.at<char>(x)) / 255;
	d[3] = (double)abs(M.at<char>(right) - M.at<char>(x)) / 255;
	double max = 0;
	for (int i = 0; i < 4; ++i)
	{
		max = (d[i] > max) ? d[i] : max;
	}
	return max;
}

bool compPoints(cv::Point2f a, cv::Point2f b)
{
	return (a.x < b.x || (a.x == b.x && a.y < b.y));
}
bool compPointsi(cv::Point2f a, cv::Point2f b)
{
	cv::Point2i ai = (cv::Point2i)a;
	cv::Point2i bi = (cv::Point2i)b;
	return (ai.x < bi.x || (ai.x == bi.x && ai.y < bi.y));
}

bool byPoint1(miMatch a, miMatch b)
{
	return (compPoints(a.point1, b.point1));
}
bool byPoint1i(miMatch a, miMatch b)
{
	return (compPointsi(a.point1, b.point1));
}

bool byPoint2(miMatch a, miMatch b)
{
	return (compPoints(a.point2, b.point2));
}
bool byPoint2i(miMatch a, miMatch b)
{
	return (compPointsi(a.point2, b.point2));
}

bool byQuality(miMatch a, miMatch b)
{
	return (a.quality < b.quality);
}