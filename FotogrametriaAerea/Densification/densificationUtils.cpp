#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <fstream>
#include "densificationUtils.h"

double ZNCC(cv::Point2f point1, cv::Point2f point2, cv::Mat gImg1, cv::Mat gImg2)
{
	cv::Mat result;
	cv::Rect r1(point1.x - 1, point1.y - 1, 3, 3);
	cv::Rect r2(point2.x - 1, point2.y - 1, 3, 3);
	std::cout << "matchtemplate\n\n";
	cv::matchTemplate(gImg2(r2), gImg1(r1), result, cv::TM_CCOEFF_NORMED);
	std::cout << "fin correlacion\n\n";
	return result.at<double>(0);
}

double s(cv::Point2f x, cv::Mat M)
{
	cv::Point2f up(x), down(x), left(x), right(x);
	up.y++;
	down.y--;
	left.x--;
	right.x++;
	double d[4] = { 0 };

	d[0] = abs(M.at<char>(up) - M.at<char>(x)) / (double)255;
	d[1] = abs(M.at<char>(down) - M.at<char>(x)) / (double)255;
	d[2] = abs(M.at<char>(left) - M.at<char>(x)) / (double)255;
	d[3] = abs(M.at<char>(right) - M.at<char>(x)) / (double)255;
	double max = 0;
	for (int i = 0; i < 4; ++i)
	{
		max = d[i] > max ? d[i] : max;
	}
	return max;
}

bool compPoints(cv::Point2f a, cv::Point2f b)
{
	return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

bool byPoint1(myMatch a, myMatch b)
{
	return (compPoints(a.point1,b.point1));
}

bool byPoint2(myMatch a, myMatch b)
{
	return (compPoints(a.point2, b.point2));
}

bool byDistance(myMatch a, myMatch b)
{
	return (a.distance < b.distance);
}

bool compMatch1(myMatch a, myMatch b)
{
	return (a.point1.x < b.point1.x || (a.point1.x == b.point1.x && a.point1.y < b.point1.y));
}

bool compMatch2(myMatch a, myMatch b)
{
	return (a.point2.x < b.point2.x || (a.point2.x == b.point2.x && a.point2.y < b.point2.y));
}