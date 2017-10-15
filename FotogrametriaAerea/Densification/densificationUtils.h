#pragma once

double ZNCC(cv::Point2f, cv::Point2f, cv::Mat, cv::Mat);

struct myMatch
{
	cv::Point2f point1;
	cv::Point2f point2;
	double distance;

	myMatch(cv::Point2f point1, cv::Point2f point2, double distance)
	{
		this->point1 = point1;
		this->point2 = point2;
		this->distance = distance;
	}
};