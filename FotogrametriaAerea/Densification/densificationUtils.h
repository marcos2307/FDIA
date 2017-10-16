#pragma once

double ZNCC(cv::Point2f, cv::Point2f, cv::Mat, cv::Mat);

double s(cv::Point2f, cv::Mat);

bool compPoints(cv::Point2f, cv::Point2f);

bool byPoint1(myMatch, myMatch);

bool byPoint2(myMatch, myMatch);

bool byDistance(myMatch, myMatch);

bool compPointMatch1(myMatch, cv::Point2f);

bool compPointMatch2(myMatch, cv::Point2f);

bool compMatch1(myMatch, myMatch);

bool compMatch2(myMatch, myMatch);

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