#pragma once

struct myMatch
{
	cv::Point2f point1;
	cv::Point2f point2;
	double quality;

	myMatch(cv::Point2f point1, cv::Point2f point2, double quality)
	{
		this->point1 = point1;
		this->point2 = point2;
		this->quality = quality;
	}
};

double ZNCC(cv::Point2f, cv::Point2f, cv::Mat, cv::Mat);

double s(cv::Point2f, cv::Mat);

bool compPoints(cv::Point2f, cv::Point2f);

bool byPoint1(myMatch, myMatch);

bool byPoint2(myMatch, myMatch);

bool byQuality(myMatch, myMatch);

bool compMatch1(myMatch, myMatch);

bool compMatch2(myMatch, myMatch);