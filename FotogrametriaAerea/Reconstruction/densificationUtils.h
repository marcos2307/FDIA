#pragma once

struct miMatch
{
	cv::Point2f point1;
	cv::Point2f point2;
	double quality;

	miMatch(cv::Point2f point1, cv::Point2f point2, double quality)
	{
		this->point1 = point1;
		this->point2 = point2;
		this->quality = quality;
	}
	void print(void)
	{
		std::cout << "distance: " << quality << std::endl;
		std::cout << "p1: (" << point1.x << ", " << point1.y << ")\t";
		std::cout << "p2: (" << point2.x << ", " << point2.y << ")" << std::endl;
	}
};

double ZNCC(cv::Point2f, cv::Point2f, cv::Mat, cv::Mat);

double s(cv::Point2f, cv::Mat);

bool compPoints(cv::Point2f, cv::Point2f);
bool compPointsi(cv::Point2f, cv::Point2f);

bool byPoint1(miMatch, miMatch);
bool byPoint1i(miMatch, miMatch);

bool byPoint2(miMatch, miMatch);
bool byPoint2i(miMatch, miMatch);

bool byQuality(miMatch, miMatch);
