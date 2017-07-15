#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "matchingUtils.h"

void matchStoring(std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &matches1, std::vector<cv::DMatch> &matches2)
{
	for (int i = 0; i < matches.size(); i++)
	{
		matches1.push_back(matches[i][0]);
		matches2.push_back(matches[i][1]);
	}
}

std::vector<cv::DMatch> loweCriteria(std::vector<cv::DMatch> &inputMatch1, std::vector<cv::DMatch> &inputMatch2, const float ratio)
{
	std::vector<cv::DMatch> goodMatches;
	for (int i = 0; i < inputMatch1.size(); i++)
	{
		if (inputMatch1[i].distance < ratio*inputMatch2[i].distance)
		{
			goodMatches.push_back(inputMatch1[i]);
		}
	}
	return goodMatches;
}