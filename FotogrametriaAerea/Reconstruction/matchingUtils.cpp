#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "matchingUtils.h"



std::vector<cv::DMatch> loweCriteria(std::vector<std::vector<cv::DMatch>> matches, const float ratio)
{
	std::vector<cv::DMatch> goodMatches;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < ratio*matches[i][1].distance)
		{
			goodMatches.push_back(matches[i][0]);
		}
	}
	return goodMatches;
}

void retrieveKeyPoints(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keyPoints1, std::vector<cv::KeyPoint> keyPoints2, std::vector<cv::Point2f>& srcPoints, std::vector<cv::Point2f>& dstPoints)
{
	for (int i = 0; i < matches.size(); i++)
	{
		srcPoints.push_back(keyPoints1[matches[i].queryIdx].pt);
		dstPoints.push_back(keyPoints2[matches[i].trainIdx].pt);
	}
}
