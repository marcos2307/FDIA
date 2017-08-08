#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "reconstructionUtils.h"

void getCameraParameters(cv::String camPath, cv::Mat & K, cv::Mat & dist)
{
	cv::FileStorage f(camPath, cv::FileStorage::READ, cv::String());
	f["K"] >> K;
	f["dist"] >> dist;
}

void getInliers(std::vector<cv::Point2f> srcPoints, std::vector<cv::Point2f> dstPoints, std::vector<uchar> mask, std::vector<cv::Point2f>& inliers1, std::vector<cv::Point2f>& inliers2)
{
	for (int i = 0; i < mask.size(); i++)
	{
		if (mask.at(i) != 0)
		{
			inliers1.push_back(srcPoints[i]);
			inliers2.push_back(dstPoints[i]);
		}
	}
}

void getColors(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> inliers1, std::vector<cv::Point2f> inliers2, std::vector<cv::Vec3b> &colors1, std::vector<cv::Vec3b> &colors2)
{
	for (int i = 0; i < inliers1.size(); i++)
	{
		colors1.push_back(img1.at<cv::Vec3b>(inliers1[i]));
		colors2.push_back(img2.at<cv::Vec3b>(inliers2[i]));
	}
}