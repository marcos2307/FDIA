#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "conversions.h"

void color2Gray(std::vector<cv::Mat>& colorImages, std::vector<cv::Mat>& grayImages)
{
	for (std::vector< cv::Mat >::iterator it = colorImages.begin(); it != colorImages.end(); ++it)
	{
		cv::Mat imggray;
		cv::cvtColor(*it, imggray, cv::COLOR_BGR2GRAY);
		grayImages.push_back(imggray);
	}
}

cv::Point2f toPoint2f(cv::KeyPoint kp)
{
	cv::Point2f result;
	result = kp.pt;
	return result;
}