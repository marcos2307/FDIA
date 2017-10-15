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
	cv::matchTemplate(gImg1(r1), gImg2(r2), result, cv::TM_CCOEFF_NORMED);
	
	return result.at<double>(0);
}