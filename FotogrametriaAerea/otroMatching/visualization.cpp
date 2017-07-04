#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "visualization.h"

void viewImages(std::vector<cv::Mat>& images)
{
	for (std::vector<cv::Mat>::iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::namedWindow("Image", cv::WINDOW_KEEPRATIO);
		cv::imshow("Image", *it);
		cv::waitKey(0);
		cv::destroyWindow("Image");
	}
}