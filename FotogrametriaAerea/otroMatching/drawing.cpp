#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "drawing.h"
#include "conversions.h"

void customDrawMatches(cv::Mat & inputImage1, std::vector<cv::KeyPoint>& keyPoints1, cv::Mat & inputImage2, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, int radius, int thickness, bool option)
{
	cv::Mat auxiliar1, auxiliar2, auxiliar3;

	auxiliar1 = inputImage1.clone();
	auxiliar2 = inputImage2.clone();
	cv::hconcat(auxiliar1, auxiliar2, auxiliar3);

	cv::Mat dspImage;
	for (int i = 0; i < matches.size(); ++i)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
		if (option)
		{
			dspImage = auxiliar3.clone();
		}
		else
		{
			dspImage = auxiliar3;
		}
		cv::Point2f kPoint1 = toPoint2f(keyPoints1.at(matches.at(i).queryIdx));
		cv::Point2f kPoint2 = toPoint2f(keyPoints2.at(matches.at(i).trainIdx));
		cv::Point2f kPoint2A(kPoint2.x + inputImage1.cols, kPoint2.y);
		cv::circle(dspImage, kPoint1, radius, color, -1);
		cv::circle(dspImage, kPoint2A, radius, color, -1);
		cv::line(dspImage, kPoint1, kPoint2A, color, thickness);
		cv::namedWindow("Matches", cv::WINDOW_KEEPRATIO);
		cv::imshow("Matches", dspImage);
		if (option)
		{
			cv::waitKey(0);
		}
	}
	cv::waitKey(0);
}

void customDrawMatches(cv::Mat & inputImage1, std::vector<cv::Point2f>& kPoints1, cv::Mat & inputImage2, std::vector<cv::Point2f>& kPoints2, int radius, int thickness, bool option)
{
	cv::Mat auxiliar1, auxiliar2, auxiliar3;

	auxiliar1 = inputImage1.clone();
	auxiliar2 = inputImage2.clone();
	cv::hconcat(auxiliar1, auxiliar2, auxiliar3);

	cv::Mat dspImage;

	for (int i = 0; i < kPoints1.size(); ++i)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
		if (option)
		{
			dspImage = auxiliar3.clone();
		}
		else
		{
			dspImage = auxiliar3;
		}
		cv::Point2f kPoint2A(kPoints2.at(i).x + inputImage1.cols, kPoints2.at(i).y);
		cv::circle(dspImage, kPoints1.at(i), radius, color, -1);
		cv::circle(dspImage, kPoint2A, radius, color, -1);
		cv::line(dspImage, kPoints1.at(i), kPoint2A, color, thickness);
		cv::namedWindow("Matches", cv::WINDOW_KEEPRATIO);
		cv::imshow("Matches", dspImage);
		if (option)
		{
			cv::waitKey(0);
		}
	}
	cv::waitKey(0);
}