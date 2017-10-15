#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "drawing.h"

void customDrawMatches(cv::Mat& inputImage1, std::vector<cv::KeyPoint>& keyPoints1, cv::Mat& inputImage2, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, cv::String winName, int radius, int thickness, int option)
{
	cv::Mat auxiliar1, auxiliar2, auxiliar3;

	auxiliar1 = inputImage1.clone();
	auxiliar2 = inputImage2.clone();
	cv::hconcat(auxiliar1, auxiliar2, auxiliar3);

	cv::Mat dspImage;
	for (int i = 0; i < matches.size(); ++i)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
		if (option != DRAW_ALL)
		{
			dspImage = auxiliar3.clone();
		}
		else
		{
			dspImage = auxiliar3;
		}
		cv::Point2f kPoint1 = keyPoints1.at(matches.at(i).queryIdx).pt;
		cv::Point2f kPoint2 = keyPoints2.at(matches.at(i).trainIdx).pt;
		cv::Point2f kPoint2A(kPoint2.x + inputImage1.cols, kPoint2.y);
		cv::circle(dspImage, kPoint1, radius, color, -1);
		cv::circle(dspImage, kPoint2A, radius, color, -1);
		cv::line(dspImage, kPoint1, kPoint2A, color, thickness);
		if (option != DRAW_ALL)
		{
			cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
			cv::imshow(winName, dspImage);
			cv::waitKey(0);
		}
	}
	if (option == DRAW_ALL)
	{
		cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
		cv::imshow(winName, dspImage);
		cv::waitKey(0);
	}
}

void customDrawMatches(cv::Mat& inputImage1, std::vector<cv::Point2f>& kPoints1, cv::Mat& inputImage2, std::vector<cv::Point2f>& kPoints2, cv::String winName, int radius, int thickness, int option)
{
	cv::Mat auxiliar1, auxiliar2, auxiliar3;

	auxiliar1 = inputImage1.clone();
	auxiliar2 = inputImage2.clone();
	cv::hconcat(auxiliar1, auxiliar2, auxiliar3);

	cv::Mat dspImage;

	for (int i = 0; i < kPoints1.size(); ++i)
	{
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, rand() % 255);
		if (option != DRAW_ALL)
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
		if (option != DRAW_ALL)
		{
			cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
			cv::imshow(winName, dspImage);
			cv::waitKey(0);
		}
	}
	if (option == DRAW_ALL)
	{
		cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
		cv::imshow(winName, dspImage);
		cv::waitKey(0);
	}
}

void drawLine(cv::Mat& inputImage, cv::Point2f inlier, cv::Vec3f dLine, cv::Scalar color, int radius, int thickness)
{
	std::vector<cv::Point2f> points;
	points.push_back(cv::Point2f(0, dLine[2] / dLine[1]));
	points.push_back(cv::Point2f(inputImage.cols, (inputImage.cols + dLine[2]) / dLine[1]));
	points.push_back(cv::Point2f(0, dLine[2] / dLine[0]));
	points.push_back(cv::Point2f(inputImage.rows, (inputImage.rows + dLine[2]) / dLine[0]));

	std::vector<cv::Point2f> linePoints;

	for (int i = 0; i < points.size(); i++)
	{
		if (points.at(i).x >= 0 && points.at(i).x <= inputImage.cols)
		{
			if (points.at(i).y >= 0 && points.at(i).y <= inputImage.rows)
			{
				linePoints.push_back(points.at(i));
			}
		}
	}
	cv::circle(inputImage, inlier, radius, color, -1);
	cv::line(inputImage, linePoints.at(0), linePoints.at(1), color, thickness);
}