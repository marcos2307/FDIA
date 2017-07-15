#pragma once
enum drawingOption { DRAW_ALL, DRAW_ONE };


void customDrawMatches(cv::Mat& inputImage1, std::vector<cv::KeyPoint>& keyPoints1, cv::Mat& inputImage2, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, int radius, int thickness, cv::String winName = "Matches", int option = DRAW_ALL);
void customDrawMatches(cv::Mat& inputImage1, std::vector<cv::Point2f>& kPoints1, cv::Mat & inputImage2, std::vector<cv::Point2f>& kPoints2, int radius, int thickness, cv::String winName = "Matches", int option = DRAW_ALL);
void drawLine(cv::Mat& inputImage, cv::Point2f inlier, cv::Vec3f dLine, cv::Scalar color, int radius, int thickness);