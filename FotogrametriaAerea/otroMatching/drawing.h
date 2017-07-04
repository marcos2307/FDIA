#pragma once

void customDrawMatches(cv::Mat& inputImage1, std::vector<cv::KeyPoint>& keyPoints1, cv::Mat& inputImage2, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, int radius, int thickness, bool option = false);
void customDrawMatches(cv::Mat & inputImage1, std::vector<cv::Point2f>& kPoints1, cv::Mat & inputImage2, std::vector<cv::Point2f>& kPoints2, int radius, int thickness, bool option = false);