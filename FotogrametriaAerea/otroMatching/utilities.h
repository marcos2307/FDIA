#pragma once

void getImages(std::vector<cv::Mat>&, cv::String);
void filterMatches(std::vector<cv::KeyPoint>& keyPoints1, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, int columns, double tolerance);