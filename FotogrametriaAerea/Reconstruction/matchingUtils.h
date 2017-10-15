#pragma once

std::vector<cv::DMatch> loweCriteria(std::vector<std::vector<cv::DMatch>> matches, const float ratio);
void retrieveKeyPoints(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keyPoints1, std::vector<cv::KeyPoint> keyPoints2, std::vector<cv::Point2f>& srcPoints, std::vector<cv::Point2f>& dstPoints);