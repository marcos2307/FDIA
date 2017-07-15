#pragma once

void matchStoring(std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &matches1, std::vector<cv::DMatch> &matches2);
std::vector<cv::DMatch> loweCriteria(std::vector<cv::DMatch> &inputMatch1, std::vector<cv::DMatch> &inputMatch2, const float ratio);
