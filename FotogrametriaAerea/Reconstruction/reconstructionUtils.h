#pragma once

void getCameraParameters(cv::String camPath, cv::Mat &K, cv::Mat &dist);
void getInliers(std::vector<cv::Point2f> srcPoints, std::vector<cv::Point2f> dstPoints, std::vector<uchar> mask, std::vector<cv::Point2f> &inliers1, std::vector<cv::Point2f> &inliers2);
void getColors(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> inliers1, std::vector<cv::Point2f> inliers2, std::vector<cv::Vec3b> &colors1, std::vector<cv::Vec3b> &colors2);