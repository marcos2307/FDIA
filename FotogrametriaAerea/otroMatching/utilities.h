#pragma once

void getImages(std::vector<cv::Mat>&, cv::String);
void color2Gray(std::vector<cv::Mat>&, std::vector<cv::Mat>&);
void viewImages(std::vector<cv::Mat>&);
void generatePLY(cv::String name, cv::Mat pts3D, std::vector< cv::Vec3b > color);