#pragma once
#include "includes.h"
#include "defines.h"

//declaraciones 1de funciones

std::vector <cv::Mat> readImages(std::vector <cv::String>, int flag);

void graficarMatches(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, int flag = ALL_IN_ONE);

void graficarEpipolares(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> pts1, vector<cv::Point2f> pts2, std::vector<cv::Vec3f> lines1, std::vector<cv::Vec3f> lines2, cv::Mat* imout1, cv::Mat* imout2);

void generatePLY(cv::String name, cv::Mat pts3D, std::vector< cv::Vec3b > color);

void generatePLYcameras(String name, vector < Camera > camera, Mat pts3D, vector< Vec3b > color);

vector < string > split(string line, char separator = ' ');

vector <ImageInfo> getImageInfo(String txt_file);

cv::Mat ypr2rm(ImageInfo Im);