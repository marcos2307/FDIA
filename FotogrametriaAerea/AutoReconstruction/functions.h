#pragma once
#include "includes.h"
#include "defines.h"

//declaraciones 1de funciones
vector <cv::Mat> readImages(cv::String folder, int flag);


bool comp(DMatch& A, DMatch& B);

void graficarMatches(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, int flag = ALL_IN_ONE);

void graficarEpipolares(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, vector<Vec3f> lines1, vector<Vec3f> lines2, Mat* imout1, Mat* imout2);

void generatePLY(String name, Mat pts3D, vector< Vec3b > color);