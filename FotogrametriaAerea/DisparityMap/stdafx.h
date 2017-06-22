// stdafx.h: archivo de inclusión de los archivos de inclusión estándar del sistema
// o archivos de inclusión específicos de un proyecto utilizados frecuentemente,
// pero rara vez modificados
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: mencionar aquí los encabezados adicionales que el programa necesita

//includes de libreria estandar
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

//includes de opencv
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;

//defines
#define IMAGESQUANTITY  4   //cantidad inicial de imagenes
#define POINTSQUANTITY 100  //cantidad inicial de puntos
#define INF 90000000		//infinito
#define POINT_SIZE 20		//tama~no de punto
#define THICKNESS 5			//grosor de linea
#define MINMATCHES 7        // minimo numero de matches para hallar F
#define PI 3.1415926535897932384626433832795


//declaraciones de funciones
vector <cv::Mat> readImages(cv::String folder, int flag);


bool comp(DMatch& A, DMatch& B);
void graficarMatches(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2);
void graficarEpipolares(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, vector<Vec3f> lines1, vector<Vec3f> lines2, Mat* imout1, Mat* imout2);
