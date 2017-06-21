// HolaOpenCV.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\core\core.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat a = Mat::eye(3, 3, CV_16UC1);
	cout << a << endl;
	system("pause");
}

