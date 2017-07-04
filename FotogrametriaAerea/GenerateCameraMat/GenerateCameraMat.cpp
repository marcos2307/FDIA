#include "includes.h"
#include "functions.h"

int main(int argc, const char** argv)
{
	double f, xpp, ypp;
	cout << "Bienvenido al asistente de creacion de archivos de camara. Ingrese los parametros de la camara" << endl;
	cout << "Distancia focal (f=fx=fy):" << endl;
	cin >> f;
	cout << "Coordenada x del punto principal:" << endl;
	cin >> xpp;
	cout << "Coordenada y del punto principal:" << endl;
	cin >> ypp;
	Mat K = Mat::eye(3, 3, CV_64FC1);
	K.at<double>(0, 0) = f;
	K.at<double>(1, 1) = f;
	K.at<double>(0, 2) = xpp;
	K.at<double>(1, 2) = ypp;
	cout << K << endl;
	Mat dist = Mat::zeros(1, 5, CV_64FC1);
	cv::FileStorage file("cam.xml", cv::FileStorage::WRITE, cv::String());
	file.write("K", K);
	file.write("dist", dist);
	system("pause");
	return 0;
}