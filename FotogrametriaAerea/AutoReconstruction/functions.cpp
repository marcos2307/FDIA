#include"functions.h"

vector <cv::Mat> readImages(vector<cv::String> filenames, int flag)
{
	vector<cv::Mat> images;
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		int cont = 0;
		cv::Mat src = imread(filenames[i], flag);

		if (!src.data)
		{
			++cont;
		}
		else
		{
			images.push_back(src);
		}
		cout << filenames.size() - cont <<" Image files and " << cont << " No image files detected in the curret folder" << endl;
	}
	return images;
}


bool comp(DMatch& A, DMatch& B)
{
	return A.distance < B.distance;
}

void graficarEpipolares(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, vector<Vec3f> lines1, vector<Vec3f> lines2, Mat* imout1, Mat* imout2)
{
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	img1.copyTo(outImg(rect1));
	img2.copyTo(outImg(rect2));
	for (size_t i = 0; i < pts1.size(); i++)
	{

		Scalar color1(rand() % 256, rand() % 256, rand() % 256);
		Scalar color2(rand() % 256, rand() % 256, rand() % 256);
		line(outImg(rect1),
			Point(0, -lines1[i][2] / lines1[i][1]),
			Point(img1.cols, -(lines1[i][2] + lines1[i][0] * img1.cols) / lines1[i][1]),
			color1,
			THICKNESS);
		circle(outImg(rect1), pts1[i], POINT_SIZE, color1, -1, CV_AA);
		line(outImg(rect2),
			Point(0, -lines2[i][2] / lines2[i][1]),
			Point(img2.cols, -(lines2[i][2] + lines2[i][0] * img2.cols) / lines2[i][1]),
			color2,
			THICKNESS);
		circle(outImg(rect2), pts2[i], POINT_SIZE, color2, -1, CV_AA);

		*imout1 = outImg(rect1).clone();
		*imout2 = outImg(rect2).clone();
	}
	namedWindow("IMG", CV_WINDOW_NORMAL);
	imshow("IMG", outImg);
	waitKey(0);
}

void graficarMatches(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, int flag)
{
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	img1.copyTo(outImg(rect1));
	img2.copyTo(outImg(rect2));
	for (size_t i = 0; i < pts1.size(); i++)
	{
		if (flag == ONE_BY_ONE)
		{
			img1.copyTo(outImg(rect1));
			img2.copyTo(outImg(rect2));
		}

		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		circle(outImg(rect1), pts1[i], POINT_SIZE, color, -1, CV_AA);
		circle(outImg(rect2), pts2[i], POINT_SIZE, color, -1, CV_AA);
		Point2f p2 = Point2f(pts2[i].x + img1.cols, pts2[i].y);
		line(outImg, pts1[i], p2, color, 5);
		if (flag == ONE_BY_ONE)
		{
			namedWindow("IMG", CV_WINDOW_NORMAL);
			imshow("IMG", outImg);
			waitKey(0);
		}
	}
	if (flag == ALL_IN_ONE)
	{
		namedWindow("IMG", CV_WINDOW_NORMAL);
		imshow("IMG", outImg);
		waitKey(0);
	}
}

void graficarLinea(Mat img, Vec3f linea, Scalar color)
{
	line(img,
		Point(0, -linea[2] / linea[1]),
		Point(img.cols, -(linea[2] + linea[0] * img.cols) / linea[1]),
		color,
		THICKNESS);
	namedWindow("Graficar Linea", CV_WINDOW_NORMAL);
	imshow("Graficar Linea", img);
	waitKey(0);
}

void generatePLY(String name, Mat pts3D, vector< Vec3b > color)
{
	ofstream f;
	f.open(name + ".ply");
	f << "ply" << endl;
	f << "format ascii 1.0" << endl;
	f << "element vertex " << pts3D.rows << endl;
	f << "property double x" << endl;
	f << "property double y" << endl;
	f << "property double z" << endl;
	f << "property uchar blue" << endl;
	f << "property uchar green" << endl;
	f << "property uchar red" << endl;
	f << "end_header" << endl;
	for (int i = 0; i < pts3D.rows; ++i)
	{
		const float* Mi = pts3D.ptr<float>(i);
		if (Mi[0] * Mi[0] + Mi[1] * Mi[1] + Mi[2] * Mi[2] < 2000)
		{
			f << Mi[0] << " " << Mi[1] << " " << Mi[2] << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << endl;
		}
		else
		{
			f << 0 << " " << 0 << " " << 0 << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << endl;
		}
	}

	f.close();
}

void generatePLYcameras(String name, vector < Camera > camera, Mat pts3D, vector< Vec3b > color)
{
	
	ofstream f;
	f.open(name + ".ply");
	f << "ply" << endl;
	f << "format ascii 1.0" << endl;
	f << "element vertex " << 5 * camera.size() + pts3D.rows << endl;
	f << "property double x" << endl;
	f << "property double y" << endl;
	f << "property double z" << endl;
	f << "property uchar blue" << endl;
	f << "property uchar green" << endl;
	f << "property uchar red" << endl;
	f << "element edge " << 8 * camera.size() << endl;
	f << "property int vertex1" << endl;
	f << "property int vertex2" << endl;
	f << "property uchar blue" << endl;
	f << "property uchar green" << endl;
	f << "property uchar red" << endl;
	f << "end_header" << endl;


	for (int i = 0; i < camera.size(); ++i)
	{
		vector < Point3d > point = camera[i].getPoints();
		for ( int j = 0; j < point.size(); j++)
		{
			if (i == 0)
			{
				f << point[j].x << " " << point[j].y << " " << point[j].z << " " << 255 << " " << 255 << " " << 255 << endl;
			}
			else f << point[j].x << " " << point[j].y << " " << point[j].z << " " << 255 << " " << 0 << " " << 0 << endl;
		}
	}
	
	for (int i = 0; i < pts3D.rows; ++i)
	{
		const float* Mi = pts3D.ptr<float>(i);
		f << Mi[0] << " " << Mi[1] << " " << Mi[2] << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << endl;
	}

	for (int i = 0; i < camera.size(); ++i)
	{
		f << 5 * i << " " << 5 * i + 1 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i << " " << 5 * i + 2 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i << " " << 5 * i + 3 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i << " " << 5 * i + 4 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i + 1 << " " << 5 * i + 2 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i + 2 << " " << 5 * i + 3 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i + 3 << " " << 5 * i + 4 << " " << 255 << " " << 0 << " " << 0 << endl;
		f << 5 * i + 4 << " " << 5 * i + 1 << " " << 255 << " " << 0 << " " << 0 << endl;
	}

	f.close();
}

vector <ImageInfo> getImageInfo(String txt_file)
{
	vector< ImageInfo> Iminfo;
	vector < string > word;
	string line;
	ifstream info;
	info.open(txt_file);
	if (info.is_open())
	{
		//elimina la primera linea(titulos del archivo)
		getline(info, line);
		line.clear();

		//lee cada linea del archivo
		while (getline(info, line))
		{
			word = split(line);

			ImageInfo Im;
			Im.name += word[0];
			Im.latitude = stod(word[1]);
			Im.longitude = stod(word[2]);
			Im.height = stod(word[3]);
			Im.yaw = stod(word[4]);
			Im.pitch = stod(word[5]);
			Im.roll = stod(word[6]);
			Iminfo.push_back(Im);
			word.clear();

		}
		info.close();
	}

	else cout << "Unable to open file";
	return Iminfo;
}

cv::Mat ypr2rm(ImageInfo Im)
{
	double alpha = Im.yaw * DEG_TO_RAD;
	double betha = Im.pitch * DEG_TO_RAD;
	double gamma = Im.roll * DEG_TO_RAD;

	Mat R = Mat(3, 3, CV_64FC1);

	R.at<double>(0, 0) = cos(alpha)*cos(betha);
	R.at<double>(0, 1) = cos(alpha)*sin(betha)*sin(gamma) - sin(alpha)*cos(gamma);
	R.at<double>(0, 2) = cos(alpha)*sin(betha)*cos(gamma) + sin(alpha)*sin(gamma);
	R.at<double>(1, 0) = sin(alpha)*cos(betha);
	R.at<double>(1, 1) = sin(alpha)*sin(betha)*sin(gamma) + cos(alpha)*cos(gamma);
	R.at<double>(1, 2) = sin(alpha)*sin(betha)*cos(gamma) - cos(alpha)*sin(gamma);
	R.at<double>(2, 0) = (-1)*sin(betha);
	R.at<double>(2, 1) = cos(betha)*sin(gamma);
	R.at<double>(2, 2) = cos(betha)*cos(gamma);

	return R;
}

vector < string > split(string line, char separator)
{
	vector < string > word(1);
	int pos = 0;
	for (size_t i = 0; i < line.size(); i++)
	{
		if (line[i] == separator)
		{
			word.resize(word.size() + 1);
			++pos;
			++i;
		}
		word[pos] += line[i];
	}
	return word;
}