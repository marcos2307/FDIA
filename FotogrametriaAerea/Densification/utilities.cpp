#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <fstream>
#include "utilities.h"

void getImages(std::vector<cv::Mat>& images, cv::String path)
{
	std::cout << "Creating files list...\n";

	std::vector<cv::String> filesNames;
	cv::glob(path, filesNames);

	std::cout << "Loading Images...\n";

	for (std::vector<cv::String>::iterator it = filesNames.begin(); it != filesNames.end(); ++it)
	{
		images.push_back(cv::Mat(cv::imread(*it, cv::IMREAD_COLOR)));
		std::cout << "Loaded  " << *it << "\n";
	}

	std::cout << "Finished loading images\n\n";
}

void color2Gray(std::vector<cv::Mat>& colorImages, std::vector<cv::Mat>& grayImages)
{
	for (std::vector< cv::Mat >::iterator it = colorImages.begin(); it != colorImages.end(); ++it)
	{
		cv::Mat imggray;
		cv::cvtColor(*it, imggray, cv::COLOR_BGR2GRAY);
		grayImages.push_back(imggray);
	}
}

void viewImages(std::vector<cv::Mat>& images)
{
	for (std::vector<cv::Mat>::iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::namedWindow("Image", cv::WINDOW_KEEPRATIO);
		cv::imshow("Image", *it);
		cv::waitKey(0);
		cv::destroyWindow("Image");
	}
}

void generatePLY(cv::String name, cv::Mat pts3D, std::vector< cv::Vec3b > color)
{
	std::ofstream f;
	f.open(name + ".ply");
	f << "ply" << std::endl;
	f << "format ascii 1.0" << std::endl;
	f << "element vertex " << pts3D.rows << std::endl;
	f << "property double x" << std::endl;
	f << "property double y" << std::endl;
	f << "property double z" << std::endl;
	f << "property uchar blue" << std::endl;
	f << "property uchar green" << std::endl;
	f << "property uchar red" << std::endl;
	f << "end_header" << std::endl;
	for (int i = 0; i < pts3D.rows; ++i)
	{
		const float* Mi = pts3D.ptr<float>(i);
		if (Mi[0] * Mi[0] + Mi[1] * Mi[1] + Mi[2] * Mi[2] < 2000)
		{
			f << Mi[0] << " " << Mi[1] << " " << Mi[2] << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << std::endl;
		}
		else
		{
			f << 0 << " " << 0 << " " << 0 << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << std::endl;
		}
	}

	f.close();
}