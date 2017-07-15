#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
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