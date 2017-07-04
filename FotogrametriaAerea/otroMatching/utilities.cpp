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

void filterMatches(std::vector<cv::KeyPoint>& keyPoints1, std::vector<cv::KeyPoint>& keyPoints2, std::vector<cv::DMatch>& matches, int columns, double tolerance)
{

	double promAngle = atan2((keyPoints2.at(matches.at(0).trainIdx).pt.y - keyPoints1.at(matches.at(0).queryIdx).pt.y), (keyPoints2.at(matches.at(0).trainIdx).pt.x + columns - keyPoints1.at(matches.at(0).queryIdx).pt.y));
	double acumAngle = promAngle;
	std::cout << "\nAngulo primer match: " << promAngle;

	double promNorm = abs(keyPoints2.at(matches.at(0).trainIdx).pt.y - keyPoints1.at(matches.at(0).queryIdx).pt.y) + abs(keyPoints2.at(matches.at(0).trainIdx).pt.x + columns - keyPoints1.at(matches.at(0).queryIdx).pt.y);
	double acumNorm = promNorm;
	std::cout << "\nDistancia primer match: " << promNorm;

	if (promAngle < 0) tolerance = tolerance*(-1);

	for (int i = 1; i < matches.size(); ++i)
	{
		double actAngle = atan2((keyPoints2.at(matches.at(i).trainIdx).pt.y - keyPoints1.at(matches.at(i).queryIdx).pt.y), (keyPoints2.at(matches.at(i).trainIdx).pt.x + columns - keyPoints1.at(matches.at(i).queryIdx).pt.y));
		double actNorm = abs(keyPoints2.at(matches.at(i).trainIdx).pt.y - keyPoints1.at(matches.at(i).queryIdx).pt.y) + abs(keyPoints2.at(matches.at(i).trainIdx).pt.x + columns - keyPoints1.at(matches.at(i).queryIdx).pt.y);
		std::cout << "\n\n" << actAngle;
		std::cout << "\n" << actNorm;
		if (actAngle > promAngle*(1.0f + tolerance) || actAngle < promAngle*(1.0f - tolerance) || actNorm > promNorm*(1.0f + abs(tolerance)) || actNorm < promNorm*(1.0f - abs(tolerance)))
		{
			matches.erase(matches.begin() + i);
			--i;
		}
		else
		{
			acumAngle += actAngle;
			promAngle = acumAngle / (i + 1);
			acumNorm += actNorm;
			promNorm = acumNorm / (i + 1);
		}
	}
	std::cout << "\n\nAngulo promedio: " << promAngle << "\n";
	std::cout << "\nDistancia promedio: " << promNorm << "\n";
}