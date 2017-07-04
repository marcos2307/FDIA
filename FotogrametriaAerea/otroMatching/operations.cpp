#include <opencv2\core\core.hpp>
#include "operations.h"

bool comp(cv::DMatch A, cv::DMatch B)
{
	return A.distance < B.distance;
}