#pragma once

class Region
{
	private:
		std::vector<int> pixelValues;
		int r;
		int c;
	public:
		Region(cv::Mat image, int row, int column, int xSize = 1, int ySize = 4);
		int dotProduct(Region);
		int modulo();
		cv::Point2f center();
};