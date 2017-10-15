#pragma once

class Region
{
private:
	std::vector<double> pixelValues;
	double prom;
	int r;
	int c;
	int x;
	int y;
	double mod;
	double mod2;
public:
	Region(cv::Mat image, int row, int column, int xSize = 1, int ySize = 2);
	cv::Point2f center();
	int dotProduct(Region);
	double getMod();
	int getMod2();
	int modulo2();
	double modulo();
	double nCorr(Region);
	void printPixelValues();
	void draw(cv::Mat &image, cv::Scalar color);
};