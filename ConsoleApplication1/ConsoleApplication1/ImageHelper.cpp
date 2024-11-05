#include <iostream>
#include "ImageToBlur.h"
#include "opencv2/opencv.hpp"

void showDiff(cv::Mat diff) {

	// Calculate the norm of the difference
	double diff_value = cv::norm(diff, cv::NORM_L2);
	std::cout << "Difference value (L2 norm): " << diff_value << std::endl;

	//where the difference is not zero, set the pixel to white
	for (int i = 0; i < diff.rows; ++i) {
		for (int j = 0; j < diff.cols; ++j) {
			if (diff.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				diff.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	// display and wait for a key - press, then close the window
	cv::imshow("image", diff);
	int key = cv::waitKey(0);
	cv::destroyAllWindows();
}