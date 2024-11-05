#include <iostream>
#include "ImageToBlur.h"
#include "opencv2/opencv.hpp"
#include "ImageHelper.h"

/*
 * Function to convert an image
 * @param image The image to convert
 * @param i The row index
 * @param j The column index
 */
void convertImageToBlur(cv::Mat image, cv::Mat new_image, int i, int j) {

    if (i < 1 || j < 1 || i >= image.rows - 1 || j >= image.cols - 1) {
        for (int k = 0; k < image.channels(); ++k) {
            new_image.ptr<unsigned char>(i, j)[k] = image.ptr<unsigned char>(i, j)[k];
        }
        return;
    }

    std::vector<int> sum(image.channels(), 0);

    for (int k = 0; k < image.channels(); ++k) {
        sum[k] += image.ptr<unsigned char>(i - 1, j + 1)[k];
        sum[k] += image.ptr<unsigned char>(i, j + 1)[k];
        sum[k] += image.ptr<unsigned char>(i + 1, j + 1)[k];
        sum[k] += image.ptr<unsigned char>(i - 1, j)[k];
        sum[k] += image.ptr<unsigned char>(i, j)[k];
        sum[k] += image.ptr<unsigned char>(i + 1, j)[k];
        sum[k] += image.ptr<unsigned char>(i - 1, j - 1)[k];
        sum[k] += image.ptr<unsigned char>(i, j - 1)[k];
        sum[k] += image.ptr<unsigned char>(i + 1, j - 1)[k];

        new_image.ptr<unsigned char>(i, j)[k] = sum[k] / 9;
    }
}

void compareImageToBlur(cv::Mat image, cv::Mat new_image) {
	// Create a kernel for box blur
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;

	// Apply the box blur using filter2D
	cv::Mat dst;
	cv::filter2D(image, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

	// Calculate the difference between the two images
	cv::Mat diff = new_image - dst;

	// Show the difference
	showDiff(diff);
}
