#include <iostream>
#include "ImageToBlur.h"
#include "opencv2/opencv.hpp"


/*
 * Function to convert an image
 * @param image The image to convert
 * @param i The row indei
 * @param j The column indei
 */
void convertImageToBlur(cv::Mat image, cv::Mat new_image, int i, int j) {

    // Check if the kernel fits
    if (i < 1 || j < 1 || i >= image.rows - 1 || j >= image.cols - 1) {
        // For edge cases, copy the original pixel value
        new_image.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j);
        return;
    }

    // Calculate the sum of the 3i3 kernel
    cv::Vec3i sum = cv::Vec3i(image.at<cv::Vec3b>(i - 1, j + 1)) + // Top left
        cv::Vec3i(image.at<cv::Vec3b>(i + 0, j + 1)) + // Top center
        cv::Vec3i(image.at<cv::Vec3b>(i + 1, j + 1)) + // Top right
        cv::Vec3i(image.at<cv::Vec3b>(i - 1, j + 0)) + // Mid left
        cv::Vec3i(image.at<cv::Vec3b>(i + 0, j + 0)) + // Current pixel
        cv::Vec3i(image.at<cv::Vec3b>(i + 1, j + 0)) + // Mid right
        cv::Vec3i(image.at<cv::Vec3b>(i - 1, j - 1)) + // Low left
        cv::Vec3i(image.at<cv::Vec3b>(i + 0, j - 1)) + // Low center
        cv::Vec3i(image.at<cv::Vec3b>(i + 1, j - 1));  // Low right

    cv::Vec3b average;
    average[0] = cv::saturate_cast<uchar>(sum[0] / 9);
    average[1] = cv::saturate_cast<uchar>(sum[1] / 9);
    average[2] = cv::saturate_cast<uchar>(sum[2] / 9);

    
 //   cv::Vec3b original = image.at<cv::Vec3b>(i + 0, j + 0);
	//printf("Original Pixel of the 3x3 kernel: %d, %d, %d\n", original[0], original[1], original[2]);
	//printf("Sum of the 3x3 kernel: %d, %d, %d\n", sum[0], sum[1], sum[2]);
	//printf("Average of the 3x3 kernel: %d, %d, %d\n", average[0], average[1], average[2]);

    new_image.at<cv::Vec3b>(i, j) = average;
}
