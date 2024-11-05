#include <iostream>
#include "ImageToBlur.h"
#include "opencv2/opencv.hpp"
#include "ImageHelper.h"



/*
 * Function to convert an image
 * @param image The image to convert
 * @param i The row indei
 * @param j The column indei
 */
void convertImageToBlur(cv::Mat image, cv::Mat new_image, int i, int j) {

    if (i < 1 || j < 1 || i >= image.rows - 1 || j >= image.cols - 1) {
        switch (image.channels()) {
            case 1:
                new_image.at<uchar>(i, j) = image.at<uchar>(i, j);
				break;
			case 3:
				new_image.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j);
				break;
			case 4:
				new_image.at<cv::Vec4b>(i, j) = image.at<cv::Vec4b>(i, j);
				break;
			default:
				throw std::runtime_error("Unsupported number of image channels");
        }
        return;
    }

    switch (image.channels()) {
        case 1: { // Grayscale image
            int sum = image.at<uchar>(i - 1, j + 1) +
                image.at<uchar>(i, j + 1) +
                image.at<uchar>(i + 1, j + 1) +
                image.at<uchar>(i - 1, j) +
                image.at<uchar>(i, j) +
                image.at<uchar>(i + 1, j) +
                image.at<uchar>(i - 1, j - 1) +
                image.at<uchar>(i, j - 1) +
                image.at<uchar>(i + 1, j - 1);

            uchar average = cv::saturate_cast<uchar>(sum / 9);
            new_image.at<uchar>(i, j) = average;
            break;
        }
        case 3: { // RGB image
            cv::Vec3i sum = cv::Vec3i(image.at<cv::Vec3b>(i - 1, j + 1)) +
                cv::Vec3i(image.at<cv::Vec3b>(i, j + 1)) +
                cv::Vec3i(image.at<cv::Vec3b>(i + 1, j + 1)) +
                cv::Vec3i(image.at<cv::Vec3b>(i - 1, j)) +
                cv::Vec3i(image.at<cv::Vec3b>(i, j)) +
                cv::Vec3i(image.at<cv::Vec3b>(i + 1, j)) +
                cv::Vec3i(image.at<cv::Vec3b>(i - 1, j - 1)) +
                cv::Vec3i(image.at<cv::Vec3b>(i, j - 1)) +
                cv::Vec3i(image.at<cv::Vec3b>(i + 1, j - 1));

            cv::Vec3b average;
            average[0] = cv::saturate_cast<uchar>(sum[0] / 9);
            average[1] = cv::saturate_cast<uchar>(sum[1] / 9);
            average[2] = cv::saturate_cast<uchar>(sum[2] / 9);

            new_image.at<cv::Vec3b>(i, j) = average;
            break;
        }
        case 4: { // RGBA image
            cv::Vec4i sum = cv::Vec4i(image.at<cv::Vec4b>(i - 1, j + 1)) +
                cv::Vec4i(image.at<cv::Vec4b>(i, j + 1)) +
                cv::Vec4i(image.at<cv::Vec4b>(i + 1, j + 1)) +
                cv::Vec4i(image.at<cv::Vec4b>(i - 1, j)) +
                cv::Vec4i(image.at<cv::Vec4b>(i, j)) +
                cv::Vec4i(image.at<cv::Vec4b>(i + 1, j)) +
                cv::Vec4i(image.at<cv::Vec4b>(i - 1, j - 1)) +
                cv::Vec4i(image.at<cv::Vec4b>(i, j - 1)) +
                cv::Vec4i(image.at<cv::Vec4b>(i + 1, j - 1));

            cv::Vec4b average;
            average[0] = cv::saturate_cast<uchar>(sum[0] / 9);
            average[1] = cv::saturate_cast<uchar>(sum[1] / 9);
            average[2] = cv::saturate_cast<uchar>(sum[2] / 9);
            average[3] = cv::saturate_cast<uchar>(sum[3] / 9);

            new_image.at<cv::Vec4b>(i, j) = average;
            break;
        }
        default:
            // Handle unexpected number of channels
            throw std::runtime_error("Unsupported number of image channels");
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
