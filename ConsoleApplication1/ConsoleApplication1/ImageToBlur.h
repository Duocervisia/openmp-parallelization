#ifndef IMAGETOBLUR_H
#define IMAGETOBLUR_H
#include "opencv2/opencv.hpp"

// Function to convert an image
void convertImageToBlur(cv::Mat image, cv::Mat new_image, int i, int j);
void compareImageToBlur(cv::Mat image, cv::Mat new_image);

#endif // IMAGETOBLUR_H