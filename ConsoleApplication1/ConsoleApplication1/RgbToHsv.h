#include <opencv2/opencv.hpp>
#include <string>

#ifndef RGBTOHSV_H
#define RGBTOHSV_H

// Function to convert an image
void imageToHsv(cv::Mat image, cv::Mat outputImage);
void imageToHsvParallelInner(cv::Mat image, cv::Mat outputImage);
void imageToHsvParallelOuter(cv::Mat image, cv::Mat outputImage);
void pixelToHsv(cv::Vec3b& orgPixel,cv::Vec3b& newPixel);

#endif // RGBTOHSV_H
