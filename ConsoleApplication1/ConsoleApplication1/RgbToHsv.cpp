#include <algorithm>
#include <iostream>
#include <opencv4/opencv2/core/matx.hpp>
//#include <opencv2/core/matx.hpp>
#include "RgbToHsv.h"

double normalize(uchar value){
  return value/255.0;
}

double getHue(double cmax, double cmin,double diff, double r, double g, double b){
  if(cmax == cmin) return 0.0;
  if(cmax == r) return fmod(((60.0 * ((g - b) / diff)) + 360.0), 360.0);
  if(cmax == g) return fmod(((60.0 * ((b - r) / diff)) + 120.0), 360.0);
  return fmod(((60.0 * ((r - g) / diff)) + 240.0), 360.0);
}

void pixelToHsv(cv::Vec3b& orgPixel,cv::Vec3b& newPixel) {
  // extract the pixels as normalized rgb values
  double b = normalize(orgPixel[0]);
  double g = normalize(orgPixel[1]);
  double r = normalize(orgPixel[2]);

  double cmax = std::max({r, g,b});
  double cmin = std::min({r, g, b});
  double diff = cmax - cmin;

  double h = getHue(cmax, cmin, diff, r, g, b);

  double s = (cmax == 0.0) ? 0.0 : (diff / cmax) * 255.0;
  double v = cmax * 255.0;

  newPixel[0] = (int)(std::round(h)/2.0);
  newPixel[1] = (int)std::round(s);
  newPixel[2] = (int)std::round(v);
}

// Static function with internal linkage
void imageToHsv(cv::Mat image, cv::Mat outputImage) {
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      pixelToHsv(image.at<cv::Vec3b>(i, j),outputImage.at<cv::Vec3b>(i, j));
    }
  }
}
void imageToHsvParallelOuter(cv::Mat image, cv::Mat outputImage) {
#pragma omp parallel for
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      pixelToHsv(image.at<cv::Vec3b>(i, j),outputImage.at<cv::Vec3b>(i, j));
    }
  }
}

void imageToHsvParallelInner(cv::Mat image, cv::Mat outputImage) {
  for (int i = 0; i < image.rows; ++i) {
    #pragma omp parallel for
    for (int j = 0; j < image.cols; ++j) {
      pixelToHsv(image.at<cv::Vec3b>(i, j),outputImage.at<cv::Vec3b>(i, j));
    }
  }
}
