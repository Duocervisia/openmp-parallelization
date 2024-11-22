#include <omp.h>
#include "RgbToHsv.h"
#include "ImageToBlur.h"

static const std::map<std::string, void(*)(cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage)> parallelizationFunctions = {
  {"noParallelization", [](cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage) { 
    for (int i = 0; i < image.rows; ++i) {
      for (int j = 0; j < image.cols; ++j) {
        convertImageToBlur(image, bluredImage, i, j);
        pixelToHsv(image.at<cv::Vec3b>(i, j),hsvImage.at<cv::Vec3b>(i, j));
      }
    }
  }},
  {"outerParallelization", [](cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage) { 
    #pragma omp parallel for
    for (int i = 0; i < image.rows; ++i) {
      for (int j = 0; j < image.cols; ++j) {
        convertImageToBlur(image, bluredImage, i, j);
        pixelToHsv(image.at<cv::Vec3b>(i, j),hsvImage.at<cv::Vec3b>(i, j));
      }
    }
  }},
  {"innerParallelization", [](cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage) { 
    for (int i = 0; i < image.rows; ++i) {
      #pragma omp parallel for
      for (int j = 0; j < image.cols; ++j) {
        convertImageToBlur(image, bluredImage, i, j);
        pixelToHsv(image.at<cv::Vec3b>(i, j),hsvImage.at<cv::Vec3b>(i, j));
      }
    }
  }},
  {"oneLoopParallelization", [](cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage) { 
    #pragma omp parallel for
    for (int index = 0; index < image.rows * image.cols; ++index) {
      int i = index / image.cols; 
      int j = index % image.cols;
      convertImageToBlur(image, bluredImage, i, j);
      pixelToHsv(image.at<cv::Vec3b>(i, j),hsvImage.at<cv::Vec3b>(i, j));
    }
  }}
};
