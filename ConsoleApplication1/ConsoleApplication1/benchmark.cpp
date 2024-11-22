#include <iostream>
#include <cstdio>
#include <omp.h>
#include <algorithm> 
#include <iterator> 
#include <numeric> 

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"

#include "RgbToHsv.h"
#include "ImageToBlur.h"
#include "parallelization.cpp"


void benchmarkImage(cv::Mat image, cv::Mat hsvImage, cv::Mat bluredImage, int repetitionsPerTest){
  double t0;

  for(const auto& [testName, func] : parallelizationFunctions) { 
    std::vector<double> runtimes;
    for (int i = 0; i < repetitionsPerTest; i++) {
      t0 = omp_get_wtime(); 
      func(image,hsvImage, bluredImage);
      runtimes.push_back(omp_get_wtime() - t0);
    }
    std::cout << testName << ": " << std::endl;
    std::copy(runtimes.begin(), runtimes.end(), std::ostream_iterator<double>(std::cout, ", ")); std::cout << std::endl; 
    std::cout << "avg:" << 1.0 * std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() << std::endl << std::endl;
  }
}


int main(int argc, char* argv[]) {

  if (argc < 2) {
    std::cerr << "Error: No image paths provided." << std::endl;
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    // read image
    std::string path = argv[i];
    std::cout << "Read image: "<< path << std::endl;

    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    if (image.channels() != 3) {
        std::cerr << "Error: Image has more or less than 3 channels." << std::endl;
        return -1;
    }
	  cv::Mat hsvImage = cv::Mat::zeros(image.size(), image.type());
	  cv::Mat bluredImage = cv::Mat::zeros(image.size(), image.type());
    benchmarkImage(image, hsvImage, bluredImage, 20);
  }
  return 0;
}
