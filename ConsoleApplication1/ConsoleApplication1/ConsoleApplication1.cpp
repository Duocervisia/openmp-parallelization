#include <iostream>
#include <cstdio>
#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "ImageHelper.h"
#include "RgbToHsv.h"
#include "ImageToBlur.h"

#include "opencv2/opencv.hpp"
#include "parallelization.cpp"

int main(int argc, char* argv[]) {
    //read path
    if (argc < 2) {
        std::cerr << "Error: No image path provided." << std::endl;
        return 1;
    }

    // read image
    cv::Mat org_image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (org_image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    if (org_image.channels() != 3) {
        std::cerr << "Error: Image has more or less than 3 channels." << std::endl;
        return -1;
    }

    int variant = 0;
    if (argc > 2) {
        try {
            variant = std::stoi(argv[2]);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid variant argument. Must be an integer." << std::endl;
        }
    }

	  cv::Mat blured_image = cv::Mat::zeros(org_image.size(), org_image.type());
	  cv::Mat hsv_image = cv::Mat::zeros(org_image.size(),CV_8UC3);

  std::cout << parallelizationFunctions.size() << std::endl;

    switch (variant) {
        case 0:
            //Doppelte for-Schleife; Parallelisierung außen
            parallelizationFunctions.at("outerParallelization")(org_image,hsv_image,blured_image);
            break;
        case 1:
            //Doppelte for-Schleife; Parallelisierung innen
            parallelizationFunctions.at("innerParallelization")(org_image,hsv_image,blured_image);
            break;
        case 2:
            //Vereinte for-Schleife
            parallelizationFunctions.at("oneLoopParallelization")(org_image,hsv_image,blured_image);
            break;
    }

    // display and wait for a key-press, then close the window
    cv::imshow("origanal image", org_image);
    cv::waitKey(0);
	  cv::destroyAllWindows();

    cv::imshow("blured image", blured_image);
    cv::waitKey(0);
	  cv::destroyAllWindows();
	  compareImageToBlur(org_image, blured_image);

    cv::imshow("hsv image", hsv_image);
    cv::waitKey(0);
	  cv::destroyAllWindows();

	  cv::Mat hsv_compare_image = cv::Mat::zeros(org_image.size(), org_image.type());
    cv::cvtColor(org_image, hsv_compare_image,cv::COLOR_BGR2HSV);
    cv::subtract(hsv_compare_image, hsv_image, hsv_compare_image);
    showDiff(hsv_compare_image);
}
