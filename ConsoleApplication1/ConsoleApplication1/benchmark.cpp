#include <iostream>
#include <cstdio>
#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"

#include "RgbToHsv.h"
#include "ImageToBlur.h"


// Forward declaration of showDiff function
void showDiff(cv::Mat diff);

int main(int argc, char* argv[]) {
  double t0;

   //read path
   if (argc < 2) {
        std::cerr << "Error: No image path provided." << std::endl;
        return 1;
    }
   std::string path = argv[1];

    // read image
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
  
    //print image channels dynamically
   	printf("Image has %d channels\n", image.channels());

    int variant = 0;
    if (argc > 2) {
        try {
            variant = std::stoi(argv[2]);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid variant argument. Must be an integer." << std::endl;
        }
    }

    // display and wait for a key-press, then close the window
    cv::imshow("image", image);
    int key = cv::waitKey(0);
    cv::destroyAllWindows();

	cv::Mat hsvImage = cv::Mat::zeros(image.size(),CV_8UC3);
    t0 = omp_get_wtime(); // start time
    imageToHsv(image, hsvImage);
    std::cout << "No parallel processing took " << (omp_get_wtime() - t0) << " seconds" << std::endl;
    t0 = omp_get_wtime(); // start time
    imageToHsvParallelOuter(image, hsvImage);
    std::cout << "Outer loop parallel processing took " << (omp_get_wtime() - t0) << " seconds" << std::endl;
    t0 = omp_get_wtime(); // start time
    imageToHsvParallelInner(image, hsvImage);
    std::cout << "Inner loop parallel processing took " << (omp_get_wtime() - t0) << " seconds" << std::endl;
	// cv::Mat rgbImage = cv::Mat::zeros(image.size(), image.type());
    // cv::cvtColor(hsvImage, rgbImage,cv::COLOR_HSV2BGR);
    // cv::subtract(rgbImage, image, rgbImage);

    cv::imshow("image", hsvImage);
    key = cv::waitKey(0);
    cv::destroyAllWindows();

    //init new_image
	cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());

    t0 = omp_get_wtime(); // start time

    switch (variant) {
        case 0:
            //Doppelte for-Schleife; Parallelisierung außen
            #pragma omp parallel for
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    convertImageToBlur(image, new_image, i, j);
                }
            }
            break;
        case 1:
            //Doppelte for-Schleife; Parallelisierung innen
            for (int i = 0; i < image.rows; ++i) {
                #pragma omp parallel for
                for (int j = 0; j < image.cols; ++j) {
                    convertImageToBlur(image, new_image, i, j);
                }
            }
            break;
        case 2:
            //Vereinte for-Schleife
            #pragma omp parallel for
            for (int index = 0; index < image.rows * image.cols; ++index) {
                int i = index / image.cols; // Zeile
                int j = index % image.cols; // Spalte

                convertImageToBlur(image, new_image, i, j);
            }
            break;
    }

    double t1 = omp_get_wtime();  // end time

    std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;

    // display and wait for a key-press, then close the window
    cv::imshow("image", new_image);
    key = cv::waitKey(0);

	compareImageToBlur(image, new_image);

    cv::destroyAllWindows();
}
