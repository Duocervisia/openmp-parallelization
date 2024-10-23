//#include <iostream>
//#include <omp.h>
//#include <opencv2/opencv.hpp>

//int main()
//{
//    int sum = 0;
//    printf("Sum = %d\n", sum);
//
//    double time_start = omp_get_wtime();
//
//	int width = 10;
//	int height = 10;
//
//    #pragma omp parallel for collapse(2) #Ab version 2.x
//
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			printf("i = %d, j = %d\n", i, j);
//		}
//	}
//
//	int j = 0;
//
//	//Testen was schneller ist (eine for Schleife parallelisieren, zwei ineinander verschachteln, mit collapse und wie hier drunter)
//
//	#pragma omp parallel for
//	for (int i = 0; i < width * height; i++)
//	{
//		int j = i % width;
//		int i2 = i - (j * width);
//	}
//
//
//    #pragma omp parallel for
//    for (int i = 0; i < 100000000; i++)
//    {
//        #pragma omp atomic
//        sum++;
//        
//    }
//	printf("Sum = %d\n", sum);
//
//
//    double time_end = omp_get_wtime();
//    double time_diff = time_end - time_start;
//    printf("Program worked for %f seconds\n", time_diff);
//}

//int main() {
//	printf("Hello World\n");
//
//
//	//channels 4 = rgba, 3 = rgb, 1 = grayscale
//	if (image.channels() == 1) {
//		printf("Image has 1 channel\n");
//	}
//	else if (image.channels() == 3) {
//		printf("Image has 3 channels\n");
//	}
//	else {
//		printf("Image has %d channels\n", image.channels());
//	}
//
//	//Überprüfung der converteirung mit hsv konvertierung von cv selbst
//	cv::Mat hsv_image;
//	cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
//
//	cv::Mat diff = new_image - new_image2
//
//		//Für Bericht, überprüfung der Abweichungen integrieren
//		//Laufzeit gegenüber anzahl der Threads als Graphen darstellen
//}

#include <iostream>
#include <cstdio>
#include <omp.h>
#include "RgbToHsv.h"
#include "ImageToBlur.h"

#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]) {
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

    int variant = 0;
    if (argc > 2) {
        try {
            variant = std::stoi(argv[2]);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid variant argument. Must be an integer." << std::endl;
        }
    }


    //init new_image
	cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());

    //print image channels dynamically
	printf("Image has %d channels\n", image.channels());

    // display and wait for a key-press, then close the window
    cv::imshow("image", image);
    int key = cv::waitKey(0);
    cv::destroyAllWindows();

    double t0 = omp_get_wtime(); // start time

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
    cv::destroyAllWindows();
}
