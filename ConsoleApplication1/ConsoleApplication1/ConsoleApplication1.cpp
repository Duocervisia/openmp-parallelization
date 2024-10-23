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

int main(int argc, char** argv)
{
    // read image
    cv::Mat image = cv::imread("C:\\top.jpg", cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    // display and wait for a key-press, then close the window
    cv::imshow("image", image);
    int key = cv::waitKey(0);
    cv::destroyAllWindows();

    internalImageConversion();
    convertImageToBlur();

    double t0 = omp_get_wtime(); // start time

    #pragma omp parallel for
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {

            // get pixel at [i, j] as a <B,G,R> vector
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);

            // extract the pixels as uchar (unsigned 8-bit) types (0..255)
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];

            // Note: this is actually the slowest way to extract a pixel in OpenCV
            // Using pointers like this:
            //   uchar* ptr = (uchar*) image.data; // get raw pointer to the image data
            //   ...
            //   for (...) { 
            //       uchar* pixel = ptr + image.channels() * (i * image.cols + j);
            //       uchar b = *(pixel + 0); // Blue
            //       uchar g = *(pixel + 1); // Green
            //       uchar r = *(pixel + 2); // Red
            //       uchar a = *(pixel + 3); // (optional) if there is an Alpha channel
            //   }
            // is much faster

            uchar temp = r;
            r = b;
            b = temp;

            image.at<cv::Vec3b>(i, j) = pixel;
            // or: 
            // image.at<cv::Vec3b>( i, j ) = {r, g, b};
        }
    }
    double t1 = omp_get_wtime();  // end time

    std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;

    // display and wait for a key-press, then close the window
    cv::imshow("image", image);
    key = cv::waitKey(0);
    cv::destroyAllWindows();
}