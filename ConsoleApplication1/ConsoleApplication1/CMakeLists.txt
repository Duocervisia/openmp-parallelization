cmake_minimum_required(VERSION 2.8)
project( openmp-parallelization )
find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( openmp-parallelization ConsoleApplication1.cpp ImageToBlur.cpp RgbToHsv.cpp)
target_link_libraries( openmp-parallelization ${OpenCV_LIBS} OpenMP::OpenMP_CXX )

