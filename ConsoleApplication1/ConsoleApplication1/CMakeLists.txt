cmake_minimum_required(VERSION 2.8)
project( openmp-parallelization )

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(lib ImageToBlur.cpp RgbToHsv.cpp ImageHelper.cpp Parallelization.cpp)

add_executable( openmp-parallelization ConsoleApplication1.cpp)
target_link_libraries( openmp-parallelization lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX )

add_executable( openmp-parallelization-benchmark Benchmark.cpp)
target_link_libraries( openmp-parallelization-benchmark lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX )

