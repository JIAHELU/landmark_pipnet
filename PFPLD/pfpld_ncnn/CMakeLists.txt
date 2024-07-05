cmake_minimum_required (VERSION 2.6)

project (pfpld_ncnn)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(OpenCV_DIR "/home/bill/D/Opencv/opencv-3.4.5/build")
find_package( OpenCV REQUIRED )

include_directories(/home/bill/E/dl_base/ncnn_new/ncnn_11_27/ncnn/src
        /home/bill/E/dl_base/ncnn_new/ncnn_11_27/ncnn/build/src)
link_directories(/home/bill/E/dl_base/ncnn_new/ncnn_11_27/ncnn/build/src)
#include_directories(/home/bill/E/dl_base/ncnn_new/ncnn_20_6_28/src


#set_target_properties(pfpld_ncnn PROPERTIES CXX_STANDARD 11)


#add_executable(pfpld_ncnn src/anchor_generator.cpp  src/config.cpp  src/detect.cpp  src/tools.cpp
#        src/anchor_generator.h src/config.h src/tools.h src/pfpld.id.h)
add_executable(pfpld_ncnn src/anchor_generator.cpp  src/config.cpp  src/detect_track.cpp  src/tools.cpp
               src/anchor_generator.h src/config.h src/tools.h src/pfpld.id.h src/Hungarian.h src/Hungarian.cpp src/Tracker.h src/Tracker.cpp)
#include_directories(./ncnn/include)
#add_library(ncnn STATIC IMPORTED)
#set_property(TARGET ncnn PROPERTY IMPORTED_LOCATION ../ncnn/libncnn.a)
target_link_libraries(pfpld_ncnn ncnn)
target_link_libraries(pfpld_ncnn ${OpenCV_LIBS} )
