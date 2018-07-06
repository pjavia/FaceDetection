//
// Created by peri
//

#ifndef FACEDETECTION_PATCH_HPP
#define FACEDETECTION_PATCH_HPP
#include <opencv2/opencv.hpp>

class Patch {

public:
    void patch_creation(cv::Mat& img, size_t x, size_t y, int stride, std::vector<std::tuple<int, int, size_t, size_t>*>& bunch);

public:
    void patch_view(cv::Mat& img, std::vector<std::tuple<int, int, size_t, size_t>*> &bunch);

};



#endif //FACEDETECTION_PATCH_HPP
