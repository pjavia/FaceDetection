//
// Created by peri
//

#ifndef FACEDETECTION_HOG_HPP
#define FACEDETECTION_HOG_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <jpeglib.h>
//#include <boost/python/numpy/ndarray.hpp>
#include <opencv2/opencv.hpp>


using namespace boost::numeric::ublas;

class HoG{

public:
    std::vector<double> final_feature_vector;

    void gradients(cv::Mat& img, cv::Mat& gx, cv::Mat& gy);

    void magnitude_orientation(cv::Mat& gx, cv::Mat& gy, cv::Mat& mag, cv::Mat& angle);

public:
    void feature(cv::Mat& img, size_t patch, size_t block_size);

    void create_histogram(cv::Mat& img, cv::Mat& mag, cv::Mat& angle, size_t block_size);

public:
    void GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);

};

#endif //FACEDETECTION_HOG_HPP
