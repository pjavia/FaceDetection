//
// Created by peri on 7/4/18.
//

#include "../header/patch.hpp"
#include <opencv2/opencv.hpp>


void Patch::patch_creation(cv::Mat& img, size_t w, size_t h, int stride, std::vector<std::tuple<int, int, size_t, size_t>*>& bunch){

    for(int i = 0; i<=img.cols - w; i=i+stride){
        for(int j = 0; j<=img.rows - h; j=j+stride){
            std::tuple<int, int, size_t, size_t>* cropped;
            cropped = new std::tuple<int, int, size_t, size_t>;
            *cropped = std::make_tuple(i, j, w, h);
            bunch.push_back(cropped);
        }
    }
}


void Patch::patch_view(cv::Mat& img, std::vector<std::tuple<int, int, size_t, size_t>*>& bunch){

    for (int i=0; i<bunch.size(); i++) {
        cv::Rect roi(std::get<0>(*bunch[i]), std::get<1>(*bunch[i]), std::get<2>(*bunch[i]), std::get<3>(*bunch[i]));
        cv::Mat display_img(img, roi);
        std::string winName= "Test";
        cv::namedWindow(winName);
        cv::cvtColor(display_img, display_img, cv::COLOR_RGB2BGR);
        cv::imshow(winName, display_img);
        cv::waitKey(0);
        cv::destroyAllWindows();

    }
}