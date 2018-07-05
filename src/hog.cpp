//
// Created by peri on 11/29/17.
//

#include "../header/hog.hpp"
#include "opencv2/opencv.hpp"
#include "../header/patch.hpp"
#include <cmath>

void HoG::gradients(cv::Mat& img, cv::Mat& gx, cv::Mat& gy){

    img.convertTo(img, CV_32F, 1/255.0);
    cv::Sobel(img, gx, CV_32F, 1, 0, 1);
    cv::Sobel(img, gy, CV_32F, 0, 1, 1);

}


void HoG::magnitude_orientation(cv::Mat& gx, cv::Mat& gy, cv::Mat& mag, cv::Mat& angle){
    cv::cartToPolar(gx, gy, mag, angle, true);
}

void HoG::feature(cv::Mat &img, size_t patch, size_t block_size) {

    //cv::Size s = img.size();
    //int rows = s.height;
    //int cols = s.width;
    //std::cout << rows << cols << "   " << img.channels() << std::endl;
    Patch patch_maker;
    std::vector<std::tuple<int, int, size_t, size_t>*> bunch;
    patch_maker.patch_creation(img, patch, patch, int(patch), bunch);
    std::cout << bunch.size() << "Total number of patches in image" << std::endl;
    for (int i=0; i<bunch.size(); i++) {
        cv::Rect roi(std::get<0>(*bunch[i]), std::get<1>(*bunch[i]), std::get<2>(*bunch[i]), std::get<3>(*bunch[i]));
        cv::Mat work_img(img, roi);
        cv::Mat gx, gy, mag, angle;
        //std::string winName= "Test";
        //cv::namedWindow(winName);
        //cv::imshow(winName, work_img);
        //cv::waitKey(0);
        //cv::destroyAllWindows();
        HoG::gradients(work_img, gx, gy);
        HoG::magnitude_orientation(gx, gy, mag, angle);
        //std::vector<float>* normalized_container[img.rows/block_size][img.cols/block_size];
        HoG::final_feature_vector.clear();
        HoG::create_histogram(work_img, mag, angle, block_size);
    }
}

void HoG::create_histogram(cv::Mat& img, cv::Mat& mag, cv::Mat& angle, size_t block_size){

    // Here img is just a patch;
    // As the training set has 32*32 size images we will use 4*4 for block and 8*8 for normalization
    Patch patch_maker;
    std::vector<std::tuple<int, int, size_t, size_t>*> bunch;
    std::vector<float>* normalized_container[img.rows/block_size][img.cols/block_size];
    patch_maker.patch_creation(img, block_size, block_size, int(block_size), bunch);
    //std::cout << bunch.size() << std::endl;
    for (int i=0; i<bunch.size(); i++) {
        assert((img.rows/block_size)*(img.cols/block_size) == bunch.size());
        cv::Rect roi(std::get<0>(*bunch[i]), std::get<1>(*bunch[i]), std::get<2>(*bunch[i]), std::get<3>(*bunch[i]));
        cv::Mat work_img(img, roi);
        cv::Mat work_mag(mag, roi);
        cv::Mat work_angle(angle, roi);
        //cv::Size s = work_angle.size();
        //int rows = s.height;
        //int cols = s.width;
        //std::cout << rows << cols << "    " << work_angle.channels() << std::endl;
        assert(work_mag.rows == work_angle.rows);
        assert(work_img.rows == work_angle.rows);
        assert(work_img.cols == work_angle.cols);
        assert(work_mag.cols == work_angle.cols);
        assert(work_angle.channels() == 3);
        assert(work_mag.channels() == 3);
        //std::cout << "Hello" << std::endl;
        std::vector<float>* unnormalized_feature_vector = new std::vector<float>(9, 0.0);
        //cv::Mat max_mag = cv::Mat::zeros(cv::Size(work_mag.rows, work_mag.cols), CV_32F);
        //cv::Mat max_angle = cv::Mat::zeros(cv::Size(work_angle.rows, work_angle.cols), CV_32F);
        for (int rows_count=0; rows_count < work_mag.rows; rows_count++){
            for (int cols_count=0; cols_count < work_mag.cols; cols_count++){
                //float  r, g, b;
                float r_a, g_a, b_a, r_m, g_m, b_m;
                cv::Vec3f pixel_m = work_mag.at<cv::Vec3f>(rows_count, cols_count);
                cv::Vec3f pixel_a = work_angle.at<cv::Vec3f>(rows_count, cols_count);
                //cv::Vec3f pixel = work_img.at<cv::Vec3f>(rows_count, cols_count);
                b_m = pixel_m.val[0];
                g_m = pixel_m.val[1];
                r_m = pixel_m.val[2];
                b_a = pixel_a.val[0];
                g_a = pixel_a.val[1];
                r_a = pixel_a.val[2];
                //b = pixel.val[0];
                //g = pixel.val[1];
                //r = pixel.val[2];
                float max_m = b_m;
                float max_a = b_a;
                if(g_m > max_m){
                    max_m = g_m;
                    max_a = g_a;
                }
                if(r_m > max_m){
                    max_m = r_m;
                    max_a = r_a;
                }

                if (max_a > 180){
                    max_a = max_a - 180;
                }

                //max_mag.at<float>(rows_count, cols_count) = max_m;
                //max_angle.at<float>(rows_count, cols_count) = max_a;

                //b = pixel.val[0];
                //g = pixel.val[1];
                //r = pixel.val[2];
                //std::cout << r << " " << g << " " << b  << "  Original "<< std::endl;
                //std::cout << r_m << " " << g_m << " " << b_m  << "  Magnitude "<< std::endl;
                //std::cout << r_a << " " << g_a << " " << b_a  << "  Angle "<< std::endl;
                //std::cout << max_a << " " << max_m <<  "  Max "<< std::endl;


                if(max_a >= 0 && max_a < 20){
                    float ratio = (max_a - 0)/20;
                    //std::cout << ratio << "ratio" << std::endl;
                    (*unnormalized_feature_vector)[0] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[1] += max_m*(ratio);
                }
                else if(max_a >= 20 && max_a < 40){
                    float ratio = (max_a - 20)/20;
                    (*unnormalized_feature_vector)[1] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[2] += max_m*(ratio);
                }
                else if(max_a >= 40 && max_a < 60){
                    float ratio = (max_a - 40)/20;
                    (*unnormalized_feature_vector)[2] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[3] += max_m*(ratio);
                }
                else if(max_a >= 60 && max_a < 80){
                    float ratio = (max_a - 60)/20;
                    (*unnormalized_feature_vector)[3] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[4] += max_m*(ratio);
                }
                else if(max_a >= 80 && max_a < 100){
                    float ratio = (max_a - 80)/20;
                    (*unnormalized_feature_vector)[4] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[5] += max_m*(ratio);
                }
                else if(max_a >= 100 && max_a < 120){
                    float ratio = (max_a - 100)/20;
                    (*unnormalized_feature_vector)[5] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[6] += max_m*(ratio);
                }
                else if(max_a >= 120 && max_a < 140){
                    float ratio = (max_a - 120)/20;
                    (*unnormalized_feature_vector)[6] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[7] += max_m*(ratio);
                }
                else if(max_a >= 140 && max_a < 160){
                    float ratio = (max_a - 140)/20;
                    (*unnormalized_feature_vector)[7] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[8] += max_m*(ratio);
                }
                else {
                    float ratio = (max_a - 160)/20;
                    (*unnormalized_feature_vector)[8] += max_m*(1 - ratio);
                    (*unnormalized_feature_vector)[0] += max_m*(ratio);
                }
                //for(int k=0; k<(*unnormalized_feature_vector).size(); k++){
                //    std::cout << (*unnormalized_feature_vector)[k] << ", ";
                //}
                //std::cout << "" << std::endl;


            }


        }

        auto q = int(i/((img.rows/block_size))); // Norm over twice the size
        int r = i%int((img.rows/block_size)); // Norm over twice the size
        //std::cout << q << " " << r << " " << std::endl;
        normalized_container[q][r] = unnormalized_feature_vector;
        //for(int k=0; k<unnormalized_feature_vector.size(); k++){
        //    std::cout << unnormalized_feature_vector[k] << ", ";
        //}

        //std::string winName= "Test";
        //cv::namedWindow(winName);
        //cv::cvtColor(work_img, work_img, cv::COLOR_RGB2BGR);
        //cv::imshow(winName, work_img);
        //cv::waitKey(0);
        //cv::destroyAllWindows();
    }

    // write normalized here
    //std::vector<double> final_feature_vector;
    size_t count_vec = 0;
    for(int i=0; i<(img.rows/block_size)-1; i++){
        for(int j=0; j<(img.cols/block_size)-1; j++){
            double sum_l2 = 0;
            std::vector<float> a = *normalized_container[i][j];
            std::vector<float> b = *normalized_container[i][j+1];
            std::vector<float> c = *normalized_container[i+1][j];
            std::vector<float> d = *normalized_container[i+1][j+1];
            //std::cout << a.size() << b.size() << c.size() << d.size() << std::endl;
            assert(a.size() == b.size());
            assert(b.size() == c.size());
            assert(c.size() == d.size());

            for(int l=0; l<a.size();l++){
                HoG::final_feature_vector.push_back(a[l]);
                sum_l2 += std::pow(a[l], 2);
            }
            for(int m=0; m<b.size();m++){
                HoG::final_feature_vector.push_back(b[m]);
                sum_l2 += std::pow(b[m], 2);
            }
            for(int n=0; n<c.size();n++){
                HoG::final_feature_vector.push_back(c[n]);
                sum_l2 += std::pow(c[n], 2);
            }
            for(int o=0; o<d.size();o++){
                HoG::final_feature_vector.push_back(d[o]);
                sum_l2 += std::pow(d[o], 2);
            }
            sum_l2 = std::sqrt(sum_l2);
            for(size_t k=count_vec; k<count_vec+(9*4); k++) {
                HoG::final_feature_vector[k] = HoG::final_feature_vector[k]/sum_l2;
                //std::cout << HoG::final_feature_vector[k] << ", ";
            }
            //std::cout << "final _feature **************************************" << std::endl;
            count_vec += 9*4;



        }

    }
}
