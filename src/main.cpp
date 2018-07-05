// Created by Peri

#include <iostream>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "../header/hog.hpp"
#include <boost/gil/extension/io/jpeg_io.hpp>

#include "../header/patch.hpp"

int main(){


    std::cout << "Hello World" << std::endl;
    HoG hog;
    std::ifstream train_face_file ("/home/peri/Downloads/proj5/data/train_face.csv");
    std::ifstream train_non_face_file ("/home/peri/Downloads/proj5/data/train_non_face.csv");
    std::string filename;
    std::vector<std::vector<double>> train_set_vector;
    std::vector<float> train_set_label_vector;

    while (train_face_file.good()) {

        std::getline(train_face_file, filename, '\n');
        //std::cout << filename << std::endl;
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        if (!image.data)
        {
            std::cout << "Image not loaded";
            return -1;
        }
        std::cout << image.size() << "Image size" <<std::endl;
        std::cout << image.channels() << "Channels " <<std::endl;
        cv::resize(image, image, cv::Size(32, 32));
        hog.feature(image, 32, 4);
        std::cout << hog.final_feature_vector.size() << "Feature vector size" << std::endl;
        train_set_vector.push_back(hog.final_feature_vector);
        train_set_label_vector.push_back(1);


    }


    while (train_non_face_file.good()) {

        std::getline(train_non_face_file, filename, '\n');
        std::cout << filename << std::endl;
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data)
        {
            std::cout << "Image not loaded";
            return -1;
        }

        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::Rect roi1(0, 0, 32, 32);
        cv::Rect roi2(0, 5, 32, 32);
        cv::Rect roi3(5, 5, 32, 32);
        cv::Mat image1(image, roi1);
        cv::Mat image2(image, roi2);
        cv::Mat image3(image, roi3);
        cv::resize(image1, image1, cv::Size(32, 32));
        cv::resize(image2, image2, cv::Size(32, 32));
        cv::resize(image3, image3, cv::Size(32, 32));
        hog.feature(image1, 32, 4);
        train_set_vector.push_back(hog.final_feature_vector);
        //std::cout << image1.size() << "Image size" <<std::endl;
        //std::cout << image1.channels() << "Channels " <<std::endl;
        std::cout << hog.final_feature_vector.size() << "Feature vector size" << std::endl;
        hog.feature(image2, 32, 4);
        train_set_vector.push_back(hog.final_feature_vector);
        //std::cout << image2.size() << "Image size" <<std::endl;
        //std::cout << image2.channels() << "Channels " <<std::endl;
        std::cout << hog.final_feature_vector.size() << "Feature vector size" << std::endl;
        hog.feature(image3, 32, 4);
        train_set_vector.push_back(hog.final_feature_vector);
        //std::cout << image3.size() << "Image size" <<std::endl;
        //std::cout << image3.channels() << "Channels " <<std::endl;
        std::cout << hog.final_feature_vector.size() << "Feature vector size" << std::endl;
        train_set_label_vector.push_back(-1);
        train_set_label_vector.push_back(-1);
        train_set_label_vector.push_back(-1);
    }

    std::cout << train_set_vector[0].size() <<std::endl ;
    std::cout << train_set_label_vector.size() <<std::endl;

    cv::Mat training_set(int(train_set_vector.size()), 1764, CV_32F);
    cv::Mat training_set_labels(int(train_set_label_vector.size()), 1, CV_32S);


    for(int i=0; i<training_set.rows; i++) {
        for (int j = 0; j < training_set.cols; j++) {
            training_set.at<float>(i, j) = train_set_vector[i][j];
            //std::cout << training_set.at<float>(i, j) << "M";
            //std::cout << train_set_vector[i][j] << "V ";
        }
        std::cout << "Vector ..........." << std::endl;
    }



    for(int i=0; i<training_set_labels.rows; i++) {
        for (int j = 0; j < training_set_labels.cols; j++) {
            training_set_labels.at<float>(i, j) = train_set_label_vector[i];
            std::cout << training_set_labels.at<float>(i, j) << std::endl;
        }
    }

    training_set.convertTo(training_set, CV_32F);
    std::cout << training_set.size() << std::endl;
    std::cout << training_set_labels.size() << std::endl;

    // Set up SVM's parameters
    std::cout << "Training Start" << std::endl;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    svm->setGamma(3);
    svm->setDegree(3);
    svm->train(training_set, cv::ml::ROW_SAMPLE, training_set_labels);

    //cv::Ptr<cv::ml::LogisticRegression> lr1 = cv::ml::LogisticRegression::create();
    //lr1->setLearningRate(0.001);
    //lr1->setIterations(10);
    //lr1->setRegularization(cv::ml::LogisticRegression::REG_L2);
    //lr1->setTrainMethod(cv::ml::LogisticRegression::BATCH);
    //lr1->setMiniBatchSize(1);
    //lr1->train(training_set, cv::ml::ROW_SAMPLE, training_set_labels);

    //cv::Mat image = cv::imread("/home/peri/Documents/Computer Vision/faceDetection/src/gtech.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    //cv::Mat image = cv::imread("/home/peri/Downloads/proj5/data/caltech_faces/Caltech_CropFaces/caltech_web_crop_10524.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image = cv::imread("/home/peri/Downloads/proj5/data/test_scenes/test_jpg/boat.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    if (!image.data)
    {
        std::cout << "Image not loaded";
        return -1;
    }
    std::cout << image.size() << "Image size" <<std::endl;
    std::cout << image.channels() << "Channels " <<std::endl;
    //cv::resize(image, image, cv::Size(32, 32));
    Patch p;
    std::vector<std::tuple<int, int, size_t, size_t>*> bunch;
    p.patch_creation(image, 32, 32, 32, bunch);
    for (int i=0; i<bunch.size(); i++) {
        cv::Rect roi(std::get<0>(*bunch[i]), std::get<1>(*bunch[i]), std::get<2>(*bunch[i]), std::get<3>(*bunch[i]));
        cv::Mat work_img(image, roi);
        hog.feature(work_img, 32, 4);
        cv::Mat testMat(int(train_set_vector.size()), 1764, CV_64FC1);
        cv::Mat testResponse(int(train_set_label_vector.size()), 1, CV_32S);
        testMat.convertTo(testMat, CV_32F);
        for(int j=0; j<hog.final_feature_vector.size(); j++)
                testMat.at<float>(0, j) = hog.final_feature_vector[j];
        svm->predict(testMat, testResponse);
        if (testResponse.at<float>(0, 0) > 0) {
            std::cout << "True" << std::endl;
            cv::Point pt1, pt2;
            pt1.x = std::get<0>(*bunch[i]);
            pt1.y = std::get<1>(*bunch[i]);
            pt2.x = std::get<0>(*bunch[i]) + std::get<2>(*bunch[i]);
            pt2.y = std::get<1>(*bunch[i]) + std::get<3>(*bunch[i]);
            rectangle(image, pt1, pt2, CV_RGB(255, 0, 0), 1);
        }
    }

    //Patch p;
    //std::vector<std::tuple<int, int, size_t, size_t>*> bunch;
    //p.patch_creation(image, 8, 8, 8, bunch);
    //p.patch_view(image, bunch);


    // Benchmarking things with Boost GIL
    //boost::gil::rgb8_image_t img;
    //boost::gil::jpeg_read_image("/home/peri/Documents/Computer Vision/faceDetection/src/gtech.jpg", img);

    std::string winName= "Test";
    cv::namedWindow(winName);
    cv::imshow(winName, image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
