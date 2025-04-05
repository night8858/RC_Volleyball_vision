#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <mutex>
#include <vector>

#include "common_struct.hpp"
#include <opencv2/video/tracking.hpp>

class BallTracker {
private:
    cv::KalmanFilter KF;
    cv::Mat measurement;
    bool isInitialized = false;

public:
    BallTracker() {
        // 状态维度：4 (x, y, vx, vy)，测量维度：2 (x, y)
        KF.init(4, 2, 0);
        
        // 转移矩阵 (假设匀速运动模型)
        KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // 测量矩阵 (只能观测x,y)
        cv::setIdentity(KF.measurementMatrix);

        // 过程噪声协方差 (调节参数Q)
        cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-2));

        // 测量噪声协方差 (调节参数R)
        cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));

        // 后验误差协方差初始化
        cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));

        measurement = cv::Mat::zeros(2, 1, CV_32F);
    }

    cv::Point2f update(cv::Point2f detectedPoint) {
        if (!isInitialized) {
            KF.statePost.at<float>(0) = detectedPoint.x;
            KF.statePost.at<float>(1) = detectedPoint.y;
            isInitialized = true;
        }

        // 预测阶段
        cv::Mat prediction = KF.predict();

        // 更新阶段
        measurement.at<float>(0) = detectedPoint.x;
        measurement.at<float>(1) = detectedPoint.y;
        KF.correct(measurement);

        return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }
};



class detect_nx
{
public:
    detect_nx();
    ~detect_nx();

    const int max_size_ = 10;

    std::vector<cv::Mat> input_imgs_;
    std::vector<cv::Mat> input_cam1_imgs_;
    std::vector<cv::Mat> input_cam2_imgs_;

    
    std::mutex img_mutex_;
    std::mutex img_cam1_mutex_;
    std::mutex img_cam2_mutex_;

    int hik_img_flag; // hik相机图像标志位
    int usb_img_flag; // usb相机图像标志位

    cv::Mat input_img_;
    cv::Mat input_cam1_img_;
    cv::Mat input_cam2_img_;

    cv::Mat show_img_;
    cv::Mat show_cam1_img_;
    cv::Mat show_cam2_img_;

    volleyball volley;      // 存储排球数据
    volleyball volley_cam1; // 存储排球数据
    volleyball volley_cam2; // 存储排球数据

    void push_img(cv::Mat &img ,  int cam_id);

public:
    void RT_engine_init(std::string engine_path);
    void preprocess(void);
    bool infer(void);
    void postprocess(void);
    void show_result(cv::Mat &show_img , Detection &det);

private:

    int flag = 2;    //用来判断单路推理双路推理

    cv::Mat m_img_src;
    AffineMat m_dst2src;

    Detection det;
    Detection det1;
    Detection det2;

    std::vector<Detection> res;
    std::vector<Detection> res1;
    std::vector<Detection> res2;

    nvinfer1::Dims m_output_dims;
    int m_output_area;
    int m_total_objects;

    std::vector<unsigned char> engine_data_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;

    int inputIndex;
    int outputIndex;

    // input
    float *input_host;

    float *device_buffers[2];          //存储输入输出的数据
    // output
    float *output_device_host;

    uint8_t *img_buffer_host_1;
    uint8_t *img_buffer_device_1;

    uint8_t *img_buffer_host_2;
    uint8_t *img_buffer_device_2;


    float *output_device;
    float *output_objects_device;
    float *output_objects_host;

};
