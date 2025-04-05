#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"

#include "detect.hpp"
#include "process.h"
#include "common.h"
#include "postprocess.hpp"

using namespace nvinfer1;
// 全局或类成员变量
BallTracker ballTracker;

detect_nx::detect_nx(void)
{

    runtime_ = nullptr;
    engine_ = nullptr;
    context_ = nullptr;
    ////////// init cuda ///////////

    int max_img_size = 3000000;

    int output_objects_size = 1 * (1 + 300 * 6); // 1: count
    output_objects_host = new float[output_objects_size];
}

detect_nx::~detect_nx(void)
{
}

/**
 * @brief 为 TensorRT 推理准备输入输出缓冲区
 * @param engine TensorRT 引擎指针
 * @param input_buffer_device 设备端输入缓冲区指针的指针
 * @param output_buffer_device 设备端输出缓冲区指针的指针
 * @param output_buffer_host 主机端输出缓冲区指针的指针
 *
 * 该函数分配用于推理的 GPU 和 CPU 内存缓冲区。
 * 输入缓冲区大小为 1x3x640x640 的浮点数组。
 * 输出缓冲区大小为 6x300 的浮点数组。
 * 函数会验证引擎绑定点数量及输入输出 tensor 的正确性。
 */
void prepare_buffer(std::shared_ptr<nvinfer1::ICudaEngine> engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host)
{
    assert(engine->getNbBindings() == 2);

    const int kOutputSize = 300 * sizeof(Detection) / sizeof(float);

    int batch_size = 2;
    // 这里是获取输入输出tensor的索引
    const int inputIndex = engine->getBindingIndex("images");
    const int outputIndex = engine->getBindingIndex("output0");
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    // if (1)
    // {
    CUDA_CHECK(cudaMalloc((void **)input_buffer_device, batch_size * 3 * 640 * 640 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)output_buffer_device, batch_size * kOutputSize * sizeof(float)));

    *output_buffer_host = new float[kOutputSize * batch_size];
    // }

    // if (1)
    // {
    //     // 改为推理两个图片

    //     CUDA_CHECK(cudaMalloc((void **)input_buffer_device, 2 * 3 * 640 * 640 * sizeof(float)));
    //     CUDA_CHECK(cudaMalloc((void **)output_buffer_device, 2 * 6 * 300 * sizeof(float)));

    //     *output_buffer_host = new float[2 * 6 * 300];
    // }
}

/**
 * @brief 初始化用于目标检测的TensorRT推理引擎
 * @param engine_path 序列化的TensorRT引擎文件路径
 *
 * 本函数执行以下初始化步骤：
 * 1. 从文件加载序列化的推理引擎
 * 2. 创建TensorRT运行时并反序列化引擎
 * 3. 创建用于推理的执行上下文
 * 4. 设置输出维度及缓冲区大小
 * 5. 初始化CUDA内存缓冲区和推理流
 *
 * @throws 当无法打开引擎文件时抛出断言错误
 * @note 需要CUDA和TensorRT运行时环境
 */

void detect_nx::RT_engine_init(std::string engine_path)
{
    detect_nx();
    // 初始化
    std::ifstream engine_file(engine_path, std::ios::binary);
    // 打印引擎路径（个人理解---night）
    assert(engine_file.is_open() && "Unable to load engine_ file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    if (length <= 0)
    {
        std::cerr << "Error: Invalid engine file size (" << length << ")" << std::endl;
        exit(-1);
    }
    engine_data_.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data_.data()), length);

    // 创建runtime
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime_)
    {
        std::cout << "runtime_ create failed" << std::endl;
        exit(-1);
    }

    auto plan = engine_data_;
    // 创建引擎
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_)
    {
        exit(-1);
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        std::cout << "context_ create failed" << std::endl;
        exit(-1);
    }

    nvinfer1::Dims output_dims = engine_->getBindingDimensions(0);
    for (int i = 0; i < output_dims.nbDims; i++)
    {
        std::cout << i << "   " << output_dims.d[i] << std::endl;
    }

    // context_->setBindingDimensions(0, Dims4(2, 3, 640, 640));
    // nvinfer1::Dims output_dims = context_->getBindingDimensions(0);
    // m_output_dims = this->context_->getBindingDimensions(1);
    // m_total_objects = m_output_dims.d[2];
    // std::cout << "total_objects: " << m_total_objects << std::endl;
    // assert(m_param.batch_size <= m_output_dims.d[0]);
    // m_output_area = 1;
    // for (int i = 1; i < m_output_dims.nbDims; i++)
    // {
    //     if (m_output_dims.d[i] != 0)
    //     {
    //         m_output_area *= m_output_dims.d[i];
    //     }
    // }
    // std::cout << m_output_area << std::endl;

    // 创建GPU和CPU内存
    prepare_buffer(engine_, &device_buffers[0], &device_buffers[1], &output_device_host);
    CHECK(cudaStreamCreate(&stream_));

    cuda_preprocess_init(1440 * 1080 * 3); // 分配处理的最大的空间
}

void detect_nx::push_img(cv::Mat &img, int cam_id)
{
    // img_mutex_.lock();
    // if (input_imgs_.size() == max_size_)
    //     input_imgs_.clear();
    // input_imgs_.emplace_back(img);
    // img_mutex_.unlock();
    if (flag == 1)
    {
        if (cam_id == 0)
        {
            img_mutex_.lock();
            input_img_ = img.clone();
            img_mutex_.unlock();
        }
    }
    else if (cam_id == 1)
    {
        img_cam1_mutex_.lock();
        input_cam1_img_ = img.clone();
        img_cam1_mutex_.unlock();
    }
    else if (cam_id == 2)
    {
        img_cam2_mutex_.lock();
        input_cam2_img_ = img.clone();
        img_cam2_mutex_.unlock();
    }
}

/// @brief 输入图像预处理
/// @param imgsBatch 输入图像（单张）
void detect_nx::preprocess(void)
{
    if (flag == 1)
    {
        if (show)
        {
            show_img_ = input_img_.clone();
        }
        // 处理单路数据
        // cuda_batch_preprocess(input_img_, device_buffers[0], 640, 640, stream_);
    }

    if (flag == 2)
    {
        if (show)
        {
            show_cam1_img_ = input_cam1_img_.clone();
            show_cam2_img_ = input_cam2_img_.clone();
        }

        // 处理两路数据
        // cuda_batch_preprocess(input_cam1_img_,  device_buffers[0], 640, 640, stream_);
        cuda_2batch_preprocess(input_cam1_img_, input_cam2_img_, device_buffers[0], 640, 640, stream_);

        // CHECK(cudaStreamSynchronize(stream_));
        // cuda_batch_preprocess(input_cam2_img_,img_buffer_host_2, img_buffer_device_2 , device_buffers[0] + 640 * 640 * 3 *sizeof(float), 640, 640, stream_);
        CHECK(cudaStreamSynchronize(stream_));
    }
}

bool detect_nx::infer(void)
{

    bool context = detect_nx::context_->enqueueV2((void **)device_buffers, stream_, nullptr);

    CHECK(cudaStreamSynchronize(stream_));
    return context;
}

void detect_nx::postprocess(void)
{

    output_device_host = new float[2 * 6 * 300];
    CHECK(cudaMemcpy(output_device_host, device_buffers[1], 2 * 1800 * sizeof(float), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 7; i++)
    // {
    //     std::cout <<  output_device_host[i];
    // }
    // std::cout<<std::endl;
    // for (int i = 300; i < 307; i++)
    // {
    //     std::cout <<  output_device_host[i];
    // }
    // std::cout<<std::endl;

    if (flag == 1)
    {
        decode(det, &output_device_host[0], 0.6, 1);
        // topk( res, output_device_host, 0, 0.8 ,100);
        volley = get_ball(input_img_, det);
        // draw_bbox(imgsBatch, res);
        if (show)
        {
            // detect_nx::show_result(show_img_, det);
            // std::cout << "X : " << volley.center_x
            //           << "Y : " << volley.center_y << std::endl;
            // std::cout << "deepth : " << volley.deepth << std::endl;

            cv::imshow("1", show_img_);
            cv::waitKey(1);
        }
    }

    if (flag == 2)
    {

        decode(det1, &output_device_host[0], 0.6, 1);
        decode2(det2, &output_device_host[0], 0.6, 1);

        // topk( res, output_device_host, 0, 0.8 ,100);
        volley_cam1 = get_ball(input_cam1_img_, det1);
        volley_cam2 = get_ball(input_cam2_img_, det2);

        // 在检测循环中调用
        // volley_cam1 = get_ball(input_cam1_img_, det1); // 原始检测
        // if (volley_cam1.isValid)
        // { // 假设返回结构包含有效性标志
        //     cv::Point2f filteredPos = ballTracker.update(cv::Point2f(volley_cam1.x, volley_cam1.y));
        //     volley_cam1.x = filteredPos.x;
        //     volley_cam1.y = filteredPos.y;
        // }
        // else
        // {
        //     // 若未检测到球，使用预测值（可选）
        //     cv::Mat prediction = ballTracker.KF.predict();
        //     volley_cam1.x = prediction.at<float>(0);
        //     volley_cam1.y = prediction.at<float>(1);
        // }

        volley_cam1.deepth = 1 / (volley_cam1.radius) * 340;
        volley_cam2.deepth = 1 / (volley_cam2.radius) * 41.6;
        // draw_bbox(imgsBatch, res);
        std::cout << "X1 : " << volley_cam1.center_x << "   "
                  << "Y1 : " << volley_cam1.center_y << "   "
                  << "deepth1 : " << volley_cam1.deepth << std::endl;

        std::cout << "X2 : " << volley_cam2.center_x << "   "
                  << "Y2 : " << volley_cam2.center_y << "   "
                  << "deepth : " << volley_cam2.deepth << std::endl;

        // std::cout << "D1 : " << volley_cam1.deepth  << "   "
        //           << "D2 : " << volley_cam2.deepth << std::endl;

        if (show)
        {
            detect_nx::show_result(show_cam1_img_, det1);
            detect_nx::show_result(show_cam2_img_, det2);

            std::cout << "X1 : " << volley_cam1.center_x << "   "
                      << "Y1 : " << volley_cam1.center_y << "   "
                      << "deepth1 : " << volley_cam1.deepth << std::endl;

            std::cout << "X2 : " << volley_cam2.center_x << "   "
                      << "Y2 : " << volley_cam2.center_y << "   "
                      << "deepth : " << volley_cam2.deepth << std::endl;

            cv::imshow("1", show_cam1_img_);
            cv::imshow("2", show_cam2_img_);

            cv::waitKey(1);
        }

    }
}

void detect_nx::show_result(cv::Mat &show_img, Detection &det)
{
    draw_bbox_single(show_img, det);
}