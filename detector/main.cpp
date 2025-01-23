#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
// #include "buffers.h"

#include "process.h"
#include "detect.hpp"

using namespace nvinfer1;

detect_nx my_nx_detector;

void process_decode_ptr_host(std::vector<Detection> &res, const float *decode_ptr_host, int bbox_element, cv::Mat &img,
                             int count)
{
    Detection det;
    for (int i = 0; i < count; i++)
    {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 5];
        if (keep_flag == 1)
        {
            det.bbox[0] = decode_ptr_host[basic_pos + 0];
            det.bbox[1] = decode_ptr_host[basic_pos + 1];
            det.bbox[2] = decode_ptr_host[basic_pos + 2];
            det.bbox[3] = decode_ptr_host[basic_pos + 3];
            det.conf = decode_ptr_host[basic_pos + 4];
            // det.class_id = decode_ptr_host[basic_pos + 5];
            res.push_back(det);
        }
    }
}

void single_process(std::vector<Detection> &res_batch, const float *decode_ptr_host,
                    int bbox_element, const cv::Mat &img)
{

    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, 8400);

    auto &img_ = const_cast<cv::Mat &>(img);
    process_decode_ptr_host(res_batch, &decode_ptr_host[count], bbox_element, img_, count);
}

void batch_process(std::vector<std::vector<Detection>> &res_batch, const float *decode_ptr_host, int batch_size,
                   int bbox_element, const std::vector<cv::Mat> &img_batch)
{
    res_batch.resize(batch_size);
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, 8400);
    for (int i = 0; i < batch_size; i++)
    {
        auto &img = const_cast<cv::Mat &>(img_batch[i]);
        process_decode_ptr_host(res_batch[i], &decode_ptr_host[i * count], bbox_element, img, count);
    }
}

static float data[1 * 3 * 640 * 640];

uint8_t *image_device;
uint8_t *image_host;

uchar *midDevData;
uchar *srcDevData;
int inputIndex;
int outputIndex;

float *nms_out;
float *decode_out;
float *final_out;

float *buffers_[2];
cudaStream_t stream_;

// std::vector<unsigned char> engine_data_;

std::vector<cv::Mat> imgsBatch_;

int main()
{
    std::string engine_path = "/home/nvidia/RC_Volleyball_track_2025/detector/data/volley_v10.engine";
    std::string path = "/home/nvidia/RC_Volleyball_track_2025/detector/data/20241017_1562.jpg";
    cv::Mat img = cv::imread(path);
    // std::vector<cv::Mat> img_batch;
    // img_batch.push_back(img);

    my_nx_detector.RT_engine_init(engine_path);
    std::cout << "engine init success" << std::endl;
    // 读出输出和输入的tensor索引
    // target data
    //std::cout << "inputIndex: " << outputIndex << std::endl;
    // const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    cuda_preprocess_init(640*640*3);
    //img_batch.emplace_back(img);
    //my_nx_detector.batch_copy(img);
    //my_nx_detector.copy(img);
    my_nx_detector.preprocesss(img);
    std::cout << "preprocess success" << std::endl;

    if(my_nx_detector.infer() == true)
    {
        std::cout << "infer success" << std::endl;
    }
    my_nx_detector.postprocess();
    std::cout << "postprocess success" << std::endl;
    return 0;
}
