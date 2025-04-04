#ifndef PROCESS_H
#define PROCESS_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "common_struct.hpp"

struct AffineMatrix
{
    float value[6];
};

// __device__ void affine_project_device_kernel(AffineMat *matrix, int x, int y, float *proj_x, float *proj_y);

// __global__
// void resize_rgb_padding_device_kernel(unsigned char* src, int src_width, int src_height, int src_area, int src_volume,
// 	float* dst, int dst_width, int dst_height, int dst_area, int dst_volume,
// 	int batch_size, float padding_value, AffineMat matrix);

// __global__ void bgr2rgb_device_kernel(float *src, float *dst,
//                                       int batch_size, int img_height, int img_width, int img_area, int img_volume);

// __global__ void norm_device_kernel(float *src, float *dst,
//                                    int batch_size, int img_height, int img_width,
//                                    int img_area, int img_volume);
// __global__ void hwc2chw_device_kernel(float *src, float *dst,
//                                       int batch_size, int img_height, int img_width, int img_area, int img_volume);

 void cuda_preprocess_init(int max_image_size);

void cuda_batch_preprocess(cv::Mat img_batch,
                           float *dst, int dst_width, int dst_height,
                           cudaStream_t stream);

#endif // ___PREPROCESS_CUH___
