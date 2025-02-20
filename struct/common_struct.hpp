#ifndef COMMON_STRUCT_HPP
#define COMMON_STRUCT_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

#define CHECK(call)                                                           \
    do                                                                        \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            fprintf(stderr, "CUDA Error:\n");                                 \
            fprintf(stderr, "    File: %s\n", __FILE__);                      \
            fprintf(stderr, "    Line: %d\n", __LINE__);                      \
            fprintf(stderr, "    Error code: %d\n", err);                     \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 这里是各种所需结构体
struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};



struct alignas(float) Detection
{
    float bbox[4];
    float conf;
    float class_id;
};


// 相机内参
typedef struct
{
    int device_id;
    int width;
    int height;
    int offset_x;
    int offset_y;
    int exposure;

} s_camera_params;


typedef struct 
{
    float center_x;
    float center_y;
    float deepth;

    int id;

}volleyball;


typedef struct
{
    int id;
    std::vector<cv::Point2f> merge_pts;
    std::vector<float> merge_confs;
} pick_merge_store;

// struct armor_compare{
//     bool operator ()(const s_ball& a,const s_ball& b) {
//         return a.conf > b.conf;
//     }
// };

#endif
