#pragma once

#include "opencv2/opencv.hpp"
#include "common_struct.hpp"

cv::Rect get_rect(cv::Mat& img, float bbox[4]);
void topk(std::vector<Detection>& res_batch, float* output, int output_size,
                float conf_thresh, int topk);
void draw_bbox(cv::Mat& img_batch, std::vector<Detection>& res_batch) ;
