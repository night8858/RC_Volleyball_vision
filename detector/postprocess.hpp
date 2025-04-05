#pragma once

#include "opencv2/opencv.hpp"
#include "common_struct.hpp"

cv::Rect get_rect(cv::Mat& img, float bbox[4]);
void topk(std::vector<Detection>& res_batch, float* output, int output_size,
                float conf_thresh, int topk);
void draw_bbox(cv::Mat& getted_frame, std::vector<Detection>& res_batch) ;
void decode(Detection &res ,float* output, float conf_thresh, int tokp );
void decode2(Detection &res, float *output, float conf_thresh, int tokp);

void double_decode(Detection &res1, Detection &res2, float *output, float conf_thresh, int tokp);
void draw_bbox_single(cv::Mat& getted_frame, Detection& res_batch) ;
volleyball get_ball(cv::Mat &getted_frame , Detection &volley);

