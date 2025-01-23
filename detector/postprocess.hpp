#pragma once

#include "opencv2/opencv.hpp"

cv::Rect get_rect(cv::Mat& img, float bbox[4]);
