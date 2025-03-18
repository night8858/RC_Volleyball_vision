#ifndef INTEL_REALSENCE_HPP
#define INTEL_REALSENCE_HPP

#include "librealsense2/rs.hpp"
#include <mutex>
class toe_RS_camera
{
public:
    toe_RS_camera() = default;
    ~toe_RS_camera() {};

    bool RS_init(void);
    bool RS_get_frames(void);
    cv::Mat RS_get_color_img(void);
    cv::Mat RS_get_depth_img(void);
    float RS_get_depth_data(int x, int y);

    std::mutex frames_mutex;

private:
    rs2::pipeline pipe; // 初始化管道类
    rs2::config cfg;    // 配置类

    float depth_scale; // 深度缩放因子
    int width, height; // 图像宽高
    float depth_val;   // 深度值

    rs2::frameset frames; // 帧集类
};

#endif