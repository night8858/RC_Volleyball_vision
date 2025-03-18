#include "opencv4/opencv2/opencv.hpp"
#include "librealsense2/rs.hpp"
#include "intel_realsence.hpp"
#include <iostream>

// 初始化
bool toe_RS_camera::RS_init(void)
{
    // 初始化相机,配置为640*480,60fps,BGR8格式（吊d435只能最高60的rgb）
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 90);
//    pipe.start(cfg); // 启动管道

    // 启动管道
    try
    {
        auto profile = pipe.start(cfg);
        return true;
    }
    catch (const rs2::error &e)
    {
        std::cerr << "Failed to start RealSense pipeline: " << e.what() << std::endl;
        return false;
    }
    auto depth_sensor = pipe.get_active_profile().get_device().first<rs2::depth_sensor>();
    // 启用自动曝光
    depth_sensor.set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
    
}

// 获取帧
bool toe_RS_camera::RS_get_frames(void)
{
    // 之后考虑换成异步的方式获取数据
    frames = pipe.wait_for_frames();

    return true;
}

// 获取彩色图像
cv::Mat toe_RS_camera::RS_get_color_img(void)
{
    auto color_frame = frames.get_color_frame();

    return cv::Mat(cv::Size(640, 480), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
}
// 获取深度图像
cv::Mat toe_RS_camera::RS_get_depth_img(void)
{
    auto depth_frame = frames.get_depth_frame();
    return cv::Mat(cv::Size(640, 480), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
}

// 取得指定点的深度值
float toe_RS_camera::RS_get_depth_data(int x, int y)
{
    if(x < 1 || x > 639 || y < 1 || y > 479)
        return 0;

    frames_mutex.lock();
    auto depth_frame = frames.get_depth_frame();
    frames_mutex.unlock();


    return depth_frame.get_distance(x, y);
}
