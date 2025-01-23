#include "opencv4/opencv2/opencv.hpp"
#include "librealsense2/rs.hpp"
#include "intel_realsence.hpp"
#include <iostream>

void toe_RS_camera::RS_init(void)
{
    // 初始化相机,配置为640*480,60fps,BGR8格式（吊d435只能最高60的rgb）
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 90);
    pipe.start(cfg); // 启动管道
}

void toe_RS_camera::RS_get_img(void)
{
    //之后考虑换成异步的方式获取数据
    frames = pipe.wait_for_frames();

    // 获取深度图和彩色图
    rs2::depth_frame  depth_frame = frames.get_depth_frame();
    rs2::video_frame  color_frame = frames.get_color_frame();

    // 将深度图转化为cv::Mat格式,存一下
    cv::Mat color_image(color_frame.get_height(), color_frame.get_width(), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    
    width = depth_frame.get_width();
    height = depth_frame.get_height();
    depth_val = depth_frame.get_distance(width / 2, height / 2);

    //暂时的，调试用
    cv::imshow("rs" , color_image);
    std::cout << "depth_val: " << depth_val << std::endl;
}

void toe_RS_camera::RS_get_data(void)
{

}
