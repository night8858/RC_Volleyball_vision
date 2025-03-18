
/*

               _____                       _____                       ____
              /\    \                     /\    \                     /\    \
             /::\    \                   /::\    \                   /::\    \
             \:::\    \                 /::::\    \                 /::::\    \
              \:::\    \               /::::::\    \               /::::::\    \
               \:::\    \             /:::/\:::\    \             /:::/\:::\    \
                \:::\    \           /:::/  \:::\    \           /:::/__\:::\    \
                /::::\    \         /:::/    \:::\    \         /::::\   \:::\    \
               /::::::\    \       /:::/      \:::\    \       /::::::\   \:::\    \
              /:::/\:::\    \     /:::/        \:::\    \     /:::/\:::\   \:::\    \
             /:::/  \:::\____\   /:::/          \:::\____\   /:::/__\:::\   \:::\____\
            /:::/   /\::/    /   \:::\          /:::/    /   \:::\   \:::\   \::/    /
           /:::/   /  \/____/     \:::\        /:::/    /     \:::\   \:::\   \/____/
          /:::/   /                \:::\      /:::/    /       \:::\   \:::\    \
         /:::/   /                  \:::\    /:::/    /         \:::\   \:::\____\
        /:::/   /                    \:::\  /:::/    /           \:::\   \::/    /
       /:::/   /                      \:::\/:::/    /             \:::\   \/____/
      /:::/   /                        \::::::/    /               \:::\    \
     /:::/   /                          \::::/    /                 \:::\____\
     \::/   /                            \::/    /                   \::/    /
      \/___/                              \/____/                     \/____/

TOE创新实验室
*/

#include <iostream>
#include <string>
#include <thread>
#include <fstream>
#include <atomic>
#include <chrono>

// ctrl+c的中断捕获
#include "signal.h"

#include "my_serial.h"
#include <opencv2/opencv.hpp>
#include "detect.hpp"
#include "intel_realsence.hpp"
#include "NvInfer.h"

std::atomic<bool> state;

toe_RS_camera RS_camera;

detect_nx detector;

cv::Mat frame;
cv::Mat show_frame;
// 监控命令行ctrl+c,用于手动退出
void sigint_handler(int sig)
{
    if (sig == SIGINT)
    {
        state.store(false);
    }
}

// 串口线程
void serial_process()
{
    std::vector<double> msg;
    //serial.init_port(config);
 
    toe::serial_port serial("/dev/ttyACM0"); // 初始化串口

    while (state.load())
    {
        // msg.push_back(ball_posion.x); // 这里可以根据实际情况修改串口信息
        // msg.push_back(ball_posion.y);
        // msg.push_back(ball_posion.Deep);
        // std::cout << "串口线程" << std::endl;
        // serial.send_msg(msg);
        serial.write_port(detector.volley.center_x , detector.volley
        .center_y , detector.volley.deepth , 1 , 1 );

    }
}

// 处理线程
void detect_process(void)
{
    int k = 0;

    std::string engine_path = "/home/nvidia/RC_Volleyball_track_2025/detector/data/v10_fp16.engine";
    detector.RT_engine_init(engine_path);

    const int num_frames_to_test = 100; // 测试100帧以计算平均FPS
    auto start_time = std::chrono::high_resolution_clock::now(); // 记录开始时间

    while (state.load())
    {
        // 括号不能删，这是锁的生命周期
        detector.input_img_ = frame.clone();
        show_frame = detector.input_img_ .clone();

        detector.preprocess(detector.input_img_);
        detector.infer();
        detector.postprocess(show_frame);
        
        // if(detector.volley.flag_detected = 1)
        // {
            detector.volley.deepth = 
            RS_camera.RS_get_depth_data((int)detector.volley.center_x , (int)detector.volley.center_y);
 
        // }
        
        auto end_time = std::chrono::high_resolution_clock::now(); // 记录结束时间
        std::chrono::duration<double> elapsed_time = end_time - start_time; // 计算时间差
        if(1)
        {
 
            detector.show_result(show_frame);
            std::cout << elapsed_time.count() << std::endl;
        }
        
        // std::cout << "推理线程" << std::endl;
        if (k == 27)
        {
            state.store(false);
            break;
        }   
    }
    cv::destroyAllWindows();
}
// 图像获取线程
// 这里我需要解释一下，因为海康的sdk直接有非用户触发，由相机内部的定时器触发来生成图像的api
// 所以对于使用该api的海康相机只要初始化就是异步的，不需要这个线程，初始化可以在主线程做，使用这个api是为了以后能够更方便的切换为硬触发，不需要修改代码
// 但是对于usb相机和realsence都需要手动由用户触发，不支持这种模式，所以需要保留这个线程以备手动触发获取图像

// 暂时注释掉，因为要简单使用，不用海康
void grab_img(void)
{
    // 初始化相机
    if(!RS_camera.RS_init())
    {
        std::cerr << "Failed to initialize camera!" << std::endl;
        return;
    }

    while (state.load())
    {
        RS_camera.RS_get_frames();

        RS_camera.frames_mutex.lock();
        frame = RS_camera.RS_get_color_img();
        RS_camera.frames_mutex.unlock();

    }

    cv::destroyAllWindows(); // 销毁窗口

    // int device_num = 0;
    // hik_cam.hik_init(config, device_num);
    //// 重启一次防止不正常退出后启动异常
    // hik_cam.hik_end();
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // hik_cam.hik_init(config, device_num);
    // while (state.load())
    //{
    //     // 手动触发获取图像
    //     sleep(5);
    // }
    // hik_cam.hik_end();
}

int main()
{
    // 初始化全局变量
    state.store(true);
    // 为了提升cout的效率关掉缓存区同步，此时就不能使用c风格的输入输出了，例如printf
    // oi上常用的技巧，还有提升输出效率的就是减少std::endl和std::flush的使用
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    // 看一下项目的路径，防止执行错项目
    // std::cout << PROJECT_PATH << std::endl;

    // std::ifstream f(std::string(PROJECT_PATH) + std::string("/config.json"));
    // config = nlohmann::json::parse(f);

    // 启动线程
    std::thread grab_thread = std::thread(grab_img);
    std::thread detect_thread = std::thread(detect_process);
    std::thread serial_thread = std::thread(serial_process);
    grab_thread.detach();
    detect_thread.detach();
    serial_thread.detach();

    // 简单线程看门狗实现，需配合循环挂起的bash脚本使用
    // 这里会检测线程是否不正常运行，如果不正常立刻退出
    while (state.load())
    {
        sleep(1000);                                               
        if (grab_thread.joinable() && detect_thread.joinable() && serial_thread.joinable())
        {
            state.store(false);
            break;
        }
    }

    return 0;
}
