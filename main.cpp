
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
#include "hikvision.hpp"
#include "usbcam.hpp"

#include "NvInfer.h"

std::atomic<bool> state;

detect_nx detector; // 推理器

usb_camera usb_cam; // USB相机
// toe_RS_camera RS_camera;  //未使用的d435i相机
s_camera_params config;  // 海康相机参数
HikGrab hik_cam(config); // 海康相机

cv::VideoCapture cap(0, cv::CAP_V4L2); // USB相机的图片捕获

cv::Mat frame;
cv::Mat frame_hik;
cv::Mat frame_usb;
cv::Mat show_frame;
cv::Mat show_frame_hik;
cv::Mat show_frame_usb;

enum E_CAMERA_RESULT
{
    NO_IMAGE = 0,
    SUCCESS,
    CAMERA_ERROR
};

// 监控命令行ctrl+c,用于手动退出
void sigint_handler(int sig)
{
    if (sig == SIGINT)
    {
        state.store(false);
    }
}

// 串口线程
/**
 * @brief 串口通信处理函数，负责将排球位置数据通过串口发送
 *
 * 该函数创建一个串口连接(/dev/ttyACM0),并在循环中持续发送排球的三维位置信息:
 * - 中心点x坐标
 * - 中心点y坐标
 * - 深度值
 * 函数会一直运行直到state标志被设置为false
 */
void serial_process()
{
    std::vector<double> msg;
    // serial.init_port(config);

    toe::serial_port serial("/dev/ttyACM0"); // 初始化串口

    while (state.load())
    {
        // msg.push_back(ball_posion.x); // 这里可以根据实际情况修改串口信息
        // msg.push_back(ball_posion.y);
        // msg.push_back(ball_posion.Deep);
        // std::cout << "串口线程" << std::endl;
        // serial.send_msg(msg);
        serial.write_port(detector.volley.center_x, detector.volley.center_y, detector.volley.deepth, 1, 1);
    }
}

// 处理线程
/**
 * @brief 执行目标检测处理的主函数
 *
 * 该函数初始化TensorRT推理引擎,并在循环中执行以下步骤:
 * 1. 获取输入图像并预处理
 * 2. 执行目标检测推理
 * 3. 后处理检测结果并可视化
 * 4. 获取检测到的排球目标的深度信息
 * 5. 显示处理结果和运行时间
 *
 * 函数会持续运行直到接收到退出信号(按下ESC键)或state标志被设置为false
 *
 * @note 函数依赖全局变量frame用于获取输入图像,state用于控制循环
 */
void detect_process(void)
{
    int k = 0;

    std::string engine_path = "/home/nvidia/RC_Volleyball_track_2025/detector/data/v10_fp16.engine";
    detector.RT_engine_init(engine_path);

    const int num_frames_to_test = 100;                          // 测试100帧以计算平均FPS
    auto start_time = std::chrono::high_resolution_clock::now(); // 记录开始时间

    while (state.load())
    {
        if (detector.usb_img_flag)
        {
            if (E_CAMERA_RESULT::CAMERA_ERROR == detector.usb_img_flag)
            {
                // if (fps == 50)
                std::cout << "CAMERA ALL ERROR!!!!!!!!" << std::endl;
                sleep(1);
                continue;
            }

            // 括号不能删，这是锁的生命周期
            detector.hik_img_flag = E_CAMERA_RESULT::NO_IMAGE;
            detector.usb_img_flag = E_CAMERA_RESULT::NO_IMAGE;
            detector.input_img_ = frame_usb.clone();
            show_frame = detector.input_img_.clone();

            detector.preprocess(detector.input_img_);
            detector.infer();
            detector.postprocess(show_frame);

            // if(detector.volley.flag_detected = 1)
            // {
            // detector.volley.deepth =
            // RS_camera.RS_get_depth_data((int)detector.volley.center_x , (int)detector.volley.center_y);

            // }

            auto end_time = std::chrono::high_resolution_clock::now();          // 记录结束时间
            std::chrono::duration<double> elapsed_time = end_time - start_time; // 计算时间差
            if (1)
            {

                detector.show_result(show_frame);
                std::cout << elapsed_time.count() << std::endl;
            }
        }

        // std::cout << "推理线程" << std::endl;
        if (k == 27)
        {
            state.store(false);
            break;
        }
    

        if (detector.hik_img_flag)
        {
            if (E_CAMERA_RESULT::CAMERA_ERROR == detector.hik_img_flag)
            {
                // if (fps == 50)
                std::cout << "CAMERA ALL ERROR!!!!!!!!" << std::endl;
                sleep(1);
                continue;
            }

            // 括号不能删，这是锁的生命周期
            detector.hik_img_flag = E_CAMERA_RESULT::NO_IMAGE;
            detector.usb_img_flag = E_CAMERA_RESULT::NO_IMAGE;
            detector.input_img_ = frame_usb.clone();
            show_frame = detector.input_img_.clone();

            detector.preprocess(detector.input_img_);
            detector.infer();
            detector.postprocess(show_frame);

            // if(detector.volley.flag_detected = 1)
            // {
            // detector.volley.deepth =
            // RS_camera.RS_get_depth_data((int)detector.volley.center_x , (int)detector.volley.center_y);

            // }

            auto end_time = std::chrono::high_resolution_clock::now();          // 记录结束时间
            std::chrono::duration<double> elapsed_time = end_time - start_time; // 计算时间差
            if (1)
            {

                detector.show_result(show_frame);
                std::cout << elapsed_time.count() << std::endl;
            }
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

/**
 * @brief 获取相机图像的线程函数
 *
 * 初始化并启动 RealSense 相机，持续获取彩色图像帧。
 * 在全局状态变量 state 为 true 时循环运行，通过互斥锁保护帧数据的读写。
 * 函数结束时自动销毁所有 OpenCV 窗口。
 *
 * 注: 包含被注释掉的海康相机相关代码
 */
void grab_img(void)
{

    int hik_id = 0;
    hik_cam.Hik_init();
    hik_cam.Hik_end();
    sleep(2);
    hik_cam.Hik_init();
    // 初始化相机
    // if(!RS_camera.RS_init())
    // {
    //     std::cerr << "Failed to initialize camera!" << std::endl;
    //     return;
    // }

    while (state.load())
    {
        if (E_CAMERA_RESULT::SUCCESS == detector.hik_img_flag)
        {
            usleep(10);
            continue;
        }
        if (hik_cam.get_one_frame(frame, 0))
        {
            detector.hik_img_flag = E_CAMERA_RESULT::SUCCESS;
        }
        // RS_camera.RS_get_frames();
        // detector.img_mutex_.lock();
        // hik_cam.get_one_frame(frame , 0);
        // detector.img_mutex_.unlock();
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
    hik_cam.Hik_end();
}

void grab_img_hikcam(void)
{
    int hik_id = 0;
    hik_cam.Hik_init();
    hik_cam.Hik_end();
    sleep(2);
    hik_cam.Hik_init();
    // 初始化相机
    // if(!RS_camera.RS_init())
    // {
    //     std::cerr << "Failed to initialize camera!" << std::endl;
    //     return;
    // }

    while (state.load())
    {
        if (E_CAMERA_RESULT::SUCCESS == detector.hik_img_flag)
        {
            usleep(10);
            continue;
        }
        if (hik_cam.get_one_frame(frame, 0))
        {
            detector.hik_img_flag = E_CAMERA_RESULT::SUCCESS;
        }
    }
    cv::destroyAllWindows(); // 销毁窗口

    hik_cam.Hik_end();
}
/**
 * @brief 从USB摄像头获取图像的线程函数
 *
 * 该函数负责初始化USB摄像头、持续获取图像帧并进行有效性检查。
 * 包含以下主要功能:
 * - 相机初始化与重连机制
 * - 图像帧获取与有效性验证
 * - 超时检测与错误恢复
 * - 资源自动释放
 *
 * 函数会持续运行直到state标志被设置为false。
 * 如果检测到相机错误或帧获取超时,会自动尝试重新初始化相机。
 */
void grab_img_usbcam(void)
{
    auto last_valid_time = std::chrono::steady_clock::now();

    // 首次初始化相机
    if (!usb_cam.usb_camera_init( cap))
    {
        std::cerr << "Camera initial initialization failed!" << std::endl;
        return;
    }
    while (state.load())
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - last_valid_time);
        // if (elapsed.count() > 3000)
        // {
        //     std::cerr << "Frame timeout detected! Attempting to recover..." << std::endl;
        //     detector.usb_img_flag = E_CAMERA_RESULT::CAMERA_ERROR;
        //     goto CAMERA_RECOVERY;
        // }

        if (E_CAMERA_RESULT::SUCCESS == detector.usb_img_flag)
        {
            usleep(10);
            continue;
        }
        if (usb_cam.usb_camera_get_frame(cap , frame_usb))
        {
            // 检查帧有效性
            if (frame_usb.empty() || frame_usb.cols < 640 || frame_usb.rows < 480)
            {
                std::cerr << "Invalid frame received!" << std::endl;
                detector.usb_img_flag = E_CAMERA_RESULT::NO_IMAGE;
                continue;
            }
            // 帧获取成功
            detector.usb_img_flag = E_CAMERA_RESULT::SUCCESS;
            last_valid_time = std::chrono::steady_clock::now();
        }
        else
        {
            std::cerr << "Failed to get frame from camera!" << std::endl;
            detector.usb_img_flag = E_CAMERA_RESULT::NO_IMAGE;
        }

        if (detector.usb_img_flag == E_CAMERA_RESULT::CAMERA_ERROR)
        {
            CAMERA_RECOVERY:
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            // 重新初始化相机
            if (usb_cam.usb_camera_init(cap))
            {
                std::cout << "Camera reinitialized successfully!" << std::endl;
                detector.usb_img_flag = E_CAMERA_RESULT::SUCCESS;
                last_valid_time = std::chrono::steady_clock::now();
            }
            else
            {
                std::cerr << "Camera reinitialization failed!" << std::endl;
            }
        }

    }
    cv::destroyAllWindows(); // 销毁窗口
    // cap.release(); // 确保释放资源
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
    // std::thread grab_thread = std::thread(grab_img);
    // std::thread grab_thread = std::thread(grab_img_hikcam);
    std::thread grab_usbcam_thread = std::thread(grab_img_usbcam);
    std::thread detect_thread = std::thread(detect_process);
    std::thread serial_thread = std::thread(serial_process);
    grab_usbcam_thread.detach();
    detect_thread.detach();
    serial_thread.detach();

    // 简单线程看门狗实现，需配合循环挂起的bash脚本使用
    // 这里会检测线程是否不正常运行，如果不正常立刻退出
    while (state.load())
    {
        sleep(1000);
        if (grab_usbcam_thread.joinable() && detect_thread.joinable() 
            && serial_thread.joinable())
        {
            state.store(false);
            break;
        }
    }

    return 0;
}
