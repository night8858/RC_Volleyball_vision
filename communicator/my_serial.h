/*
* my_serial.h
* Created on: 20230614
* Author: sumang, sunge
* Description: communicate
*/
#ifndef MY_SERIAL_H_
#define MY_SERIAL_H_
#include <boost/asio.hpp>
#include <iostream>
#include <vector>
#include "memory.h"
#include "fcntl.h"
#include "errno.h"
#include "sys/types.h"
#include "sys/stat.h"
#include "termios.h"
#include "unistd.h"

namespace toe
{
    union acm_data
    {
        uint8_t bit[4];
        float data;
    };

    class serial_port
    {
    public:
        serial_port(const char* port);
        ~serial_port();
        bool write_port(const float& x1, const float& y1, const float& z1,
            const float& x2, const float& y2, const float& z2 );
        bool read_port(int& color, int& mode);
        int left_or_right_;

    public:
        int fd;
        uint8_t get_temp[8];
        uint8_t send_temp[30];
    private:
        // 轮询打开串口
        void init_port();
        // boost串口相关    
        boost::asio::io_service io_s;
        boost::asio::serial_port serial_fd = boost::asio::serial_port(io_s);
        boost::system::error_code m_ec;
        // 接收发送缓冲区
        uint8_t rbuff[1024];
        uint8_t sbuff[24];
        //记录的模式信息
        // int mode = DEFALUTE_MODE;
        // //记录的颜色信息
        // bool color = DEFALUTE_COLOR;
        
    }; 
}



#endif