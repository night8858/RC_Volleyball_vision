#include "my_serial.h"

using namespace toe;
using namespace std;
//int xxa;
#ifndef USE_NVIDIA
serial_port::serial_port(const char* port)
{
    std::string post_path;
    char post_num = 0;
    post_path = port;
    // 轮询打开串口
    while (1)
    {
        serial_fd.open(post_path, m_ec);
        if (m_ec)
        {
            //throw std::logic_error("串口打开失败 ", post_path);
            post_num++;
            post_path[post_path.size()-1] = (post_num + '0');
            // 只遍历0到10号串口
            if (post_num >= 9)
            {
                post_num = 0;
            }
            continue;
        }
        else
        {
            break;
        }
    }
    //设置串口参数  
    serial_fd.set_option(boost::asio::serial_port::baud_rate(115200), m_ec);
    if (m_ec)
    {
        //throw std::logic_error("串口参数设置失败，错误信息:", m_ec.message());
    }
    serial_fd.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none), m_ec);  
    if (m_ec)
    {
        //throw std::logic_error("串口参数设置失败，错误信息:", m_ec.message());
    }
    serial_fd.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none), m_ec);  
    if (m_ec)
    {
        //throw std::logic_error("串口参数设置失败，错误信息:", m_ec.message());
    }
    serial_fd.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one), m_ec);  
    if (m_ec)
    {
        //throw std::logic_error("串口参数设置失败，错误信息:", m_ec.message());
    }
    serial_fd.set_option(boost::asio::serial_port::character_size(8), m_ec);
    if (m_ec)
    {
        //throw std::logic_error("串口参数设置失败，错误信息:", m_ec.message());
    }
    // fd = open(port, O_RDWR | O_NOCTTY);
    // if (-1 == fd)
    // {
    //     throw std::logic_error("串口打开失败");
    // }

    // struct termios Opt;
    // if (tcgetattr(fd, &Opt) == -1)
    // {
    //     throw std::logic_error("获取串口属性失败");
    // }

    // Opt.c_cflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    // Opt.c_cflag &= ~OPOST;

    // if (tcsetattr(fd, TCSANOW, &Opt) == -1)
    // {
    //     throw std::logic_error("设置串口属性失败");
    // }

}
#else
serial_port::serial_port()
{
    fd = open("/dev/ttyACM0", O_RDWR);
    if (-1 == fd)
    {
        throw std::logic_error("串口打开失败");
        close(fd);
    }
    struct termios Opt;
    tcgetattr(fd,&Opt);
    Opt.c_cflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    Opt.c_cflag &= ~OPOST;
    tcsetattr(fd, TCSANOW, &Opt);
}
#endif
serial_port::~serial_port()
{
    // close(fd);
}

bool serial_port::write_port(const float& x, const float& y, const float& z, int id,int color)
{
    //memset(send_temp,0,sizeof(send_temp));

    acm_data temp;
    temp.data = x;
    sbuff[0] = temp.bit[0];
    sbuff[1] = temp.bit[1];
    sbuff[2] = temp.bit[2];
    sbuff[3] = temp.bit[3];
    temp.data = y;
    sbuff[4] = temp.bit[0];
    sbuff[5] = temp.bit[1];
    sbuff[6] = temp.bit[2];
    sbuff[7] = temp.bit[3];
    temp.data = z;
    sbuff[8] = temp.bit[0];
    sbuff[9] = temp.bit[1];
    sbuff[10] = temp.bit[2];
    sbuff[11] = temp.bit[3];
    //std::cout << "x: " << x<< std::endl;
    // sbuff[12] = (uint8_t)id;
    // sbuff[13] = (uint8_t)color;

    // int sum_check = 0;

    // for (int i = 0; i < 12; i++)
    // {
    //     sum_check += sbuff[i];
    // }

    // temp.data = sum_check;
    // sbuff[14] = temp.bit[0];
    // sbuff[15] = temp.bit[1];
    // sbuff[16] = temp.bit[2];
    // sbuff[17] = temp.bit[3];
    
    serial_fd.write_some(boost::asio::buffer(sbuff) ,m_ec);        
    //write(fd,send_temp,sizeof(send_temp));

    return true;
}

bool serial_port::read_port(int& color, int& mode)
{
    memset(&rbuff,0,sizeof(rbuff));
    serial_fd.read_some(boost::asio::buffer(rbuff), m_ec);  
    // memset(get_temp,0,sizeof(get_temp));
    // ssize_t bytesRead = read(fd, get_temp, sizeof(get_temp));
    if ((int)rbuff[0] == 166)
    {
        left_or_right_ = 0;
        if ((int)rbuff[1] > 100)
        {
            color = 0;
        }
        else
            color = 1;
        mode = (int)rbuff[2];
        //std::cout << "aaax1: " << (int)mode<< std::endl;
        //std::cout << "color: " << (int)color<< std::endl;
        //xxa++;
    }
    else if ((int)rbuff[0] == 167)
    {
        left_or_right_ = 1;
        if ((int)rbuff[1] > 100)
        {
            color = 0;
        }
        else
            color = 1;
        mode = (int)rbuff[2];
        //std::cout << "aaax2: " << (int)mode<< std::endl;
        //  std::cout << "aaax: " << (int)mode<< std::endl;
        // std::cout << "color: " << (int)color<< std::endl;
    }
    else
        return false;
    //std::cout << "id: " << (int)get_temp[0]<< std::endl;
    //std::cout << "mode: " << mode << std::endl;
    
    return true;
}