#ifndef INTEL_REALSENCE_HPP
#define INTEL_REALSENCE_HPP

#include "librealsense2/rs.hpp"


class toe_RS_camera
{
    public:
    toe_RS_camera() = default;
    ~toe_RS_camera(){};
    
    rs2::pipeline pipe;           //初始化管道类
    rs2::config cfg;              //配置类

    float depth_scale;            //深度缩放因子
    int width, height;            //图像宽高
    float depth_val;              //深度值

    void RS_init(void);
    void RS_get_img(void);
    void RS_get_data(void);
    
    private:

    rs2::frameset frames;         //帧集类

};


#endif 