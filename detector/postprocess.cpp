#include "postprocess.hpp"

#include <vector>
#include "common_struct.hpp"

/**
 * @brief 将归一化的边界框坐标转换为原始图像上的矩形区域
 * @param img 输入图像
 * @param bbox 归一化的边界框坐标数组[x1,y1,x2,y2]
 * @return cv::Rect 返回在原始图像尺寸下的矩形区域
 *
 * 该函数处理检测模型输出的640x640尺寸下的边界框坐标,
 * 将其转换回原始输入图像尺寸下的矩形区域。
 * 考虑了保持宽高比的缩放处理,确保坐标在有效范围内。
 */
cv::Rect get_rect(cv::Mat &img, float bbox[4])
{
    // 解析检测框
    float l, r, t, b;
    float r_w = 640 / (img.cols * 1.0);
    float r_h = 640 / (img.rows * 1.0);

    if (r_h > r_w)
    {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (640 - r_w * img.rows) / 2;
        b = bbox[3] - (640 - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - (640 - r_h * img.cols) / 2;
        r = bbox[2] - (640 - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

/**
 * @brief 从网络输出解码检测结果
 * @param res 检测结果输出参数
 * @param output 网络输出数据指针
 * @param conf_thresh 置信度阈值
 * @param tokp 检测框数量上限
 *
 * 将网络输出的原始数据解码为检测结果。遍历所有可能的检测框,
 * 对于置信度大于阈值的检测框,将其数据复制到结果结构体中。
 */
void decode(Detection &res, float *output, float conf_thresh, int tokp)
{
    // 初始化res为全零
    memset(&res, 0, sizeof(Detection));
    
    int det_size = sizeof(Detection) / sizeof(float);
    bool has_valid_detection = false;

    for (int i = 0; i < tokp; i++)
    {
        // 检查置信度是否超过阈值
        if (output[det_size * i + 4] > conf_thresh)
        {
            // 复制有效检测数据到res
            memcpy(&res, &output[det_size * i], det_size * sizeof(float));
            has_valid_detection = true;
            break;  // 找到第一个有效检测后即可退出循环
        }
    }

    // 如果没有有效检测，确保res保持全零状态
    if (!has_valid_detection)
    {
        memset(&res, 0, sizeof(Detection));
    }
    
}

void decode2(Detection &res, float *output, float conf_thresh, int tokp)
{
    memset(&res, 0, sizeof(Detection));
    
    int det_size = sizeof(Detection) / sizeof(float);
    bool has_valid_detection = false;

    for (int i = 0; i < tokp; i++)
    {
        if (output[1800 + det_size * i + 4] > conf_thresh)
        {
            // 复制有效检测数据到res
            memcpy(&res, &output[1800 + det_size * i], det_size * sizeof(float));
            has_valid_detection = true;
            break;  // 找到第一个有效检测后即可退出循环
        }
    }
        // 如果没有有效检测，确保res保持全零状态
        if (!has_valid_detection)
        {
            memset(&res, 0, sizeof(Detection));
        }
}
/**
 * @brief 解码两个检测结果的函数
 * @param res1 第一个检测结果的引用
 * @param res2 第二个检测结果的引用
 * @param output 模型输出的浮点数数组
 * @param conf_thresh 置信度阈值
 * @param tokp 需要处理的检测框总数
 *
 * 该函数将模型输出解码为两个Detection对象。对于置信度大于阈值的检测框，
 * 将前300个框的数据复制到res1，后续框的数据复制到res2。
 */
void double_decode(Detection &res1, Detection &res2, float *output, float conf_thresh, int tokp)
{
    int det_size = sizeof(Detection) / sizeof(float);
    for (int i = 0; i < tokp; i++)
    {
        // 置信度处理
        if (output[det_size * i + 4] <= conf_thresh)
        {
            
            continue;
        }
        Detection det{};
        // 复制数据到res1
        memcpy(&res1, &output[det_size * i], det_size * sizeof(float));
    }
    //300是模型的输出最大框数目
    for (int i = 300; i < tokp; i++)
    {
        // 置信度处理
        if (output[det_size * i + 4] <= conf_thresh)
        {
            
            continue;
        }
        Detection det{};
        // 复制数据到res2
        memcpy(&res2, &output[det_size * i], det_size * sizeof(float));
    }
}

/**
 * @brief 从检测框中提取排球信息
 * @param getted_frame 输入图像帧
 * @param volley 排球检测结果
 * @return volleyball 包含排球位置和大小信息的结构体
 *
 * 该函数将检测框转换为排球结构体，提取球的中心坐标和半径信息。
 * 球的半径取检测框宽高中的较小值的一半。
 */
volleyball get_ball(cv::Mat &getted_frame, Detection &volley)
{
    // 结构体
    volleyball temp;
    cv::Rect ball = get_rect(getted_frame, volley.bbox);
    temp.center_x = ball.x;
    temp.center_y = ball.y;
    temp.radius = std::max(ball.width, ball.height) / 2 ;
///    temp.deepth = temp.radius * 0.04;
    if(ball.x == 0 || ball.y == 0)
    {
        temp.isValid = false;
    }else
    {
        temp.isValid = true;
    }
    return temp;
}

void draw_bbox(cv::Mat &getted_frame, std::vector<Detection> &res_batch)
{

    auto &res = res_batch;
    cv::Mat img = getted_frame;

    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 1);
    }

}
// void get_results(std::vector<Detection> &res, float *output, float conf_thresh, int tokp)
// {
//     int det_size = sizeof(Detection) / sizeof(float);
//     for (int i = 0; i < tokp; i++)
//     {
//         // 置信度处理
//         if (output[det_size * i + 4] <= conf_thresh)
//             continue;
//         Detection det{};
//         // 复制数据到res
//         memcpy(&det, &output[det_size * i], det_size * sizeof(float));
//         res.push_back(det);
//     }
    
// }

// void topk(std::vector<Detection> &res_batch, float *output, int output_size,
//           float conf_thresh, int topk)
// {
//     get_results(res_batch, &output[output_size], conf_thresh, topk);
// }

// 画框


void draw_bbox_single(cv::Mat &getted_frame, Detection &res_batch)
{

    auto &res = res_batch;
    cv::Mat img = getted_frame;
    // for (size_t j = 0; j < res.size(); j++)
    // {
    cv::Rect r = get_rect(img, res.bbox);
    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    cv::putText(img, std::to_string(res.conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                cv::Scalar(0xFF, 0xFF, 0xFF), 1);
                res_batch = {0};
    // }
}

