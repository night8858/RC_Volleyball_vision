#include "postprocess.hpp"

#include "common_struct.hpp"

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = 640 / (img.cols * 1.0);
    float r_h = 640 / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (640 - r_w * img.rows) / 2;
        b = bbox[3] - (640 - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
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

void get_topk(std::vector<Detection>& res, float* output, float conf_thresh, int tokp) {
    int det_size = sizeof(Detection) / sizeof(float);
    for (int i = 0; i < output[0]; i++) {
        //置信度处理
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Detection det{};
        //复制数据到res
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        res.push_back(det);
    }
}

void topk(std::vector<Detection>& res_batch, float* output, int output_size,
                float conf_thresh, int topk) {
        get_topk(res_batch, &output[ output_size], conf_thresh, topk);
}

//画框
void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<Detection>& res_batch) {
    for (size_t i = 0; i < img_batch.size(); i++) {
        auto& res = res_batch;
        cv::Mat img = img_batch[i];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }
}