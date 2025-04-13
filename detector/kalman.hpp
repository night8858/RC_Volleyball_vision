#pragma once

#include <iostream>
#include <Eigen/Dense> // 使用Eigen库进行矩阵运算


class SimpleKalmanFilter2D {
public:
    /**
     * @brief 构造函数，初始化卡尔曼滤波器
     * @param initialPosition 初始位置 (x, y)
     * @param processNoise 过程噪声 (较小的值使滤波器更信任模型)
     * @param measurementNoise 测量噪声 (较小的值使滤波器更信任测量)
     * @param initialUncertainty 初始不确定性
     */
    SimpleKalmanFilter2D(const Eigen::Vector2d& initialPosition,
                        double processNoise = 0.01,
                        double measurementNoise = 0.1,
                        double initialUncertainty = 1.0) :
        state_(initialPosition),
        processNoiseCov_(Eigen::Matrix2d::Identity() * processNoise),
        measurementNoiseCov_(Eigen::Matrix2d::Identity() * measurementNoise),
        covariance_(Eigen::Matrix2d::Identity() * initialUncertainty) {}
    
    /**
     * @brief 预测步骤
     * @param dt 时间步长 (可选，如果不提供则使用默认模型)
     */
    void predict(double dt = 1.0) {
        // 这里使用简单的恒定位置模型
        // 状态转移矩阵 (假设位置基本保持不变)
        Eigen::Matrix2d transitionMatrix = Eigen::Matrix2d::Identity();
        
        // 预测状态
        state_ = transitionMatrix * state_;
        
        // 预测协方差
        covariance_ = transitionMatrix * covariance_ * transitionMatrix.transpose() + processNoiseCov_;
    }
    
    /**
     * @brief 更新步骤
     * @param measurement 测量位置 (x, y)
     */
    void update(const Eigen::Vector2d& measurement) {
        // 计算卡尔曼增益
        Eigen::Matrix2d K = covariance_ * (covariance_ + measurementNoiseCov_).inverse();
        
        // 更新状态估计
        state_ = state_ + K * (measurement - state_);
        
        // 更新协方差估计
        Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        covariance_ = (I - K) * covariance_;
    }
    
    /**
     * @brief 获取当前位置估计
     * @return 2D向量 (x, y)
     */
    Eigen::Vector2d getPosition() const {
        return state_;
    }
    
private:
    Eigen::Vector2d state_;          // 状态向量 [x, y]^T
    Eigen::Matrix2d covariance_;     // 状态协方差矩阵
    Eigen::Matrix2d processNoiseCov_;    // 过程噪声协方差
    Eigen::Matrix2d measurementNoiseCov_; // 测量噪声协方差
};