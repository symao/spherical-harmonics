#ifndef __SH_API_H__
#define __SH_API_H__
#include <opencv2/opencv.hpp>

cv::Mat lightprobe2equirectangular(const cv::Mat& probe);

cv::Mat equirectangular2lightprobe(const cv::Mat& equirectangle);

std::vector<cv::Vec3f> envmap2sh(const cv::Mat& equirectangle, int order = 2);

cv::Mat sh2envmap(const std::vector<cv::Vec3f>& sh_coeff, const cv::Size& img_size, int order = 2);

void sh2directionalLight(const std::vector<cv::Vec3f>& sh_coeff, float& intensity, cv::Vec3f& color,
                         cv::Vec3f& direction,
                         const std::vector<float>& channel_weight = {0.333f, 0.333f, 0.333f});

#endif  //__SH_API_H__