#include <opencv2/opencv.hpp>

#include "sh/sh_api.h"

int main(int argc, char** argv) {
    cv::Mat probe = cv::imread(argv[1]);
    cv::resize(probe, probe, cv::Size(128, 128));
    cv::Mat equirectangle = lightprobe2equirectangular(probe);

    int order = 2;
    for (int order = 0; order < 100; order++) {
        auto sh_coeff = envmap2sh(equirectangle, order);
        cv::Mat sh_equirectangle = sh2envmap(sh_coeff, equirectangle.size(), order);
        cv::Mat sh_probe = equirectangular2lightprobe(sh_equirectangle);

        float intensity;
        cv::Vec3f directional_light_dir, directional_light_color;
        sh2directionalLight(sh_coeff, intensity, directional_light_color, directional_light_dir);

        printf("Spherical Harmonics, order:%d coeff:[%dx3] dir:(%.3f %.3f %.3f)\n", order,
               sh_coeff.size(), directional_light_dir[0], directional_light_dir[1],
               directional_light_dir[2]);
        // for (const auto& s : *sh_ptr) {
        //     printf("%.4f %.4f %.4f\n", s.x(), s.y(), s.z());
        // }

        cv::Mat debug_img;
        cv::hconcat(probe, equirectangle, debug_img);
        cv::hconcat(debug_img, sh_equirectangle, debug_img);
        cv::hconcat(debug_img, sh_probe, debug_img);
        const auto font = cv::FONT_HERSHEY_SIMPLEX;
        const cv::Scalar color(0, 120, 255);
        int l = 0;
        cv::putText(debug_img, "probe", cv::Point(l, 15), font, 0.6, color, 2);
        l += probe.cols;
        debug_img.colRange(l, l + 1).setTo(cv::Scalar(0, 0, 255));
        cv::putText(debug_img, "equirectangular", cv::Point(l, 15), font, 0.6, color, 2);
        l += equirectangle.cols;
        debug_img.colRange(l, l + 1).setTo(cv::Scalar(0, 0, 255));
        cv::putText(debug_img, "sh-equirectangular", cv::Point(l, 15), font, 0.6, color, 2);
        l += sh_equirectangle.cols;
        debug_img.colRange(l, l + 1).setTo(cv::Scalar(0, 0, 255));
        cv::putText(debug_img, "sh-probe", cv::Point(l, 15), font, 0.6, color, 2);

        char str[128] = {0};
        snprintf(str, 128, "order:%d, coeff:%dx3", order, sh_coeff.size());
        cv::putText(debug_img, str, cv::Point(0, debug_img.rows - 3), font, 0.6, color, 2);

        cv::imshow("sh", debug_img);
        auto key = cv::waitKey();
        if (key == 27)
            break;
    }

    cv::destroyAllWindows();
}