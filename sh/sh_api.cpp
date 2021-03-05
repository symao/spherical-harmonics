#include "sh_api.h"

#include "image.h"
#include "spherical_harmonics.h"

namespace sh {

class CvImage : public Image {
public:
    CvImage(const cv::Mat& img) : img_(img) {}

    int width() const override { return img_.cols; }
    int height() const override { return img_.rows; }

    Eigen::Array3f GetPixel(int x, int y) const override {
        switch (img_.depth()) {
        case CV_8U: {
            auto val = img_.at<cv::Vec3b>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        case CV_16U: {
            auto val = img_.at<cv::Vec3w>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        case CV_16S: {
            auto val = img_.at<cv::Vec3s>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        case CV_32S: {
            auto val = img_.at<cv::Vec3i>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        case CV_32F: {
            auto val = img_.at<cv::Vec3f>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        case CV_64F: {
            auto val = img_.at<cv::Vec3d>(y, x);
            return Eigen::Array3f(val[0], val[1], val[2]);
        }
        default:
            break;
        }
        return Eigen::Array3f(0, 0, 0);
    }

    void SetPixel(int x, int y, const Eigen::Array3f& v) override {
        switch (img_.depth()) {
        case CV_8U:
            img_.at<cv::Vec3b>(y, x) = cv::Vec3b(v.x(), v.y(), v.z());
            break;
        case CV_16U:
            img_.at<cv::Vec3w>(y, x) = cv::Vec3w(v.x(), v.y(), v.z());
            break;
        case CV_16S:
            img_.at<cv::Vec3s>(y, x) = cv::Vec3s(v.x(), v.y(), v.z());
            break;
        case CV_32S:
            img_.at<cv::Vec3i>(y, x) = cv::Vec3i(v.x(), v.y(), v.z());
            break;
        case CV_32F:
            img_.at<cv::Vec3f>(y, x) = cv::Vec3f(v.x(), v.y(), v.z());
            break;
        case CV_64F:
            img_.at<cv::Vec3d>(y, x) = cv::Vec3d(v.x(), v.y(), v.z());
            break;
        default:
            break;
        }
    }

private:
    cv::Mat img_;
};

}  // namespace sh

cv::Mat lightprobe2equirectangular(const cv::Mat& probe) {
    int radius = probe.rows / 2;
    int eh = probe.rows;
    int ew = eh * 2;
    float delta = M_PI / eh;

    cv::Mat map_x(eh, ew, CV_32FC1);
    cv::Mat map_y(eh, ew, CV_32FC1);
    float* ptr_map_x = (float*)map_x.data;
    float* ptr_map_y = (float*)map_y.data;
    float theta = 0;
    cv::Vec3f V(-1, 0, 0);
    for (int i = 0; i < eh; i++, theta += delta) {
        float st = sin(theta);
        float ct = cos(theta);
        float phi = 0;
        for (int j = 0; j < ew; j++, phi += delta) {
            float st = sin(theta);
            cv::Vec3f R(st * cos(phi), st * sin(phi), ct);
            cv::Vec3f N = R - V;
            float normN = cv::norm(N);
            if (normN < 1e-4) {
                *ptr_map_x++ = radius;
                *ptr_map_y++ = radius;
            } else {
                *ptr_map_x++ = (1 - N[1] / normN) * radius;
                *ptr_map_y++ = (1 - N[2] / normN) * radius;
            }
        }
    }
    cv::Mat res;
    cv::remap(probe, res, map_x, map_y, cv::INTER_LINEAR);
    return res;
}

cv::Mat equirectangular2lightprobe(const cv::Mat& equirectangle) {
    int radius = equirectangle.rows / 2;
    int h = equirectangle.rows;
    int w = h;
    float delta = M_PI / equirectangle.rows;

    cv::Vec3f V(-1, 0, 0);
    cv::Mat map_x(h, w, CV_32FC1);
    cv::Mat map_y(h, w, CV_32FC1);
    float* ptr_map_x = (float*)map_x.data;
    float* ptr_map_y = (float*)map_y.data;
    for (int i = 0; i < h; i++) {
        float nz = 1 - (float)i / radius;
        for (int j = 0; j < w; j++) {
            float ny = 1 - (float)j / radius;
            float nx2 = 1 - ny * ny - nz * nz;
            if (nx2 < 0) {
                *ptr_map_x++ = -1;
                *ptr_map_y++ = -1;
            } else {
                float nx = std::sqrt(nx2);
                cv::Vec3f N(nx, ny, nz);
                cv::Vec3f R = V - 2 * V.dot(N) * N;
                double theta, phi;
                sh::ToSphericalCoords(Eigen::Vector3d(R[0], R[1], R[2]), &phi, &theta);
                auto xy = sh::ToImageCoords(phi, theta, equirectangle.cols, equirectangle.rows);
                *ptr_map_x++ = xy.x();
                *ptr_map_y++ = xy.y();
            }
        }
    }
    cv::Mat res;
    cv::remap(equirectangle, res, map_x, map_y, cv::INTER_LINEAR);
    return res;
}

std::vector<cv::Vec3f> envmap2sh(const cv::Mat& equirectangle, int order) {
    sh::CvImage env(equirectangle);
    auto sh_ptr = sh::ProjectEnvironment(order, env);
    std::vector<cv::Vec3f> res;
    res.reserve(sh_ptr->size());
    for (const auto& v : *sh_ptr) {
        res.push_back(cv::Vec3f(v.x(), v.y(), v.z()));
    }
    return res;
}

cv::Mat sh2envmap(const std::vector<cv::Vec3f>& sh_coeff, const cv::Size& img_size, int order) {
    std::vector<Eigen::Array3f> coeff;
    coeff.reserve(sh_coeff.size());
    for (const auto& v : sh_coeff) {
        coeff.push_back(Eigen::Array3f(v[0], v[1], v[2]));
    }
    cv::Mat sh_equirectangle(img_size, CV_8UC3);
    uchar* ptr_sh_equirectangle = sh_equirectangle.data;
    float delta = M_PI / sh_equirectangle.rows;
    float theta = 0;
    for (int i = 0; i < sh_equirectangle.rows; i++, theta += delta) {
        float phi = 0;
        for (int j = 0; j < sh_equirectangle.cols; j++, phi += delta) {
            auto val = sh::EvalSHSum(order, coeff, phi, theta);
            *ptr_sh_equirectangle++ = std::max(0.0f, std::min((float)val[0], 255.0f));
            *ptr_sh_equirectangle++ = std::max(0.0f, std::min((float)val[1], 255.0f));
            *ptr_sh_equirectangle++ = std::max(0.0f, std::min((float)val[2], 255.0f));
        }
    }
    return sh_equirectangle;
}

void sh2directionalLight(const std::vector<cv::Vec3f>& sh_coeff, float& intensity, cv::Vec3f& color,
                         cv::Vec3f& direction, const std::vector<float>& channel_weight) {
    std::vector<cv::Vec3f> dirs = {
        cv::Vec3f(-sh_coeff[3][0], -sh_coeff[1][0], sh_coeff[2][0]),
        cv::Vec3f(-sh_coeff[3][1], -sh_coeff[1][1], sh_coeff[2][1]),
        cv::Vec3f(-sh_coeff[3][2], -sh_coeff[1][2], sh_coeff[2][2]),
    };
    for (auto& dir : dirs) {
        dir /= cv::norm(dir);
    }
    direction =
        channel_weight[0] * dirs[0] + channel_weight[1] * dirs[1] + channel_weight[2] * dirs[2];
    direction /= cv::norm(direction);

    // todo: intensity

    // todo: color
    
}