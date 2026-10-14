#include "feature/image_preprocessor.hpp"

#include <algorithm>

namespace feature {

ImagePreprocessor::ImagePreprocessor(int target)
    : target_(target) {}

int ImagePreprocessor::target() const {
    return target_;
}

void ImagePreprocessor::set_target(int target) {
    target_ = target;
}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat& image) {
    scale_x_ = 1.0;
    scale_y_ = 1.0;
    resized_ = false;

    if (target_ <= 0 || image.empty())
        return image;

    const int longest = std::max(image.rows, image.cols);
    if (longest == target_)
        return image;

    const double s = static_cast<double>(target_) / longest;
    const int rw = std::max(1, static_cast<int>(std::lround(image.cols * s)));
    const int rh = std::max(1, static_cast<int>(std::lround(image.rows * s)));
    const int interp = s < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(rw, rh), 0, 0, interp);

    scale_x_ = static_cast<double>(rw) / image.cols;
    scale_y_ = static_cast<double>(rh) / image.rows;
    resized_ = true;
    return resized;
}

torch::Tensor ImagePreprocessor::postprocess(torch::Tensor keypoints) const {
    if (!resized_)
        return keypoints;
    const auto scale = torch::tensor(
        {static_cast<float>(scale_x_), static_cast<float>(scale_y_)},
        keypoints.options());
    return (keypoints + 0.5) / scale - 0.5;
}

}  // namespace feature