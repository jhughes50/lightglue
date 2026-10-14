#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace feature {

class ImagePreprocessor {
public:
    static constexpr int kNoResize = 0;

    explicit ImagePreprocessor(int target = 1024);

    int target() const;
    void set_target(int target);

    cv::Mat preprocess(const cv::Mat& image);
    torch::Tensor postprocess(torch::Tensor keypoints) const;

private:
    int target_;
    double scale_x_ = 1.0;
    double scale_y_ = 1.0;
    bool resized_ = false;
};

}  // namespace feature