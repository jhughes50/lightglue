#pragma once

#include "blocks.hpp"
#include "input_padder.hpp"
#include "feature/image_preprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>

struct AlikedConfig {
    int c1, c2, c3, c4, dim, K, M;
};

class DKD;
class SDDH;

// Static configuration map
inline const std::unordered_map<std::string_view, AlikedConfig> ALIKED_CFGS = {
    {"aliked-t16", {8, 16, 32, 64, 64, 3, 16}},
    {"aliked-n16", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n16rot", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n32", {16, 32, 64, 128, 128, 3, 32}}};

class ALIKED : public torch::nn::Module {
public:
    explicit ALIKED(std::string_view model_name = "aliked-n32",
                    std::string_view device = "cuda",
                    int top_k = -1,
                    float scores_th = 0.2,
                    int n_limit = 20000,
                    int preprocess_resize = 1024);

    // Move semantics for tensor operations
    std::tuple<torch::Tensor, torch::Tensor>
    extract_dense_map(torch::Tensor image) &&;

    torch::Dict<std::string, torch::Tensor>
    forward(torch::Tensor image) &&;

    torch::Dict<std::string, torch::Tensor>
    forward(const torch::Tensor& image) &;

    // run() extracts features for an OpenCV RGB image.
    //   resize > 0 : longest-side resize target
    //   resize == 0: native resolution
    //   resize < 0 : the constructor's preprocess target (default 1024)
    // Keypoints are always returned in original image pixel coordinates.
    torch::Dict<std::string, torch::Tensor>
    run(cv::Mat& img_rgb, int resize = -1);

private:
    void init_layers(std::string_view model_name);
    void load_weights(std::string_view model_name);
    void load_parameters(std::string_view pt_pth);

    static std::vector<char> get_the_bytes(std::string_view filename);

    torch::nn::AvgPool2d pool2_{nullptr}, pool4_{nullptr};
    std::shared_ptr<ConvBlock> block1_;
    std::shared_ptr<ResBlock> block2_;
    std::shared_ptr<ResBlock> block3_;
    std::shared_ptr<ResBlock> block4_;
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr},
        conv3_{nullptr}, conv4_{nullptr};
    torch::nn::Sequential score_head_{nullptr};

    std::shared_ptr<DKD> dkd_;
    std::shared_ptr<SDDH> desc_head_;

    torch::Device device_;
    int dim_{};
    int preprocess_resize_{1024};
};