// bench_aliked: decompose ALIKED extraction timing into upload / backbone / (DKD+SDDH).
#include "feature/aliked.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <string>

cv::Mat load_rgb(const std::string& path) {
    cv::Mat img = cv::imread(path);
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

torch::Tensor upload(const cv::Mat& img_rgb, const torch::Device& device) {
    cv::Mat float_img;
    img_rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    std::vector<torch::Tensor> tensor_channels;
    for (const auto& channel : channels)
        tensor_channels.push_back(torch::from_blob(channel.data, {channel.rows, channel.cols},
                                                   torch::TensorOptions().dtype(torch::kFloat32)).to(device));
    return torch::stack(std::move(tensor_channels), 0).unsqueeze(0).to(device);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [n_runs]" << std::endl;
        return 1;
    }
    const int n = argc > 3 ? std::atoi(argv[3]) : 10;
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    ALIKED model("aliked-n16rot", device.str());

    cv::Mat img0 = load_rgb(argv[1]);
    cv::Mat img1 = load_rgb(argv[2]);

    // warmup (fresh tensors per pass)
    for (int i = 0; i < 3; ++i) {
        auto a0 = upload(img0, device);
        auto a1 = upload(img1, device);
        auto f = std::move(model).forward(std::move(a0));
        auto b0 = upload(img0, device);
        auto d = std::move(model).extract_dense_map(std::move(b0));
    }
    torch::cuda::synchronize();

    double up_ms = 0, fwd_ms = 0, dense_ms = 0;
    for (int i = 0; i < n; ++i) {
        // phase A: upload both images
        torch::cuda::synchronize();
        auto a = std::chrono::high_resolution_clock::now();
        auto t0 = upload(img0, device);
        auto t1 = upload(img1, device);
        torch::cuda::synchronize();
        auto b = std::chrono::high_resolution_clock::now();
        up_ms += std::chrono::duration<double, std::milli>(b - a).count();
        (void)t0; (void)t1;

        // phase B: full forward on fresh copies
        auto f0 = upload(img0, device);
        auto f1 = upload(img1, device);
        torch::cuda::synchronize();
        auto c = std::chrono::high_resolution_clock::now();
        auto r0 = std::move(model).forward(std::move(f0));
        auto r1 = std::move(model).forward(std::move(f1));
        torch::cuda::synchronize();
        auto d0 = std::chrono::high_resolution_clock::now();
        fwd_ms += std::chrono::duration<double, std::milli>(d0 - c).count();
        (void)r0; (void)r1;

        // phase C: dense map only on fresh copies
        auto g0 = upload(img0, device);
        auto g1 = upload(img1, device);
        torch::cuda::synchronize();
        auto e = std::chrono::high_resolution_clock::now();
        auto p0 = std::move(model).extract_dense_map(std::move(g0));
        auto p1 = std::move(model).extract_dense_map(std::move(g1));
        torch::cuda::synchronize();
        auto f2 = std::chrono::high_resolution_clock::now();
        dense_ms += std::chrono::duration<double, std::milli>(f2 - e).count();
        (void)p0; (void)p1;
    }
    const double fwd = fwd_ms / n, dense = dense_ms / n, up = up_ms / n;
    std::cout << "upload_ms " << up << " forward_ms " << fwd
              << " dense_map_ms " << dense
              << " dkd_sddh_ms " << (fwd - dense) << std::endl;
    return 0;
}