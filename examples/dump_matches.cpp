// dump_matches: ALIKED + LightGlue inference that dumps keypoints, matches
// and scores to an OpenCV FileStorage YAML for offline comparison scripts.
#include "feature/aliked.hpp"
#include "matcher/lightglue/matcher.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

cv::Mat load_image(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("Failed to load image: " + path);
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    return img_rgb;
}

// Convert a contiguous CPU float32 tensor (Nx2) to a cv::Mat (Nx2, CV_32F).
cv::Mat tensor_to_mat(const torch::Tensor& t, int rows = -1) {
    auto flat = t.reshape({-1}).contiguous();
    if (flat.dtype() != torch::kFloat32)
        flat = flat.to(torch::kFloat32);
    const int64_t n = rows >= 0 ? rows : flat.size(0) / 2;
    cv::Mat m(n, 2, CV_32F);
    std::memcpy(m.ptr<float>(), flat.data_ptr<float>(), n * 2 * sizeof(float));
    return m;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> <output.yml> [n_runs] [resize]" << std::endl;
        std::cerr << "  resize: longest-side target; <0 = extractor default (1024), 0 = native" << std::endl;
        return 1;
    }
    const std::string out_path = argv[3];
    const int n_runs = argc > 4 ? std::atoi(argv[4]) : 1;
    const int resize = argc > 5 ? std::atoi(argv[5]) : -1; // <0: extractor default

    try {
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto extractor = std::make_shared<ALIKED>("aliked-n16rot", device.str());
        auto matcher = std::make_shared<matcher::LightGlue>();
        matcher->to(device);

        cv::Mat img0 = load_image(argv[1]);
        cv::Mat img1 = load_image(argv[2]);

        torch::Dict<std::string, torch::Tensor> feats0, feats1, matches01;
        double extract_ms = 0.0, match_ms = 0.0, total_ms = 0.0;
        int n_timed = 0;
        for (int run = 0; run < n_runs; ++run) {
            feats0.clear();
            feats1.clear();
            matches01.clear();
            torch::cuda::synchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            feats0 = extractor->run(img0, resize);
            feats1 = extractor->run(img1, resize);
            torch::cuda::synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            feats0.insert("image_size",
                          torch::tensor({static_cast<float>(img0.cols), static_cast<float>(img0.rows)},
                                        torch::kFloat32).unsqueeze(0));
            feats1.insert("image_size",
                          torch::tensor({static_cast<float>(img1.cols), static_cast<float>(img1.rows)},
                                        torch::kFloat32).unsqueeze(0));
            matches01 = matcher->forward(feats0, feats1);
            torch::cuda::synchronize();
            auto t2 = std::chrono::high_resolution_clock::now();
            if (run > 0 || n_runs == 1) { // run 0 is a warmup when n_runs > 1
                extract_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                match_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
                ++n_timed;
            }
        }
        extract_ms /= n_timed;
        match_ms /= n_timed;
        total_ms /= n_timed;

        const auto& kpts0 = feats0.at("keypoints").cpu();
        const auto& kpts1 = feats1.at("keypoints").cpu();
        const auto& matches = matches01.at("matches0").cpu();
        const auto& scores = matches01.at("matching_scores0").cpu();
        const auto stop_layer = matches01.at("stop").item<int64_t>();
        const auto valid_matches = (matches != -1).sum().item<int64_t>();

        // Build correspondence arrays for the valid (threshold-filtered) matches.
        std::vector<float> pts0, pts1, mscores;
        auto matches_acc = matches.accessor<int64_t, 2>();
        auto scores_acc = scores.accessor<float, 2>();
        for (int i = 0; i < matches.size(1); ++i) {
            const int64_t j = matches_acc[0][i];
            if (j < 0)
                continue;
            pts0.push_back(kpts0[i][0].item<float>());
            pts0.push_back(kpts0[i][1].item<float>());
            pts1.push_back(kpts1[j][0].item<float>());
            pts1.push_back(kpts1[j][1].item<float>());
            mscores.push_back(scores_acc[0][i]);
        }

        cv::Mat m0(pts0.size() / 2, 2, CV_32F, pts0.data());
        cv::Mat m1(pts1.size() / 2, 2, CV_32F, pts1.data());
        cv::Mat msc(mscores.size(), 1, CV_32F, mscores.data());
        cv::Mat all_kpts0 = tensor_to_mat(kpts0);
        cv::Mat all_kpts1 = tensor_to_mat(kpts1);

        cv::FileStorage fs(out_path, cv::FileStorage::WRITE);
        fs << "n_keypoints0" << (int)kpts0.size(0);
        fs << "n_keypoints1" << (int)kpts1.size(0);
        fs << "n_valid_matches" << (int)valid_matches;
        fs << "stop_layer" << (int)stop_layer;
        fs << "extract_ms" << extract_ms;
        fs << "match_ms" << match_ms;
        fs << "total_ms" << total_ms;
        fs << "keypoints0" << all_kpts0;
        fs << "keypoints1" << all_kpts1;
        fs << "points0" << m0.clone();
        fs << "points1" << m1.clone();
        fs << "scores" << msc.clone();
        fs.release();

        std::cout << "keypoints0 " << kpts0.size(0)
                  << " keypoints1 " << kpts1.size(0)
                  << " valid_matches " << valid_matches
                  << " stop_layer " << stop_layer
                  << " extract_ms " << extract_ms
                  << " match_ms " << match_ms
                  << " total_ms " << total_ms << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}