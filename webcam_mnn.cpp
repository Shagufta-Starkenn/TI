// ultraface_webcam.cpp
#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

// ----------------- FaceInfo -----------------
struct FaceInfo {
    float x1, y1, x2, y2, score;
};

// ----------------- UltraFace Class -----------------
class UltraFace {
public:
    UltraFace(const std::string &mnn_path,
              int input_w = 320, int input_h = 240,
              int num_thread = 4,
              float score_thresh = 0.7f,
              float iou_thresh = 0.3f)
        : in_w(input_w), in_h(input_h),
          score_threshold(score_thresh),
          iou_threshold(iou_thresh)
    {
        // Build priors
        const std::vector<std::vector<float>> min_boxes = {
            {10.f, 16.f, 24.f}, {32.f, 48.f}, {64.f, 96.f}, {128.f,192.f,256.f}
        };
        const std::vector<float> strides = {8.f,16.f,32.f,64.f};

        for (int idx = 0; idx < (int)min_boxes.size(); idx++) {
            int fm_w = std::ceil(in_w / strides[idx]);
            int fm_h = std::ceil(in_h / strides[idx]);
            for (int y = 0; y < fm_h; y++)
            for (int x = 0; x < fm_w; x++)
                for (float m : min_boxes[idx]) {
                    float cx = (x + 0.5f) * strides[idx] / in_w;
                    float cy = (y + 0.5f) * strides[idx] / in_h;
                    float w  = m / in_w;
                    float h  = m / in_h;
                    priors.emplace_back(cv::Vec4f(cx, cy, w, h));
                }
        }
        num_anchors = priors.size();

        // Create MNN interpreter & session
        interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));
        MNN::ScheduleConfig config;
        config.numThread = num_thread;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        config.backendConfig = &backendConfig;
        session = interpreter->createSession(config);

        input_tensor = interpreter->getSessionInput(session, nullptr);
    }

    ~UltraFace() {
        interpreter->releaseSession(session);
        interpreter->releaseModel();
    }

    std::vector<FaceInfo> detect(const cv::Mat &raw) {
        cv::Mat img;
        cv::resize(raw, img, cv::Size(in_w, in_h));

        // Preprocess
        interpreter->resizeTensor(input_tensor, {1,3,in_h,in_w});
        interpreter->resizeSession(session);
        auto proc = std::shared_ptr<MNN::CV::ImageProcess>(
            MNN::CV::ImageProcess::create(
                MNN::CV::BGR, MNN::CV::RGB,
                mean_vals, 3,
                norm_vals, 3
            )
        );
        proc->convert(img.data, in_w, in_h, img.step[0], input_tensor);

        // Inference
        interpreter->runSession(session);

        // Fetch outputs
        auto ts = interpreter->getSessionOutput(session, "scores");
        auto tb = interpreter->getSessionOutput(session, "boxes");
        MNN::Tensor scores(ts, ts->getDimensionType());
        MNN::Tensor boxes(tb, tb->getDimensionType());
        ts->copyToHostTensor(&scores);
        tb->copyToHostTensor(&boxes);

        // Decode
        std::vector<FaceInfo> boxes_out;
        for (int i = 0; i < num_anchors; i++) {
            float sc = scores.host<float>()[2*i+1];
            if (sc < score_threshold) continue;

            auto p = priors[i];
            float cx = p[0] + boxes.host<float>()[4*i+0]*0.1f*p[2];
            float cy = p[1] + boxes.host<float>()[4*i+1]*0.1f*p[3];
            float w  = p[2]*std::exp(boxes.host<float>()[4*i+2]*0.2f);
            float h  = p[3]*std::exp(boxes.host<float>()[4*i+3]*0.2f);

            float x1 = clip(cx - w/2, 1.f)*raw.cols;
            float y1 = clip(cy - h/2, 1.f)*raw.rows;
            float x2 = clip(cx + w/2, 1.f)*raw.cols;
            float y2 = clip(cy + h/2, 1.f)*raw.rows;
            boxes_out.push_back({x1,y1,x2,y2,sc});
        }

        // NMS
        std::sort(boxes_out.begin(), boxes_out.end(),
                  [](const FaceInfo &a, const FaceInfo &b){ return a.score > b.score; });
        std::vector<FaceInfo> result;
        std::vector<bool> used(boxes_out.size());
        for (size_t i = 0; i < boxes_out.size(); i++) if (!used[i]) {
            auto &A = boxes_out[i];
            result.push_back(A);
            used[i] = true;
            float areaA = (A.x2 - A.x1)*(A.y2 - A.y1);
            for (size_t j = i+1; j < boxes_out.size(); j++) if (!used[j]) {
                auto &B = boxes_out[j];
                float ix1 = std::max(A.x1, B.x1),
                      iy1 = std::max(A.y1, B.y1),
                      ix2 = std::min(A.x2, B.x2),
                      iy2 = std::min(A.y2, B.y2);
                float iw = std::max(0.f, ix2-ix1),
                      ih = std::max(0.f, iy2-iy1),
                      inter = iw*ih,
                      areaB = (B.x2-B.x1)*(B.y2-B.y1);
                if (inter / (areaA+areaB-inter) > iou_threshold)
                    used[j] = true;
            }
        }
        return result;
    }

private:
    int in_w, in_h, num_anchors;
    float score_threshold, iou_threshold;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    std::vector<cv::Vec4f> priors;

    const float mean_vals[3] = {127,127,127};
    const float norm_vals[3] = {1/128.f,1/128.f,1/128.f};
};

// ----------------- main() -----------------
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.mnn>\n";
        return 1;
    }
    UltraFace detector(argv[1]);

    cv::VideoCapture cap(3);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam\n";
        return 1;
    }

    cv::Mat frame;
 //   cv::imwrite("frames", frame);
   int frame_count = 0;
    while (cap.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();
//	cv::imwrite(frame,frame);
        auto faces = detector.detect(frame);
        for (auto &f : faces) {
            cv::rectangle(frame,
                          cv::Point(int(f.x1),int(f.y1)),
                          cv::Point(int(f.x2),int(f.y2)),
                          cv::Scalar(0,255,0),2);
            cv::putText(frame,
                        cv::format("%.2f",f.score),
                        cv::Point(int(f.x1),int(f.y1)-5),
                        cv::FONT_HERSHEY_SIMPLEX,0.5,
                        cv::Scalar(0,255,0),1);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float fps = 1e9f / std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        cv::putText(frame, cv::format("FPS: %.1f", fps),
                    cv::Point(10,30),
                    cv::FONT_HERSHEY_SIMPLEX,1,
                    cv::Scalar(0,255,255),2);

	std::string filename = cv::format("frame_%04d.jpg", frame_count++);
        cv::imwrite(filename, frame);
        //cv::imshow("UltraFace MNN Webcam", frame);
        //if (cv::waitKey(1) == 27) break;  // ESC
    }
    return 0;
}

