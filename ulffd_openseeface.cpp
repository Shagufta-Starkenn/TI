#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>


// Helper: Clamp (x,y) to image boundaries
void clamp_to_im(int &x, int &y, int width, int height) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
}

// Preprocess function for OpenSeeFace (NCHW format)
cv::Mat preprocess(const cv::Mat &im, const cv::Rect &cropRect, int res, const cv::Mat &std_res, const cv::Mat &mean_res) {
    cv::Mat cropped = im(cropRect).clone();
    cv::cvtColor(cropped, cropped, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(res, res), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32F);
    cv::Mat normalized;
    cv::multiply(resized, std_res, normalized);
    normalized += mean_res;
    // Convert to NCHW format for MNN
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    cv::Mat blob(1, 3 * res * res, CV_32F);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(blob.ptr<float>() + c * res * res, channels[c].ptr<float>(), res * res * sizeof(float));
    }
    return blob.reshape(1, {1, 3, res, res});
}

// Logit helper function
float logit(float p, float factor) {
    float clipped = std::min(std::max(p, 1e-7f), 0.9999999f);
    return std::log(clipped / (1.0f - clipped)) / factor;
}

// Landmarks post-processing
const int RES = 224;
const int OUT_RES = 27;
const int OUT_RES_I = 28;
const float LOGIT_FACTOR = 16.0f;
const int c0 = 66, c1 = 132, c2 = 198;

std::pair<float, cv::Mat> landmarks(const cv::Mat &tensor, const cv::Vec<double, 5> &crop_info) {
    int res_minus1 = RES - 1;
    int grid = OUT_RES_I * OUT_RES_I;
    cv::Mat tensor_mat = tensor.reshape(1, c2);
    cv::Mat t_main = tensor_mat.rowRange(0, c0);
    cv::Mat t_off_x = tensor_mat.rowRange(c0, c1);
    cv::Mat t_off_y = tensor_mat.rowRange(c1, c2);
    std::vector<int> t_m(c0, 0);
    std::vector<float> t_conf(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        cv::Mat row = t_main.row(i);
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(row, nullptr, &maxVal, nullptr, &maxLoc);
        t_m[i] = maxLoc.x;
        t_conf[i] = static_cast<float>(maxVal);
    }
    std::vector<float> off_x(c0, 0.0f), off_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        off_x[i] = t_off_x.at<float>(i, t_m[i]);
        off_y[i] = t_off_y.at<float>(i, t_m[i]);
    }
    for (int i = 0; i < c0; i++) {
        off_x[i] = res_minus1 * logit(off_x[i], LOGIT_FACTOR);
        off_y[i] = res_minus1 * logit(off_y[i], LOGIT_FACTOR);
    }
    std::vector<float> t_x(c0, 0.0f), t_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        float row_idx = std::floor(float(t_m[i]) / float(OUT_RES_I));
        float col_idx = float(t_m[i]) - row_idx * OUT_RES_I;
        double crop_x1 = crop_info[0];
        double crop_y1 = crop_info[1];
        double scale_x = crop_info[2];
        double scale_y = crop_info[3];
        t_x[i] = static_cast<float>(crop_y1) + static_cast<float>(scale_y) * (res_minus1 * row_idx / float(OUT_RES) + off_x[i]);
        t_y[i] = static_cast<float>(crop_x1) + static_cast<float>(scale_x) * (res_minus1 * col_idx / float(OUT_RES) + off_y[i]);
    }
    float sum_conf = 0.0f;
    for (int i = 0; i < c0; i++) {
        sum_conf += t_conf[i];
    }
    float avg_conf = sum_conf / c0;
    cv::Mat lms(c0, 3, CV_32F);
    for (int i = 0; i < c0; i++) {
        lms.at<float>(i, 0) = t_x[i];
        lms.at<float>(i, 1) = t_y[i];
        lms.at<float>(i, 2) = t_conf[i];
    }
    for (int i = 0; i < c0; i++) {
        if (std::isnan(lms.at<float>(i, 0)) || std::isnan(lms.at<float>(i, 1)) || std::isnan(lms.at<float>(i, 2))) {
            lms.at<float>(i, 0) = 0.0f;
            lms.at<float>(i, 1) = 0.0f;
            lms.at<float>(i, 2) = 0.0f;
        }
    }
    return std::make_pair(avg_conf, lms);
}


struct FaceInfo {
    float x1, y1, x2, y2, score;
};

class UltraFace {
public:
    UltraFace(const std::string &mnn_path, int input_w = 320, int input_h = 240,
              int num_thread = 4, float score_thresh = 0.7f, float iou_thresh = 0.3f)
        : in_w(input_w), in_h(input_h), score_threshold(score_thresh), iou_threshold(iou_thresh) {
        const std::vector<std::vector<float>> min_boxes = {
            {10.f, 16.f, 24.f}, {32.f, 48.f}, {64.f, 96.f}, {128.f, 192.f, 256.f}
        };
        const std::vector<float> strides = {8.f, 16.f, 32.f, 64.f};

        for (int idx = 0; idx < (int)min_boxes.size(); idx++) {
            int fm_w = std::ceil(in_w / strides[idx]);
            int fm_h = std::ceil(in_h / strides[idx]);
            for (int y = 0; y < fm_h; y++)
                for (int x = 0; x < fm_w; x++)
                    for (float m : min_boxes[idx]) {
                        float cx = (x + 0.5f) * strides[idx] / in_w;
                        float cy = (y + 0.5f) * strides[idx] / in_h;
                        float w = m / in_w;
                        float h = m / in_h;
                        priors.emplace_back(cv::Vec4f(cx, cy, w, h));
                    }
        }
        num_anchors = priors.size();

        interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));
        if (!interpreter) {
            throw std::runtime_error("Failed to load UltraFace MNN model: " + mnn_path);
        }
        MNN::ScheduleConfig config;
        config.numThread = num_thread;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_Low;
        config.backendConfig = &backendConfig;
        session = interpreter->createSession(config);
        input_tensor = interpreter->getSessionInput(session, nullptr);
    }

    ~UltraFace() {
        interpreter->releaseSession(session);
        interpreter->releaseModel();
    }

    std::vector<FaceInfo> detect(const cv::Mat &raw, double &preproc_time, double &inference_time, double &postproc_time) {
        // Preprocessing
        auto start_preproc = std::chrono::high_resolution_clock::now();
        cv::Mat img;
        cv::resize(raw, img, cv::Size(in_w, in_h));
        cv::Mat blob;
        cv::cvtColor(img, blob, cv::COLOR_BGR2RGB);
        blob.convertTo(blob, CV_32F);
        cv::Scalar mean(127, 127, 127);
        blob -= mean;
        blob *= (1.0f / 128.0f);
        std::vector<cv::Mat> channels;
        cv::split(blob, channels);
        cv::Mat nchw_blob(1, channels.size() * blob.rows * blob.cols, CV_32F);
        for (size_t c = 0; c < channels.size(); ++c) {
            std::memcpy(nchw_blob.ptr<float>() + c * blob.rows * blob.cols, channels[c].ptr<float>(), blob.rows * blob.cols * sizeof(float));
        }
        auto input_blob = nchw_blob.reshape(1, {1, 3, blob.rows, blob.cols});
        auto end_preproc = std::chrono::high_resolution_clock::now();
        preproc_time = std::chrono::duration<double, std::milli>(end_preproc - start_preproc).count();

        // Inference
        auto start_inference = std::chrono::high_resolution_clock::now();
        interpreter->resizeTensor(input_tensor, {1, 3, in_h, in_w});
        interpreter->resizeSession(session);
        std::memcpy(input_tensor->host<float>(), input_blob.ptr<float>(), input_blob.total() * sizeof(float));
        interpreter->runSession(session);
        auto ts = interpreter->getSessionOutput(session, "scores");
        auto tb = interpreter->getSessionOutput(session, "boxes");
        MNN::Tensor scores(ts, ts->getDimensionType());
        MNN::Tensor boxes(tb, tb->getDimensionType());
        ts->copyToHostTensor(&scores);
        tb->copyToHostTensor(&boxes);
        auto end_inference = std::chrono::high_resolution_clock::now();
        inference_time = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();

        // Postprocessing
        auto start_postproc = std::chrono::high_resolution_clock::now();
        std::vector<FaceInfo> boxes_out;
        for (int i = 0; i < num_anchors; i++) {
            float sc = scores.host<float>()[2 * i + 1];
            if (sc < score_threshold) continue;

            auto p = priors[i];
            float cx = p[0] + boxes.host<float>()[4 * i + 0] * 0.1f * p[2];
            float cy = p[1] + boxes.host<float>()[4 * i + 1] * 0.1f * p[3];
            float w = p[2] * std::exp(boxes.host<float>()[4 * i + 2] * 0.2f);
            float h = p[3] * std::exp(boxes.host<float>()[4 * i + 3] * 0.2f);

            float x1 = std::max(0.f, std::min(cx - w / 2, 1.f)) * raw.cols;
            float y1 = std::max(0.f, std::min(cy - h / 2, 1.f)) * raw.rows;
            float x2 = std::max(0.f, std::min(cx + w / 2, 1.f)) * raw.cols;
            float y2 = std::max(0.f, std::min(cy + h / 2, 1.f)) * raw.rows;
            boxes_out.push_back({x1, y1, x2, y2, sc});
        }

        std::sort(boxes_out.begin(), boxes_out.end(),
                  [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });
        std::vector<FaceInfo> result;
        std::vector<bool> used(boxes_out.size());
        for (size_t i = 0; i < boxes_out.size(); i++)
            if (!used[i]) {
                auto &A = boxes_out[i];
                result.push_back(A);
                used[i] = true;
                float areaA = (A.x2 - A.x1) * (A.y2 - A.y1);
                for (size_t j = i + 1; j < boxes_out.size(); j++)
                    if (!used[j]) {
                        auto &B = boxes_out[j];
                        float ix1 = std::max(A.x1, B.x1),
                              iy1 = std::max(A.y1, B.y1),
                              ix2 = std::min(A.x2, B.x2),
                              iy2 = std::min(A.y2, B.y2);
                        float iw = std::max(0.f, ix2 - ix1),
                              ih = std::max(0.f, iy2 - iy1),
                              inter = iw * ih,
                              areaB = (B.x2 - B.x1) * (B.y2 - B.y1);
                        if (inter / (areaA + areaB - inter) > iou_threshold)
                            used[j] = true;
                    }
            }
        auto end_postproc = std::chrono::high_resolution_clock::now();
        postproc_time = std::chrono::duration<double, std::milli>(end_postproc - start_postproc).count();

        return result;
    }

private:
    int in_w, in_h, num_anchors;
    float score_threshold, iou_threshold;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    std::vector<cv::Vec4f> priors;
};

// Main function with profiling and frame counter
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ulffd_model.mnn> <openseeface_model.mnn>\n";
        return 1;
    }

    std::string ulffdModelPath = argv[1];
    std::string lmModelPath = argv[2];

    // Initialize video capture
    cv::VideoCapture cap(0, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open webcam" << std::endl;
        return -1;
    }


    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Set resolution: " << width << "x" << height << std::endl;

    // Initialize UltraFace detector
    UltraFace ulffdDetector(ulffdModelPath, 320, 240, 4, 0.7f, 0.3f);

    // Initialize OpenSeeFace landmark model
    std::shared_ptr<MNN::Interpreter> lm_interpreter(MNN::Interpreter::createFromFile(lmModelPath.c_str()));
    if (!lm_interpreter) {
        std::cerr << "Failed to load OpenSeeFace MNN model." << std::endl;
        return -1;
    }
    MNN::ScheduleConfig config;
    config.numThread = 4;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    MNN::Session* lm_session = lm_interpreter->createSession(config);
    MNN::Tensor* lm_input_tensor = lm_interpreter->getSessionInput(lm_session, nullptr);

    // Normalization constants for OpenSeeFace
    cv::Mat mean_res = cv::Mat(224, 224, CV_32FC3, cv::Scalar(-2.1179f, -2.0357f, -1.8044f));
    cv::Mat std_res = cv::Mat(224, 224, CV_32FC3, cv::Scalar(0.01713f, 0.01751f, 0.01743f));

    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    std::deque<double> fps_history;
    const size_t FPS_WINDOW = 10;
    double avg_fps = 0.0;
    size_t frame_count = 0;

    while (true) {
        ++frame_count;

        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Cannot read a frame from webcam" << std::endl;
            break;
        }

        // Face detection using UltraFace with separate timing
        double uf_preproc_time = 0.0, uf_inference_time = 0.0, uf_postproc_time = 0.0;
        auto start_fd = std::chrono::high_resolution_clock::now();
        auto faces = ulffdDetector.detect(frame, uf_preproc_time, uf_inference_time, uf_postproc_time);
        auto end_fd = std::chrono::high_resolution_clock::now();
        double fd_time = std::chrono::duration<double, std::milli>(end_fd - start_fd).count();

        cv::Vec<double, 5> crop_info;
        double pp_time = 0.0, lm_inference_time = 0.0, post_time = 0.0;

        if (!faces.empty()) {
            // Use the first detected face
            auto face = faces[0];
            cv::Rect cvFaceRect(int(face.x1), int(face.y1), int(face.x2 - face.x1), int(face.y2 - face.y1));
            cv::rectangle(frame, cvFaceRect, cv::Scalar(0, 255, 0), 2);

            // Compute crop coordinates with margins
            int x = int(face.x1);
            int y = int(face.y1);
            int w = int(face.x2 - face.x1);
            int h = int(face.y2 - face.y1);
            int crop_x1 = x - static_cast<int>(w * 0.1);
            int crop_y1 = y - static_cast<int>(h * 0.125);
            int crop_x2 = x + w + static_cast<int>(w * 0.1);
            int crop_y2 = y + h + static_cast<int>(h * 0.125);

            clamp_to_im(crop_x1, crop_y1, frame.cols, frame.rows);
            clamp_to_im(crop_x2, crop_y2, frame.cols, frame.rows);

            double scale_x = double(crop_x2 - crop_x1) / 224;
            double scale_y = double(crop_y2 - crop_y1) / 224;
            double bonus_value = 0.1;
            crop_info = cv::Vec<double, 5>(crop_x1, crop_y1, scale_x, scale_y, bonus_value);

            if ((crop_x2 - crop_x1) >= 4 && (crop_y2 - crop_y1) >= 4) {
                // Preprocess the crop
                auto start_pp = std::chrono::high_resolution_clock::now();
                cv::Rect cropRect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
                cv::Mat blob = preprocess(frame, cropRect, 224, std_res, mean_res);
                auto end_pp = std::chrono::high_resolution_clock::now();
                pp_time = std::chrono::duration<double, std::milli>(end_pp - start_pp).count();

                // Run landmark inference with OpenSeeFace
                auto start_lm = std::chrono::high_resolution_clock::now();
                lm_interpreter->resizeTensor(lm_input_tensor, {1, 3, 224, 224});
                lm_interpreter->resizeSession(lm_session);
                std::memcpy(lm_input_tensor->host<float>(), blob.ptr<float>(), blob.total() * sizeof(float));
                lm_interpreter->runSession(lm_session);
                auto output_tensor = lm_interpreter->getSessionOutput(lm_session, nullptr);
                MNN::Tensor host_tensor(output_tensor, MNN::Tensor::CAFFE);
                output_tensor->copyToHostTensor(&host_tensor);
                auto end_lm = std::chrono::high_resolution_clock::now();
                lm_inference_time = std::chrono::duration<double, std::milli>(end_lm - start_lm).count();

                // Postprocess landmarks
                auto start_post = std::chrono::high_resolution_clock::now();
                cv::Mat output(1, 198 * 28 * 28, CV_32F, host_tensor.host<float>());
                auto lm_result = landmarks(output, crop_info); // Fixed deepSearch error
                auto end_post = std::chrono::high_resolution_clock::now();
                post_time = std::chrono::duration<double, std::milli>(end_post - start_post).count();

                float avg_conf = lm_result.first;
                cv::Mat lms = lm_result.second;
                std::cout << "Average confidence: " << avg_conf << std::endl;

                // Draw landmarks
                for (int i = 0; i < lms.rows; i++) {
                    float lx = lms.at<float>(i, 0);
                    float ly = lms.at<float>(i, 1);
                    cv::circle(frame, cv::Point(int(ly), int(lx)), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        // Calculate total frame time as sum of model processing times
        double total_time = uf_preproc_time + uf_inference_time + uf_postproc_time + pp_time + lm_inference_time + post_time;
        double current_fps = total_time > 0 ? 1000.0 / total_time : 0.0;

        fps_history.push_back(current_fps);
        if (fps_history.size() > FPS_WINDOW) {
            fps_history.pop_front();
        }
        double sum_fps = 0.0;
        for (double fps : fps_history) {
            sum_fps += fps;
        }
        avg_fps = sum_fps / fps_history.size();

        // Display timing information
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Frame " << frame_count << " Timing (ms): "
                  << "Total: " << total_time
                  << ", UF Preproc: " << uf_preproc_time
                  << ", UF Inference: " << uf_inference_time
                  << ", UF Postproc: " << uf_postproc_time
                  << ", OSF Preproc: " << pp_time
                  << ", OSF Inference: " << lm_inference_time
                  << ", OSF Postproc: " << post_time
                  << ", FPS: " << current_fps
                  << ", Avg FPS: " << avg_fps << std::endl;

        // Display FPS on frame
        cv::putText(frame, cv::format("FPS: %.2f", avg_fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    lm_interpreter->releaseSession(lm_session);
    lm_interpreter->releaseModel();
    return 0;
}
