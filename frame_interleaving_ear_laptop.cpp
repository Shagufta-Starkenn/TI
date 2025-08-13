#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <numeric> // For std::accumulate

using namespace cv;
using namespace std;

// --- OpenSeeFace Landmark Code (unchanged) ---

const int RES = 224;
const int OUT_RES = 27;
const int OUT_RES_I = 28;
const float LOGIT_FACTOR = 16.0f;
const int c0 = 66, c1 = 132, c2 = 198;

void clamp_to_im(int &x, int &y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
}

float logit(float p, float factor) {
    float clipped = min(max(p, 1e-7f), 0.9999999f);
    return log(clipped / (1.0f - clipped)) / factor;
}

pair<float, Mat> process_landmarks(const Mat &tensor, const Vec<double, 5> &crop_info) {
    int res_minus1 = RES - 1;
    int grid = OUT_RES_I * OUT_RES_I;
    Mat tensor_mat = tensor.reshape(1, c2);
    Mat t_main = tensor_mat.rowRange(0, c0);
    Mat t_off_x = tensor_mat.rowRange(c0, c1);
    Mat t_off_y = tensor_mat.rowRange(c1, c2);
    vector<int> t_m(c0, 0);
    vector<float> t_conf(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        Mat row = t_main.row(i);
        double maxVal;
        Point maxLoc;
        minMaxLoc(row, nullptr, &maxVal, nullptr, &maxLoc);
        t_m[i] = maxLoc.x;
        t_conf[i] = static_cast<float>(maxVal);
    }
    vector<float> off_x(c0, 0.0f), off_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        off_x[i] = t_off_x.at<float>(i, t_m[i]);
        off_y[i] = t_off_y.at<float>(i, t_m[i]);
    }
    for (int i = 0; i < c0; i++) {
        off_x[i] = res_minus1 * logit(off_x[i], LOGIT_FACTOR);
        off_y[i] = res_minus1 * logit(off_y[i], LOGIT_FACTOR);
    }
    vector<float> t_x(c0, 0.0f), t_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        double crop_x1 = crop_info[0];
        double crop_y1 = crop_info[1];
        double scale_x = crop_info[2];
        double scale_y = crop_info[3];
        float row_idx = floor(float(t_m[i]) / float(OUT_RES_I));
        float col_idx = float(t_m[i]) - row_idx * OUT_RES_I;
        t_x[i] = static_cast<float>(crop_y1) + static_cast<float>(scale_y) * (res_minus1 * row_idx / float(OUT_RES) + off_x[i]);
        t_y[i] = static_cast<float>(crop_x1) + static_cast<float>(scale_x) * (res_minus1 * col_idx / float(OUT_RES) + off_y[i]);
    }
    float sum_conf = 0.0f;
    for (int i = 0; i < c0; i++) {
        sum_conf += t_conf[i];
    }
    float avg_conf = sum_conf / c0;
    Mat lms(c0, 3, CV_32F);
    for (int i = 0; i < c0; i++) {
        lms.at<float>(i, 0) = t_x[i];
        lms.at<float>(i, 1) = t_y[i];
        lms.at<float>(i, 2) = t_conf[i];
    }
    for (int i = 0; i < c0; i++) {
        if (isnan(lms.at<float>(i, 0)) || isnan(lms.at<float>(i, 1)) || isnan(lms.at<float>(i, 2))) {
            lms.at<float>(i, 0) = 0.0f;
            lms.at<float>(i, 1) = 0.0f;
            lms.at<float>(i, 2) = 0.0f;
        }
    }
    return make_pair(avg_conf, lms);
}

// --- UltraFace Face Detection Code (unchanged) ---

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

struct FaceInfo {
    float x1, y1, x2, y2, score;
};

class UltraFace {
public:
    UltraFace(const std::string &mnn_path,
              int input_w = 320, int input_h = 240,
              int num_thread = 4,
              float score_thresh = 0.7f,
              float iou_thresh = 0.3f)
        : in_w(input_w), in_h(input_h),
          score_threshold(score_thresh),
          iou_threshold(iou_thresh) {
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
        interpreter->runSession(session);
        auto ts = interpreter->getSessionOutput(session, "scores");
        auto tb = interpreter->getSessionOutput(session, "boxes");
        MNN::Tensor scores(ts, ts->getDimensionType());
        MNN::Tensor boxes(tb, tb->getDimensionType());
        ts->copyToHostTensor(&scores);
        tb->copyToHostTensor(&boxes);
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

// --- Landmark model using MNN (unchanged) ---

class LandmarkModel {
public:
    LandmarkModel(const string& model_path) {
        interpreter = shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
        if (!interpreter) {
            throw runtime_error("Failed to load landmark model");
        }
        MNN::ScheduleConfig config;
        config.numThread = 4;
        config.type = MNN_FORWARD_CPU;
        session = interpreter->createSession(config);
        input_tensor = interpreter->getSessionInput(session, nullptr);
        const float original_mean[3] = {0.485f, 0.456f, 0.406f};
        const float original_std[3] = {0.229f, 0.224f, 0.225f};
        for (int i = 0; i < 3; ++i) {
            lm_mean_vals[i] = original_mean[i] * 255.0f;
            lm_norm_vals[i] = 1.0f / (original_std[i] * 255.0f);
        }
        cout << "Landmark model initialized" << endl;
    }

    Mat predict(const Mat& im, const Rect& cropRect) {
        Mat cropped = im(cropRect).clone();
        Mat resized;
        resize(cropped, resized, Size(RES, RES), 0, 0, INTER_LINEAR);
        auto proc = std::shared_ptr<MNN::CV::ImageProcess>(
            MNN::CV::ImageProcess::create(
                MNN::CV::BGR, MNN::CV::RGB,
                lm_mean_vals, 3,
                lm_norm_vals, 3
            )
        );
        interpreter->resizeTensor(input_tensor, {1, 3, RES, RES});
        interpreter->resizeSession(session);
        proc->convert(resized.data, RES, RES, resized.step[0], input_tensor);
        interpreter->runSession(session);
        auto output = interpreter->getSessionOutput(session, nullptr);
        MNN::Tensor output_tensor(output, output->getDimensionType());
        output->copyToHostTensor(&output_tensor);
        return Mat(1, output_tensor.elementSize(), CV_32F, output_tensor.host<float>());
    }

private:
    shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    MNN::Tensor* input_tensor = nullptr;
    float lm_mean_vals[3];
    float lm_norm_vals[3];
};

// --- New functions for Eye Aspect Ratio (EAR) ---

/**
 * @brief Calculates the Euclidean distance between two 2D points.
 * @param p1 First point.
 * @param p2 Second point.
 * @return The Euclidean distance.
 */
double euclideanDistance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/**
 * @brief Calculates the Eye Aspect Ratio (EAR) for a single eye.
 * The EAR is a measure of how open the eye is.
 * Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
 * where p1, p2, ..., p6 are specific landmark points of the eye.
 * @param eyePoints A vector of 6 Point objects representing the eye landmarks.
 * Expected order: p1, p2, p3, p4, p5, p6
 * @return The calculated EAR value. Returns 0.0 if points are invalid or divisor is zero.
 */
double eyeAspectRatio(const vector<Point>& eyePoints) {
    if (eyePoints.size() != 6) {
        cerr << "Error: eyePoints must contain exactly 6 points." << endl;
        return 0.0;
    }

    // Extract points for clarity
    Point p1 = eyePoints[0]; // outer corner
    Point p2 = eyePoints[1]; // top eyelid, inner
    Point p3 = eyePoints[2]; // top eyelid, outer
    Point p4 = eyePoints[3]; // inner corner
    Point p5 = eyePoints[4]; // bottom eyelid, outer
    Point p6 = eyePoints[5]; // bottom eyelid, inner

    // Calculate vertical distances
    double d_vertical1 = euclideanDistance(p2, p6);
    double d_vertical2 = euclideanDistance(p3, p5);

    // Calculate horizontal distance
    double d_horizontal = euclideanDistance(p1, p4);

    if (d_horizontal == 0.0) { // Avoid division by zero
        return 0.0;
    }

    return (d_vertical1 + d_vertical2) / (2.0 * d_horizontal);
}

// --- Main Function (modified for every 5th frame landmark processing) ---

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <ultraface_model.mnn> <osf_landmarks_model.mnn>\n";
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open webcam\n";
        return -1;
    }

    UltraFace detector(argv[1]);
    LandmarkModel landmark_model(argv[2]);

    auto last_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    double detection_time_ms = 0.0;
    double landmark_time_ms = 0.0;
    int frame_count = 0; // Initialize frame counter
    
    // Threshold for Eye Aspect Ratio (EAR) - this value may need tuning
    const double EAR_THRESHOLD = 0.15; // Common threshold, adjust as needed

    // For smoothing EAR values
    const int EAR_HISTORY_SIZE = 5; // Number of previous EAR values to average
    std::vector<double> ear_history(EAR_HISTORY_SIZE, 0.0);
    int ear_history_idx = 0;

    // Persist landmark points and eye status across skipped frames
    vector<Point> last_landmark_points;
    string eye_status = "Unknown";
    double smoothed_ear = 0.0;
    double last_left_ear = 0.0; // Store last calculated left EAR
    double last_right_ear = 0.0; // Store last calculated right EAR

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        auto start_time_overall = std::chrono::high_resolution_clock::now();
        flip(frame, frame, 1);

        // Face detection (every frame)
        auto start_time_detect = std::chrono::high_resolution_clock::now();
        vector<FaceInfo> faces = detector.detect(frame);
        auto end_time_detect = std::chrono::high_resolution_clock::now();
        detection_time_ms = std::chrono::duration<double, std::milli>(end_time_detect - start_time_detect).count();

        landmark_time_ms = 0.0; // Reset landmark time for each frame
        
        if (!faces.empty()) {
            auto &f = faces[0]; // Process first face
            
            // Draw face bounding box
            cv::rectangle(frame, cv::Point(int(f.x1), int(f.y1)), cv::Point(int(f.x2), int(f.y2)), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("Score: %.2f", f.score), cv::Point(int(f.x1), int(f.y1)-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // Landmark detection (only on every 5th frame)
            if (frame_count % 5 == 0) { // Check for every 5th frame
                auto start_time_landmark = std::chrono::high_resolution_clock::now();

                int face_width = f.x2 - f.x1;
                int face_height = f.y2 - f.y1;
                int center_x = f.x1 + face_width / 2;
                int center_y = f.y1 + face_height / 2;
                int side = static_cast<int>(std::max(face_width, face_height) * 1.5);
                int x1 = center_x - side / 2;
                int y1 = center_y - side / 2;
                int x2 = x1 + side;
                int y2 = y1 + side;
                clamp_to_im(x1, y1, frame.cols, frame.rows);
                clamp_to_im(x2, y2, frame.cols, frame.rows);
                cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);

                if (crop_rect.width > 0 && crop_rect.height > 0) {
                    double scale_x = double(crop_rect.width) / RES;
                    double scale_y = double(crop_rect.height) / RES;
                    double bonus_value = 0.1;
                    Vec<double, 5> crop_info(static_cast<double>(crop_rect.x), static_cast<double>(crop_rect.y), scale_x, scale_y, bonus_value);
                    Mat output = landmark_model.predict(frame, crop_rect);
                    auto lm_result = process_landmarks(output, crop_info);

                    auto end_time_landmark = std::chrono::high_resolution_clock::now();
                    landmark_time_ms = std::chrono::duration<double, std::milli>(end_time_landmark - start_time_landmark).count();

                    float avg_conf = lm_result.first;
                    Mat lms = lm_result.second;
                    
                    last_landmark_points.clear(); // Clear old points
                    // Collect new landmark points
                    for (int i = 0; i < lms.rows; i++) {
                        float ly_coord = lms.at<float>(i, 0); // Y-coordinate in image space
                        float lx_coord = lms.at<float>(i, 1); // X-coordinate in image space
                        if (!isnan(lx_coord) && !isnan(ly_coord)) {
                            Point pt(static_cast<int>(lx_coord), static_cast<int>(ly_coord));
                            last_landmark_points.push_back(pt);
                        }
                    }

                    // --- EAR Calculation Logic (only when new landmarks are detected) ---
                    // Assuming OpenSeeFace 66 points map to dlib-like 68 points eye indices:
                    // Left eye: P37-P42 (dlib indices 36-41)
                    // Right eye: P43-P48 (dlib indices 42-47)
                    // If OpenSeeFace landmarks are 0-indexed and correspond to dlib's, these indices should be correct.
                    if (last_landmark_points.size() >= 48) { 
                        // Left eye landmarks
                        vector<Point> left_eye_points = {
                            last_landmark_points[36], last_landmark_points[37], last_landmark_points[38],
                            last_landmark_points[39], last_landmark_points[40], last_landmark_points[41]
                        };

                        // Right eye landmarks
                        vector<Point> right_eye_points = {
                            last_landmark_points[42], last_landmark_points[43], last_landmark_points[44],
                            last_landmark_points[45], last_landmark_points[46], last_landmark_points[47]
                        };

                        last_left_ear = eyeAspectRatio(left_eye_points);
                        last_right_ear = eyeAspectRatio(right_eye_points);
                        double current_raw_ear = (last_left_ear + last_right_ear) / 2.0; // Average EAR for both eyes

                        // Add current raw EAR to history and calculate smoothed EAR
                        ear_history[ear_history_idx] = current_raw_ear;
                        ear_history_idx = (ear_history_idx + 1) % EAR_HISTORY_SIZE;
                        smoothed_ear = std::accumulate(ear_history.begin(), ear_history.end(), 0.0) / EAR_HISTORY_SIZE;
                    }
                }
            } // End of frame_count % 5 == 0 block

            // Draw all landmarks (from last detected set)
            for (const auto& pt : last_landmark_points) {
                circle(frame, pt, 2, Scalar(0, 0, 255), -1); // Draw all landmarks
            }
            
            // Determine eye status based on smoothed EAR (using the last calculated smoothed_ear)
            if (smoothed_ear < EAR_THRESHOLD) {
                eye_status = "Closed 😴";
            } else {
                eye_status = "Open 😊";
            }

            // --- Removed drawing of colored rectangles around eyes ---
            // The user requested to remove the eye bounding boxes.
        }

        auto end_time_overall = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds_overall = end_time_overall - start_time_overall;
        fps = 1.0 / elapsed_seconds_overall.count();

        cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, cv::format("Detect: %.2f ms", detection_time_ms), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, cv::format("Landmark: %.2f ms (last)", landmark_time_ms), cv::Point(10, 90), // Clarified "last"
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, cv::format("Left EAR: %.2f", last_left_ear), cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, cv::format("Right EAR: %.2f", last_right_ear), cv::Point(10, 150),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, cv::format("EAR: %.2f", smoothed_ear), cv::Point(10, 180),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, cv::format("Eyes: %s", eye_status.c_str()), cv::Point(10, 210),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);


        cv::imshow("Webcam", frame);
        char k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;

        frame_count++; // Increment frame counter
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

