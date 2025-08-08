#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // Added for timing
#include <iomanip> // Added for output formatting
#include <cmath>
#include <vector>
#include <memory>

using namespace cv;
using namespace std;

// --- OpenSeeFace Landmark Code (from new provided code) ---
// Constants for the landmark model
const int RES = 224;
const int OUT_RES = 27;
const int OUT_RES_I = 28;
const float LOGIT_FACTOR = 16.0f;
const int c0 = 66, c1 = 132, c2 = 198;

/**
 * @brief Clamps a coordinate (x, y) to be within the image boundaries [0, width-1] and [0, height-1].
 */
void clamp_to_im(int &x, int &y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
}

/**
 * @brief Applies the logit function, often used to map a probability-like value to a continuous range.
 */
float logit(float p, float factor) {
    float clipped = min(max(p, 1e-7f), 0.9999999f);
    return log(clipped / (1.0f - clipped)) / factor;
}

/**
 * @brief Computes landmark coordinates from the MNN model output tensor and crop information.
 * The output coordinates are already scaled to the original image frame.
 */
pair<float, Mat> process_landmarks(const Mat &tensor, const Vec<double, 5> &crop_info) {
    int res_minus1 = RES - 1;
    int grid = OUT_RES_I * OUT_RES_I;

    // Reshape the flat tensor to a matrix of shape (198, grid)
    Mat tensor_mat = tensor.reshape(1, c2);  // (198, grid)

    Mat t_main = tensor_mat.rowRange(0, c0);   // (66, grid)
    Mat t_off_x = tensor_mat.rowRange(c0, c1); // (66, grid)
    Mat t_off_y = tensor_mat.rowRange(c1, c2); // (66, grid)

    // For each row in t_main, find the argmax index and record confidence.
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

    // Extract offset values at the argmax indices.
    vector<float> off_x(c0, 0.0f), off_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        off_x[i] = t_off_x.at<float>(i, t_m[i]);
        off_y[i] = t_off_y.at<float>(i, t_m[i]);
    }

    // Apply logit transformation and scale by res_minus1.
    for (int i = 0; i < c0; i++) {
        off_x[i] = res_minus1 * logit(off_x[i], LOGIT_FACTOR);
        off_y[i] = res_minus1 * logit(off_y[i], LOGIT_FACTOR);
    }

    // Compute landmark coordinates.
    vector<float> t_x(c0, 0.0f), t_y(c0, 0.0f);
    for (int i = 0; i < c0; i++) {
        double crop_x1 = crop_info[0]; // Original x-coordinate of the crop's top-left
        double crop_y1 = crop_info[1]; // Original y-coordinate of the crop's top-left
        double scale_x = crop_info[2]; // Scale factor for x
        double scale_y = crop_info[3]; // Scale factor for y

        float row_idx = floor(float(t_m[i]) / float(OUT_RES_I));
        float col_idx = float(t_m[i]) - row_idx * OUT_RES_I;

        // Note: t_x is calculated using crop_y1 and scale_y, making it the Y-coordinate in the original frame
        t_x[i] = static_cast<float>(crop_y1) + static_cast<float>(scale_y) * (res_minus1 * row_idx / float(OUT_RES) + off_x[i]);
        // Note: t_y is calculated using crop_x1 and scale_x, making it the X-coordinate in the original frame
        t_y[i] = static_cast<float>(crop_x1) + static_cast<float>(scale_x) * (res_minus1 * col_idx / float(OUT_RES) + off_y[i]);
    }

    // Compute average confidence.
    float sum_conf = 0.0f;
    for (int i = 0; i < c0; i++) {
        sum_conf += t_conf[i];
    }
    float avg_conf = sum_conf / c0;

    // Stack t_x (Y), t_y (X), and t_conf into a (66 x 3) matrix.
    Mat lms(c0, 3, CV_32F);
    for (int i = 0; i < c0; i++) {
        lms.at<float>(i, 0) = t_x[i]; // Y-coordinate
        lms.at<float>(i, 1) = t_y[i]; // X-coordinate
        lms.at<float>(i, 2) = t_conf[i];
    }
     
    // Replace any NaN rows with zeros.
    for (int i = 0; i < c0; i++) {
        if (isnan(lms.at<float>(i, 0)) || isnan(lms.at<float>(i, 1)) || isnan(lms.at<float>(i, 2))) {
            lms.at<float>(i, 0) = 0.0f;
            lms.at<float>(i, 1) = 0.0f;
            lms.at<float>(i, 2) = 0.0f;
        }
    }
    return make_pair(avg_conf, lms);
}

// --- UltraFace Face Detection Code (unchanged from previous Canvas) ---
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
          iou_threshold(iou_thresh)
    {
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

// --- Landmark model using MNN (from new provided code) ---
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
         
        // Calculate MNN ImageProcess mean and norm values for ImageNet normalization
        // Original normalization in preprocess_landmarks was: (pixel / 255.0f - mean) / std
        // MNN ImageProcess normalization is: (pixel - MNN_mean) * MNN_norm
        // To equate them:
        // MNN_mean = original_mean * 255.0f
        // MNN_norm = 1.0f / (original_std * 255.0f)
        const float original_mean[3] = {0.485f, 0.456f, 0.406f}; // ImageNet means for R, G, B
        const float original_std[3] = {0.229f, 0.224f, 0.225f};  // ImageNet stds for R, G, B

        for (int i = 0; i < 3; ++i) {
            lm_mean_vals[i] = original_mean[i] * 255.0f;
            lm_norm_vals[i] = 1.0f / (original_std[i] * 255.0f);
        }

        cout << "Landmark model initialized" << endl;
    }

    // Modified predict function to handle image preprocessing internally using MNN::CV::ImageProcess
    // It now takes the raw image frame and the crop rectangle.
    Mat predict(const Mat& im, const Rect& cropRect) {
        // Crop the image using OpenCV.
        Mat cropped = im(cropRect).clone();
         
        // Resize the cropped image to the model's input resolution (RES x RES).
        Mat resized;
        resize(cropped, resized, Size(RES, RES), 0, 0, INTER_LINEAR);
         
        // Prepare MNN::CV::ImageProcess for normalization and channel order conversion.
        // The input image 'im' (and thus 'resized') is in BGR format from OpenCV.
        // The landmark model expects RGB input with ImageNet normalization.
        auto proc = std::shared_ptr<MNN::CV::ImageProcess>(
            MNN::CV::ImageProcess::create(
                MNN::CV::BGR, MNN::CV::RGB, // Source is BGR, Destination is RGB
                lm_mean_vals, 3,             // Calculated mean values for MNN ImageProcess
                lm_norm_vals, 3              // Calculated norm values for MNN ImageProcess
            )
        );

        // Resize MNN input tensor to match the expected dimensions (1, C, H, W).
        interpreter->resizeTensor(input_tensor, {1, 3, RES, RES});
        // Resize the session to apply the tensor dimensions.
        interpreter->resizeSession(session);

        // Convert the resized OpenCV Mat data and populate the MNN input tensor.
        // This handles color conversion (BGR to RGB), resizing, normalization,
        // and data layout (HWC to NCHW) in one step.
        proc->convert(resized.data, RES, RES, resized.step[0], input_tensor);

        // Run inference on the MNN session.
        interpreter->runSession(session);
         
        // Get the output tensor from the session.
        auto output = interpreter->getSessionOutput(session, nullptr);
        // Create a host tensor to copy the output data to CPU memory.
        MNN::Tensor output_tensor(output, output->getDimensionType());
        output->copyToHostTensor(&output_tensor);
         
        // Return the flattened output as an OpenCV Mat.
        // The output is a 1D float array, so we reshape it to a 1xN Mat.
        return Mat(1, output_tensor.elementSize(), CV_32F, output_tensor.host<float>());
    }

private:
    shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    MNN::Tensor* input_tensor = nullptr;
    float lm_mean_vals[3]; // Mean values specifically for MNN ImageProcess for landmark model
    float lm_norm_vals[3]; // Norm values specifically for MNN ImageProcess for landmark model
};


// --- Main Function ---
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

    // Initialize Face Detector
    UltraFace detector(argv[1]);

    // Initialize Landmark Model
    LandmarkModel landmark_model(argv[2]);
    
    // Variables for FPS calculation
    auto last_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    
    // Variables for individual model timing
    double detection_time_ms = 0.0;
    double landmark_time_ms = 0.0;

    //cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE); // Uncomment if you want to explicitly name the window
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        // Start timer for overall frame processing
        auto start_time_overall = std::chrono::high_resolution_clock::now();

        // Flip frame horizontally for more intuitive viewing
        flip(frame, frame, 1);

        // --- Face detection timing ---
        auto start_time_detect = std::chrono::high_resolution_clock::now();
        vector<FaceInfo> faces = detector.detect(frame);
        auto end_time_detect = std::chrono::high_resolution_clock::now();
        detection_time_ms = std::chrono::duration<double, std::milli>(end_time_detect - start_time_detect).count();

        landmark_time_ms = 0.0; // Reset landmark time for each frame
        if (!faces.empty()) { // Only measure landmark time if a face is detected
            // For simplicity, we'll measure the landmark time for the first detected face.
            // If you need average time for multiple faces, you'd sum and divide.
            auto &f = faces[0]; // Use the first detected face for landmark timing example

            // --- Landmark model timing ---
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
            if (crop_rect.width > 0 && crop_rect.height > 0) { // Ensure valid crop_rect
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

                // Draw landmarks for the first face
                vector<Point> landmark_points;
                for (int i = 0; i < lms.rows; i++) {
                    float ly_coord = lms.at<float>(i, 0);
                    float lx_coord = lms.at<float>(i, 1);
                     
                    if (!isnan(lx_coord) && !isnan(ly_coord)) {
                        Point pt(static_cast<int>(lx_coord), static_cast<int>(ly_coord));
                        circle(frame, pt, 2, Scalar(0, 0, 255), -1);
                        landmark_points.push_back(pt);
                    }
                }

                if (!landmark_points.empty()) {
                    int min_x = frame.cols, max_x = 0;
                    int min_y = frame.rows, max_y = 0;
                    for (const auto& pt : landmark_points) {
                        min_x = min(min_x, pt.x);
                        max_x = max(max_x, pt.x);
                        min_y = min(min_y, pt.y);
                        max_y = max(max_y, pt.y);
                    }
                    int padding = max((max_x - min_x) / 10, (max_y - min_y) / 10);
                    min_x = max(0, min_x - padding);
                    max_x = min(frame.cols - 1, max_x + padding);
                    min_y = max(0, min_y - padding);
                    max_y = min(frame.rows - 1, max_y + padding);
                }
            }
            
            // Draw bounding box for the first face
            cv::rectangle(frame, cv::Point(int(f.x1), int(f.y1)), cv::Point(int(f.x2), int(f.y2)), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("Score: %.2f",f.score), cv::Point(int(f.x1),int(f.y1)-5),
                        cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar(0,255,0),1);
        }

        // Stop timer for overall frame processing
        auto end_time_overall = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds_overall = end_time_overall - start_time_overall;
        
        // Calculate instantaneous FPS
        fps = 1.0 / elapsed_seconds_overall.count();

        // Display FPS on the frame
        cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        
        // Display individual model times
        cv::putText(frame, cv::format("Detect: %.2f ms", detection_time_ms), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, cv::format("Landmark: %.2f ms", landmark_time_ms), cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

        cv::imshow("Webcam", frame);
        char k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
