#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>

using namespace std;
namespace fs = std::filesystem;

// Function to simulate the preprocessing pipeline on a single image
void process_image(const string& input_path, const string& output_dir) {

    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) return;


    cv::Mat resized, normalized;
    cv::resize(img, resized, cv::Size(1024, 1024));
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);


    int patch_size = 256;
    int patch_id = 0;
    string base_name = fs::path(input_path).stem().string();

    for (int y = 0; y <= resized.rows - patch_size; y += patch_size) {
        for (int x = 0; x <= resized.cols - patch_size; x += patch_size) {
            cv::Rect roi(x, y, patch_size, patch_size);
            cv::Mat patch = normalized(roi);
            
            // Convert back to 8-bit for saving
            cv::Mat save_patch;
            patch.convertTo(save_patch, CV_8U, 255.0);
            
            string out_path = output_dir + "/" + base_name + "_p" + to_string(patch_id++) + ".png";
            cv::imwrite(out_path, save_patch);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./omp_preprocessing <input_dir> <output_dir> <num_threads>" << endl;
        return -1;
    }

    string input_dir = argv[1];
    string output_dir = argv[2];
    int num_threads = stoi(argv[3]);

    omp_set_num_threads(num_threads);

    // Gather all PNG files
    vector<string> image_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }

    cout << "Starting OpenMP Processing with " << num_threads << " threads on " 
              << image_files.size() << " images." << endl;

    auto start_time = chrono::high_resolution_clock::now();

    // Distributes the image list across CPU cores
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < image_files.size(); ++i) {
        process_image(image_files[i], output_dir);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout<< "OpenMP Execution Time: " << elapsed.count() << " seconds" << endl;

    return 0;
}