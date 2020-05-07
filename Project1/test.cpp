#include"ClassPredict.h"

int main() {


    using torch::jit::script::Module;
    Module module = torch::jit::load("model.pt");
    
    assert(module != nullptr);
    std::cout << "model load ok\n";

    char pattern[] = "E:\\test\\OK\\*.jpg";
    std::string pattern_bmp = pattern;
    std::cout << pattern_bmp << std::endl;
    std::vector<cv::String> image_files;
    cv::glob(pattern_bmp, image_files);
    if (image_files.size() == 0) {
        std::cout << "No image files[jpg]" << std::endl;
        return 0;
    }
    for (int frame = 0; frame < image_files.size(); frame++)
    {

        cv::Mat image, input;
        image = cv::imread(image_files[frame]);
        cv::resize(image, image, cv::Size(96, 96));
        cv::cvtColor(image, input, cv::COLOR_BGR2RGB);
        predict(module, input.data);
    }

    return 1;
}