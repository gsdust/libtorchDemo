#include "ClassPredict.h"



int predict(torch::jit::script::Module module, uchar* inputImagepuBuffer)
{
    //using torch::jit::script::Module;
    torch::Tensor tensor_image = torch::from_blob(inputImagepuBuffer, { 96,96,3 }, torch::kByte);

    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    torch::Tensor mean = torch::tensor({ 0.2437, 0.2437, 0.2437 });//¾ùÖµ
    torch::Tensor dstd = torch::tensor({ 0.3154, 0.3154, 0.3154 });
    tensor_image = (tensor_image - mean) / dstd;

    //std::cout << tensor_image << std::endl;
    tensor_image = tensor_image.permute({ 2,0,1 });
    tensor_image = tensor_image.unsqueeze(0);
    //tensor_image = tensor_image.to(torch::kCPU);
    //inputs.push_back(tensor_image);
    auto output = module.forward({ tensor_image }).toTensor();
    auto max_result = output.max(1, true);
    auto max_idx = std::get<1>(max_result).item<int>();

    //auto max_index = std::get<1>(max_result).item<float>();
    std::cout << max_idx << '\n';
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
