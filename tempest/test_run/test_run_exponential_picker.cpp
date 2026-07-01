#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include <proxies/RandomPicker.cuh>

constexpr int TOTAL_TIMESTEPS = 100000000;

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

int main(const int argc, char* argv[])
{
    bool use_gpu = USE_GPU;

    if (argc > 1) {
        std::string gpu_arg = argv[1];
        std::transform(gpu_arg.begin(), gpu_arg.end(), gpu_arg.begin(),
                       [](const unsigned char c){ return std::tolower(c); });

        if (gpu_arg == "1" || gpu_arg == "true" || gpu_arg == "yes" || gpu_arg == "y") {
            use_gpu = true;
        } else if (gpu_arg == "0" || gpu_arg == "false" || gpu_arg == "no" || gpu_arg == "n") {
            use_gpu = false;
        } else {
            std::cerr << "Error: Invalid value '" << gpu_arg << "' for use_gpu parameter. Expected 1/0, true/false, yes/no, or y/n." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    const ExponentialIndexRandomPicker random_picker(use_gpu);

    std::cout << "Prioritizing end: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, true) << std::endl;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Prioritizing start: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, false) << std::endl;
    }

    return 0;
}
