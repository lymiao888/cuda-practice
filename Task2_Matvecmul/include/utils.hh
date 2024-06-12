#include <chrono>
#include <iostream>
#include <random>
#include <string>

#ifndef UTILS_DEF_11
#define UTILS_DEF_11

class Timer
{
public:
    Timer(const std::string &name)
        : timer_name_(name), start_(std::chrono::steady_clock::now()) {}

    ~Timer()
    {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start_);
        std::cout << ">>> " << timer_name_ << ": " << duration.count() << "ms"
                  << std::endl;
    }

private:
    std::string timer_name_{};
    std::chrono::time_point<std::chrono::steady_clock> start_{};
};

void RandomizeSP(const int &len, float *arr, const float &l = -2.f,
                 const float &r = 2.f)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> float_rng(l, r);
    for (int i = 0; i < len; i++)
    {
        arr[i] = float_rng(rng);
    }
}

void RandomizeBoolean(const int &len, bool *arr)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> int_rng(0, 1);
    for (int i = 0; i < len; i++)
    {
        arr[i] = int_rng(rng);
    }
}

template <class T>
void PrintArray(const int &len, T *arr)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

#undef PrintArray
#endif