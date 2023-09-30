#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <chrono>

void threadFunc(const std::vector<double> &v1, const std::vector<double> &v2, std::vector<double> &result, size_t start,
                size_t offset) {
    while (start < v1.size()) {
        result[start] = std::min(v1[start], v2[start]);
        start += offset;
    }
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::setprecision(10);
    int n;
    std::cin >> n;
    std::vector<double> first(n), second(n), result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        std::cin >> first[i];
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> second[i];
    }
    std::vector<std::thread> threads;
    int threadsNum = std::min(n, 1000);
    threads.reserve(threadsNum);
#ifdef BENCHMARK
    auto start = std::chrono::high_resolution_clock::now();
#endif
    for (int i = 0; i < threadsNum; ++i) {
        threads.emplace_back(threadFunc, std::ref(first), std::ref(second), std::ref(result), i, threadsNum);
    }
    for (int i = 0; i < threadsNum; ++i) {
        threads[i].join();
    }
#ifdef BENCHMARK
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "Elapsed time for call kernel: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;
#endif
#ifndef BENCHMARK
    for (auto val: result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
#endif
}