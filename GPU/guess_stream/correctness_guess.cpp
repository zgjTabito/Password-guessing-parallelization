#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <cuda_runtime.h>
#include "generate_cuda.h"
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
using namespace std;
using namespace chrono;


#define MAX_GUESS_LEN 64  // 假设单个猜测最大64字节

// 编译指令如下
//nvcc correctness_guess.cpp guessing.cpp train.cpp md5.cpp generate_cuda.cu -o main
//nvcc -O1 correctness_guess.cpp guessing.cpp train.cpp md5.cpp generate_cuda.cu -o main
//nvcc -O2 correctness_guess.cpp guessing.cpp train.cpp md5.cpp generate_cuda.cu -o main

std::mutex mtx;
std::condition_variable cv;
std::queue<std::vector<std::string>> guess_queue;
bool done = false;

void GPUThread(PriorityQueue& q,double& time_guess,int& total_guesses) {
    int curr_num = 0;
    int history = 0;
    auto start = std::chrono::system_clock::now();
    while (!q.priority.empty()) {
        auto pts = q.PopBatchAndSchedule(16);
        std::vector<std::string> batch;
        GeneratePTGuessesBatchCUDA(pts, q.m, batch);
        q.guesses.insert(q.guesses.end(), batch.begin(), batch.end());
        q.total_guesses = q.guesses.size();
        {
            std::unique_lock<std::mutex> lock(mtx);
            guess_queue.push(std::move(batch));
        }
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > generate_n)
            {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    guess_queue.push(std::move(batch));  
                    done = true;                         
                }
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess << "seconds"<< endl;
                cv.notify_all();  // 通知CPU也终止
                break;
            }
        }
        history += curr_num;
        cv.notify_one();
    }
    cv.notify_one(); 
}

void CPUHashThread(std::unordered_set<std::string>& test_set, int& cracked, double& time_hash) {
    bit32 state[4];

    while (true) {
        std::vector<std::string> batch;

        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [] { return !guess_queue.empty() || done; });
            if (guess_queue.empty() && done) break;
            batch = std::move(guess_queue.front());
            guess_queue.pop();
        }

        auto start_hash = system_clock::now();
        for (const std::string& pw : batch) {
            if (test_set.find(pw) != test_set.end()) cracked++;
            MD5Hash(pw, state); // 或 SIMD hash
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }
}

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    int total_guesses = 0; // 总猜测数
    PriorityQueue q;


    // ---------------- CUDA 初始化隔离器 ----------------
    cudaFree(0);  // 触发 CUDA runtime 的初始化
    // ---------------------------------------------------

    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    q.init();

    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("./input/Rockyou-singleLined-full.txt");

    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    cout<<test_count<<endl;
    int cracked=0;

    cout << "here" << endl;
        // 启动两个线程
    std::thread gpu_thread(GPUThread, std::ref(q), std::ref(time_guess), std::ref(total_guesses));
    std::thread cpu_thread(CPUHashThread, std::ref(test_set), std::ref(cracked), std::ref(time_hash));


    gpu_thread.join();
    cpu_thread.join();

    std::cout << "Train time : " << time_train << " seconds" << std::endl;
    std::cout << "Hash time  : " << time_hash << " seconds" << std::endl;
    std::cout << "Cracked    : " << cracked << " / " << test_count << std::endl;


}