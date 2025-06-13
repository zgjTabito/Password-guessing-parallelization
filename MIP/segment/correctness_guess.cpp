#include <mpi.h>  // 顶部加头文件
#include <chrono>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include "PCFG.h"
#include "md5.h"

using namespace std;
using namespace chrono;

// 编译指令如下
// g++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // 初始化MPI环境
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;   // 用于MD5哈希的时间
    double time_guess = 0;  // 哈希和猜测的总时长
    double time_train = 0;  // 模型训练的总时长
    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;



    q.init();
    cout << "here" << endl;
   
    if (rank == 0){
        int cracked = 0;
        int curr_num = 0;
        unordered_set<std::string> test_set;
        // 加载一些测试数据
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while (test_data >> pw) {
            test_count += 1;
            test_set.insert(pw);
            if (test_count >= 1000000) {
                break;
            }
        }

        auto start = system_clock::now();
        // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
        int history = 0;
        std::ofstream a("./files/results.txt");
        while (!q.priority.empty()) {
            int stop_flag = 0;
            MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 继续
            q.PopNext();
            q.total_guesses = q.guesses.size();
            if (q.total_guesses - curr_num >= 100000) {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                // 在此处更改实验生成的猜测上限
                int generate_n = 10000000;
                if (history + q.total_guesses > generate_n) {
                    int stop_flag = 1;
                    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    cout << "Cracked:" << cracked << endl;
                    break;
                }
            }
            // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
            // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
            if (curr_num > 1000000) {
                auto start_hash = system_clock::now();

                uint32x4_t state[4];   // 用于存储 SIMD MD5 的结果
                vector<string> batch;  // 用于存储每次处理的四个口令

                for (size_t i = 0; i < q.guesses.size(); i++) {
                    batch.push_back(q.guesses[i]);

                    // 当 batch 中有 4 个口令时，调用 SIMD MD5Hash 函数
                    if (batch.size() == 4) {
                    // 检查 batch 中的口令是否在 test_set 中
                    for (const auto &pw : batch) {
                        if (test_set.find(pw) != test_set.end()) {
                        cracked += 1;
                        }
                    }
                    MD5Hash(batch.data(), state);
                    batch.clear();  // 清空 batch，准备处理下一组口令
                    }
                }

                // 如果剩余的口令不足 4 个，单独处理
                if (!batch.empty()) {
                    // 检查 batch 中的口令是否在 test_set 中
                    for (const auto &pw : batch) {
                    if (test_set.find(pw) != test_set.end()) {
                        cracked += 1;
                    }
                    }
                    while (batch.size() < 4) {
                    batch.push_back("");  // 填充空字符串以满足 4 个的要求
                    }
                    MD5Hash(batch.data(), state);
                }

                // 在这里对哈希所需的总时长进行计算
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
    }
    else {
        while (true) {
            // 检查是否需要退出
            int stop_flag;
            MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 主进程广播停止标志

            if (stop_flag) {
                break;  // 主进程已完成任务，退出循环
            }

            // 非主进程只参与 Generate()，不管结果
            q.PopNext();  // 内部执行 Generate 并 MPI_Gather
        }
    }

    MPI_Finalize();  // 在main函数末尾释放MPI资源
    return 0;
}