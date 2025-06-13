#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main() {
  double time_hash = 0;  // 用于MD5哈希的时间
  double time_guess = 0; // 哈希和猜测的总时长
  double time_train = 0; // 模型训练的总时长
  PriorityQueue q;
  auto start_train = system_clock::now();
  q.m.train("/guessdata/Rockyou-singleLined-full.txt");
  q.m.order();
  auto end_train = system_clock::now();
  auto duration_train = duration_cast<microseconds>(end_train - start_train);
  time_train = double(duration_train.count()) * microseconds::period::num /
               microseconds::period::den;

  // 加载一些测试数据
  unordered_set<std::string> test_set;
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
  int cracked = 0;

  q.init();
  cout << "here" << endl;
  int curr_num = 0;
  auto start = system_clock::now();
  // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
  int history = 0;
  std::ofstream a("./files/results.txt");
  while (!q.priority.empty()) {
    q.PopNext();
    q.total_guesses = q.guesses.size();
    if (q.total_guesses - curr_num >= 100000) {
      cout << "Guesses generated: " << history + q.total_guesses << endl;
      curr_num = q.total_guesses;

      // 在此处更改实验生成的猜测上限
      int generate_n = 10000000;
      if (history + q.total_guesses > generate_n) {
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num /
                     microseconds::period::den;
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

      // bit32 state[4];
      //  for (string pw : q.guesses)
      //  {
      //      if (test_set.find(pw) != test_set.end()) {
      //          cracked+=1;
      //      }
      //      // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
      //      MD5Hash(pw, state);

      //     //
      //     以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
      //     // a<<pw<<"\t";
      //     // for (int i1 = 0; i1 < 4; i1 += 1)
      //     // {
      //     //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
      //     // }
      //     // a << endl;
      // }

      uint32x4_t state[4];  // 用于存储 SIMD MD5 的结果
      vector<string> batch; // 用于存储每次处理的四个口令

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
          batch.clear(); // 清空 batch，准备处理下一组口令
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
          batch.push_back(""); // 填充空字符串以满足 4 个的要求
        }
        MD5Hash(batch.data(), state);
      }

      // 在这里对哈希所需的总时长进行计算
      auto end_hash = system_clock::now();
      auto duration = duration_cast<microseconds>(end_hash - start_hash);
      time_hash += double(duration.count()) * microseconds::period::num /
                   microseconds::period::den;

      // 记录已经生成的口令总数
      history += curr_num;
      curr_num = 0;
      q.guesses.clear();
    }
  }
}
// const int max_threads = 8;  // 最大线程数，可根据实际调整
// double thread_hash_times[max_threads] = {0};  // 每个线程的哈希时间
// double thread_guess_times[max_threads] = {0};  // 每个线程的哈希时间
// int generate_n=10000000;
// int history = 0;
// double hash_time = 0.0;
// auto start = system_clock::now();
// #pragma omp parallel num_threads(4)
// {
//     int tid = omp_get_thread_num();  // 获取当前线程编号
//     int local_cracked = 0;
//     while (true) {
//         bool has_task = false;

//         #pragma omp critical(queue_access)
//         {
//             if (!q.priority.empty()) {
//                 q.PopNext();  // 内部调用 Generate()，将生成的猜测放入
//                 q.guesses q.total_guesses = q.guesses.size(); has_task =
//                 true;
//             }
//         }

//         if (!has_task) break;

//         // 打印进度（并发控制）
//         static int curr_num = 0;
//         #pragma omp critical(progress_print)
//         {
//             if (q.total_guesses - curr_num >= 100000) {
//                 cout << "Guesses generated: " << history + q.total_guesses
//                 << endl; curr_num = q.total_guesses;
//             }
//         }

//         // 达到阈值，结束所有线程
//         if (history + q.total_guesses > generate_n) {
//             auto end = system_clock::now();
//             auto duration = duration_cast<microseconds>(end - start);
//             time_guess = double(duration.count()) *
//             microseconds::period::num / microseconds::period::den; #pragma
//             omp critical(print_summary)
//             {
//                 thread_guess_times[tid]=time_guess-thread_hash_times[tid];
//             }
//             #pragma omp atomic
//             cracked += local_cracked;
//             break;
//         }

//         if (curr_num > 1000000) {
//         vector<string> batch;

//             #pragma omp critical(hash_fetch)
//             {
//                 batch = q.guesses;
//                 q.guesses.clear();
//                 history += curr_num;
//                 curr_num = 0;
//             }

//             // 单线程内测量哈希时间
//             auto start_hash = system_clock::now();
//             for (size_t i = 0; i < batch.size(); i += 4) {
//                 vector<string> simd_input;
//                 for (int j = 0; j < 4 && (i + j) < batch.size(); ++j) {
//                     simd_input.push_back(batch[i + j]);
//                 }
//                 while (simd_input.size() < 4) simd_input.push_back("");

//                 for (const auto &pw : simd_input) {
//                     if (test_set.find(pw) != test_set.end()) {
//                         #pragma omp atomic
//                         local_cracked++;
//                     }
//                 }
//                 uint32x4_t state[4];
//                 MD5Hash(simd_input.data(), state);
//             }
//             auto end_hash = system_clock::now();
//             double this_time = duration_cast<microseconds>(end_hash -
//             start_hash).count() / 1e6;

//             // 累加到当前线程的哈希时间
//             thread_hash_times[tid] += this_time;
//         }
//     }
// }
// for (int i = 0; i < 4; ++i) {
//     hash_time+=thread_hash_times[i];
// }
// double guess_time=0.0;
//     for (int i = 0; i < 4; ++i) {
//     guess_time=guess_time>thread_guess_times[i]?guess_time:thread_guess_times[i];
// }
// cout<<"guess time: "<<guess_time<<" seconds" << endl;
// cout << "hash time: " << hash_time << " seconds" << endl;
// cout << "Cracked: " << cracked << endl;
}

