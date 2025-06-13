#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <unordered_set>
#include <thread>
#include <sstream>

using namespace std;
using namespace chrono;

std::string Serialize(const PT &pt) {
    std::ostringstream oss;

    // content.size()
    oss << pt.content.size() << "\n";
    for (const segment &seg : pt.content) {
        oss << seg.type << " " << seg.length << "\n";
    }

    // pivot
    oss << pt.pivot << "\n";

    // curr_indices
    oss << pt.curr_indices.size() << "\n";
    for (int idx : pt.curr_indices) {
        oss << idx << " ";
    }
    oss << "\n";

    // max_indices
    oss << pt.max_indices.size() << "\n";
    for (int idx : pt.max_indices) {
        oss << idx << " ";
    }
    oss << "\n";

    // preterm_prob 和 prob
    oss << pt.preterm_prob << " " << pt.prob << "\n";

    return oss.str();
}

PT Deserialize(const std::string &str) {
    std::istringstream iss(str);
    PT pt;

    int content_size;
    iss >> content_size;
    for (int i = 0; i < content_size; ++i) {
        int type, length;
        iss >> type >> length;
        pt.content.emplace_back(type, length);
    }

    iss >> pt.pivot;

    int curr_size;
    iss >> curr_size;
    pt.curr_indices.resize(curr_size);
    for (int i = 0; i < curr_size; ++i) {
        iss >> pt.curr_indices[i];
    }

    int max_size;
    iss >> max_size;
    pt.max_indices.resize(max_size);
    for (int i = 0; i < max_size; ++i) {
        iss >> pt.max_indices[i];
    }

    iss >> pt.preterm_prob >> pt.prob;

    return pt;
}

void SendToWorker(const PT &pt) {
    static int round_robin_rank = 1;  // 简单轮转分发
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string s = Serialize(pt);
    int len = s.size();

    MPI_Send(&len, 1, MPI_INT, round_robin_rank, 0, MPI_COMM_WORLD);
    MPI_Send(s.data(), len, MPI_CHAR, round_robin_rank, 0, MPI_COMM_WORLD);

    round_robin_rank++;
    if (round_robin_rank >= size) round_robin_rank = 1;
}


void MasterLoop(PriorityQueue &q) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int active_workers = size - 1;
    int curr_num = 0;      // 当前批次猜测数
    int history = 0;       // 历史总猜测数
    int cracked = 0;       // 主进程统计总猜中数
    int generate_n = 10000000;
    auto start = system_clock::now();  // 计时起点
    double total_hash_time = 0;  // 所有worker hash耗时之和
    while (!q.priority.empty()) {
        
        PT pt = q.priority.front();

        // step 1: 分发 Generate 到 worker
        SendToWorker(pt);
        // Step 2: 接收 worker 返回的 cracked 数量
        int local_cracked = 0;
        MPI_Status status;
        MPI_Recv(&local_cracked, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        double local_hash_time = 0;
        MPI_Recv(&local_hash_time, 1, MPI_DOUBLE, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_hash_time += local_hash_time;

        cracked += local_cracked;

        // step 2: 本地执行 PopNext（只做扩展和队列管理）
        q.PopNext();  // 使用你刚写的这个版本
        int guess_count = 0;
        MPI_Recv(&guess_count, 1, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.total_guesses += guess_count;  // ⬅ 这才是真正的 guess 累加


        if (q.total_guesses - curr_num >= 100000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 判断是否已达到上限
            if (history + q.total_guesses > generate_n) {
                int stop_flag = 1;
                MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 广播终止标志

                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                double time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guess time:" << time_guess-total_hash_time << " seconds" << endl;

                cout << "Cracked: " << cracked << endl;

                // 提前通知 worker 并退出
                for (int i = 1; i < size; i++) {
                    int stop_flag = -1;
                    MPI_Send(&stop_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
                return;
            }
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();      
    }
  }

    // step 3: 通知所有从进程退出
    for (int i = 1; i < size; i++) {
        int stop_flag = -1;
        MPI_Send(&stop_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
}




void WorkerLoop(PriorityQueue &q, const unordered_set<string> &test_set) {
    MPI_Status status;
    while (true) {
        int flag = 0;
        int len = 0;

        MPI_Iprobe(0, 0, MPI_COMM_WORLD, &flag, &status); // 非阻塞探测是否有任务
        if (!flag) {
            this_thread::sleep_for(chrono::milliseconds(1));  // 稍作等待，避免忙等
            continue;
        }

        MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (len == -1) break;

        std::vector<char> buffer(len);
        MPI_Recv(buffer.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        std::string str(buffer.begin(), buffer.end());

        PT pt = Deserialize(str);

        q.Generate(pt);  // 单纯执行 Generate，不改队列

        // ====== 新增：统计被 test_set 命中的个数 ======
        int local_cracked = 0;

        uint32x4_t state[4];   // 用于存储 SIMD MD5 的结果
        vector<string> batch;  // 用于存储每次处理的四个口令
        double local_hash_time=0;
        auto start_hash = system_clock::now();
        for (size_t i = 0; i < q.guesses.size(); i++) {
            batch.push_back(q.guesses[i]);

            // 当 batch 中有 4 个口令时，调用 SIMD MD5Hash 函数
            if (batch.size() == 4) {
            // 检查 batch 中的口令是否在 test_set 中
            for (const auto &pw : batch) {
                if (test_set.find(pw) != test_set.end()) {
                local_cracked += 1;
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
                local_cracked += 1;
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
        local_hash_time += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        int guess_count = q.guesses.size();  // 记录本轮猜测数
        
        MPI_Send(&local_cracked, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&local_hash_time, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&guess_count, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);

        q.guesses.clear();
    }
}



int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double time_hash = 0, time_guess = 0, time_train = 0;
  PriorityQueue q;

  auto start_train = system_clock::now();
  q.m.train("/guessdata/Rockyou-singleLined-full.txt");
  q.m.order();
  auto end_train = system_clock::now();
  auto duration_train = duration_cast<microseconds>(end_train - start_train);
  time_train = double(duration_train.count()) * microseconds::period::num /
               microseconds::period::den;
  q.init();

  // 加载测试集（所有进程都需要）
  unordered_set<string> test_set;
  ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
  string pw;
  int test_count = 0;
  while (test_data >> pw && test_count < 1000000) {
    test_set.insert(pw);
    test_count++;
  }
  cout<<"here"<<endl;

  // === 主从进程分工 ===
  if (rank == 0) {
    MasterLoop(q);  // 动态调度任务
  } else {
    WorkerLoop(q, test_set);  // 执行任务（需传入模型与测试集）
  }
  cout << "Train time:" << time_train << " seconds" << endl;

  MPI_Finalize();
  return 0;
}