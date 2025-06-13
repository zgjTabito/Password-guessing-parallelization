#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <unordered_set>
#include <thread>
#include <sstream>

#define MASTER_RANK 0
#define GEN_START_RANK 1
#define GEN_END_RANK 4
#define HASH_START_RANK 5


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
    static int round_robin_rank = GEN_START_RANK;  // 简单轮转分发
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string s = Serialize(pt);
    int len = s.size();
    MPI_Send(&len, 1, MPI_INT, round_robin_rank, 0, MPI_COMM_WORLD);
    MPI_Send(s.data(), len, MPI_CHAR, round_robin_rank, 0, MPI_COMM_WORLD);

    round_robin_rank++;
    if (round_robin_rank > GEN_END_RANK) round_robin_rank = GEN_START_RANK;
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

        cracked += local_cracked;

        // step 2: 本地执行 PopNext（只做扩展和队列管理）
        q.PopNext();  // 使用你刚写的这个版本
        int guess_count = 0;
        MPI_Recv(&guess_count, 1, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.total_guesses += guess_count; 


        if (q.total_guesses - curr_num >= 100000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 判断是否已达到上限
            if (history + q.total_guesses > generate_n) {
                int stop_flag = 1;
                MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 广播终止标志
                // 通知生成进程退出
                for (int i = GEN_START_RANK; i <= GEN_END_RANK; i++) {
                    int stop_flag = -1;
                    MPI_Send(&stop_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }

                // 通知哈希进程退出（必须在接收耗时之前）
                for (int i = HASH_START_RANK; i < size; i++) {
                    int flag = -1;
                    MPI_Send(&flag, 1, MPI_INT, i, 10, MPI_COMM_WORLD);
                }
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                double time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                double max_hash_time = 0;
                for (int i = HASH_START_RANK; i < size; i++) {
                    double worker_wall_time;
                    MPI_Recv(&worker_wall_time, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    max_hash_time = std::max(max_hash_time, worker_wall_time);
                    total_hash_time +=worker_wall_time;
                    std::cout << "[Master] Hash Worker " << i << " wall-clock time: " << worker_wall_time << "s" << std::endl;
                }




                cout << "Guess time:" << time_guess-max_hash_time << " seconds" << endl;
                cout << "Hash time: " << total_hash_time << " seconds" << endl;
                cout << "Cracked: " << cracked << endl;

                // // 提前通知 worker 并退出
                // for (int i = GEN_START_RANK; i <= GEN_END_RANK; i++) {
                //     int stop_flag = -1;
                //     MPI_Send(&stop_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // }
                // for (int i = HASH_START_RANK; i < size; i++) {
                //     int flag = -1;
                //     MPI_Send(&flag, 1, MPI_INT, i, 10, MPI_COMM_WORLD);
                // }
                return;
            }
            // 检查是否有来自任何HashWorker的 hash_time 消息（tag=2）
            //MPI_Status hash_status;
            // int hash_flag = 0;
            // MPI_Iprobe(MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &hash_flag, &hash_status);
            // if (hash_flag) {
            //     double local_hash_time;
            //     MPI_Recv(&local_hash_time, 1, MPI_DOUBLE, hash_status.MPI_SOURCE, 2, MPI_COMM_WORLD, &hash_status);
            //     total_hash_time += local_hash_time;
            // }

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();      
    }
  }
}


void WorkerGenerateLoop(PriorityQueue &q,const unordered_set<string> &test_set) {
    MPI_Status status;
    while (true) {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int gen_rank, hash_base;
        MPI_Comm_rank(MPI_COMM_WORLD, &gen_rank);
        if (len == -1) break;

        std::vector<char> buffer(len);
        MPI_Recv(buffer.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        std::string str(buffer.begin(), buffer.end());
        PT pt = Deserialize(str);


        q.Generate(pt);
        // 验证密码是否在 test_set 中
        int cracked = 0;
        for (const string &guess : q.guesses) {
            if (test_set.count(guess)) cracked++;
        }

        // 每4个一组发给哈希进程
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        hash_base = size / 2;

        int hash_target = HASH_START_RANK + (gen_rank - GEN_START_RANK) % (size - HASH_START_RANK);

        // 将所有猜测拼接为一个字符串
        string combined;
        for (const string &g : q.guesses) {
            combined += g + "\n";
        }

        // 发送数据长度 + 实际内容
        int l = combined.size();
        MPI_Send(&l, 1, MPI_INT, hash_target, 10, MPI_COMM_WORLD);
        MPI_Send(combined.c_str(), l, MPI_CHAR, hash_target, 10, MPI_COMM_WORLD);

        // 通知主进程本轮生成结束（猜测总数）
        int count = q.guesses.size();
        MPI_Send(&count, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(&cracked, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        q.guesses.clear();
    }
}

void WorkerHashLoop() {
    MPI_Status status;
    auto start = system_clock::now();
    while (true) {
        int len;
        MPI_Recv(&len, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
        if (len == -1) break;

        std::vector<char> buf(len);
        MPI_Recv(buf.data(), len, MPI_CHAR, status.MPI_SOURCE, 10, MPI_COMM_WORLD, &status);
        std::istringstream iss(std::string(buf.begin(), buf.end()));
        vector<string> batch;
        string line;

        while (getline(iss, line)) {
            batch.push_back(line);
        }

        // 哈希
        uint32x4_t state[4];
        while (batch.size() < 4) batch.push_back("");
        MD5Hash(batch.data(), state);
    }
           
        auto end = system_clock::now();
        double hash_time = duration_cast<nanoseconds>(end - start).count() * 1e-9;
        // 返回给主进程
        MPI_Send(&hash_time, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
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
  cout << "Train time:" << time_train << " seconds" << endl;
if (rank == 0) {
    MasterLoop(q);
} else if (rank<=GEN_END_RANK) {
    WorkerGenerateLoop(q,test_set);
} else {
    WorkerHashLoop();
}


  MPI_Finalize();
  return 0;
}