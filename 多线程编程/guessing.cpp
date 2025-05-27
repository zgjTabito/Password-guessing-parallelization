#include "PCFG.h"
#include <pthread.h>
#include<omp.h>
#include <queue>
#include <vector>
#include <functional>
#include <unistd.h>
using namespace std;


//加
class PthreadThreadPool {
public:
    PthreadThreadPool(int num_threads);
    ~PthreadThreadPool();
    void enqueue(std::function<void()> task);
    void wait_all();
    static void* worker(void* arg);

    std::vector<pthread_t> threads;
    std::queue<std::function<void()>> tasks;

    pthread_mutex_t queue_mutex;
    pthread_cond_t task_available;
    pthread_cond_t all_done;

    bool stop;
    int active_tasks;
};
PthreadThreadPool::PthreadThreadPool(int num_threads)
    : stop(false), active_tasks(0) {
    pthread_mutex_init(&queue_mutex, nullptr);
    pthread_cond_init(&task_available, nullptr);
    pthread_cond_init(&all_done, nullptr);

    for (int i = 0; i < num_threads; ++i) {
        pthread_t t;
        pthread_create(&t, nullptr, worker, this);
        threads.push_back(t);
    }
}
PthreadThreadPool::~PthreadThreadPool() {
    pthread_mutex_lock(&queue_mutex);
    stop = true;
    pthread_cond_broadcast(&task_available);
    pthread_mutex_unlock(&queue_mutex);

    for (pthread_t& t : threads) {
        pthread_join(t, nullptr);
    }

    pthread_mutex_destroy(&queue_mutex);
    pthread_cond_destroy(&task_available);
    pthread_cond_destroy(&all_done);
}
void PthreadThreadPool::enqueue(std::function<void()> task) {
    pthread_mutex_lock(&queue_mutex);
    tasks.push(std::move(task));
    pthread_cond_signal(&task_available);
    pthread_mutex_unlock(&queue_mutex);
}
void* PthreadThreadPool::worker(void* arg) {
    auto* pool = static_cast<PthreadThreadPool*>(arg);
    while (true) {
        pthread_mutex_lock(&pool->queue_mutex);
        while (pool->tasks.empty() && !pool->stop) {
            pthread_cond_wait(&pool->task_available, &pool->queue_mutex);
        }

        if (pool->stop && pool->tasks.empty()) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }

        auto task = std::move(pool->tasks.front());
        pool->tasks.pop();
        pool->active_tasks++;
        pthread_mutex_unlock(&pool->queue_mutex);

        task();  // 执行任务

        pthread_mutex_lock(&pool->queue_mutex);
        pool->active_tasks--;
        if (pool->tasks.empty() && pool->active_tasks == 0) {
            pthread_cond_signal(&pool->all_done);  // 通知 wait_all
        }
        pthread_mutex_unlock(&pool->queue_mutex);
    }
    return nullptr;
}
void PthreadThreadPool::wait_all() {
    pthread_mutex_lock(&queue_mutex);
    while (!tasks.empty() || active_tasks > 0) {
        pthread_cond_wait(&all_done, &queue_mutex);  // 阻塞直到条件变量触发
    }
    pthread_mutex_unlock(&queue_mutex);
}



void ParallelGeneratePoolSingleSegment(segment* a, int max_index, std::vector<std::string>& guesses, int& total_guesses, PthreadThreadPool& pool) {
    pthread_mutex_t result_mutex;
    pthread_mutex_init(&result_mutex, nullptr);

    int num_tasks = 4;
    int chunk_size = (max_index + num_tasks - 1) / num_tasks;

    // 主线程直接处理第一个任务块
    {
        int start = 0;
        int end = std::min(chunk_size, max_index);
        std::vector<std::string> local;
        local.reserve(end - start);
        for (int j = start; j < end; ++j) {
            local.emplace_back(a->ordered_values[j]);
        }

        pthread_mutex_lock(&result_mutex);
        guesses.insert(guesses.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
        total_guesses += (end - start);
        pthread_mutex_unlock(&result_mutex);
    }

    // 其余任务块交由线程池处理
    for (int i = chunk_size; i < max_index; i += chunk_size) {
        int start = i;
        int end = std::min(i + chunk_size, max_index);

        pool.enqueue([=, &guesses, &total_guesses, &result_mutex]() {
            std::vector<std::string> local;
            local.reserve(end - start);
            for (int j = start; j < end; ++j) {
                local.emplace_back(a->ordered_values[j]);
            }

            pthread_mutex_lock(&result_mutex);
            guesses.insert(guesses.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
            total_guesses += (end - start);
            pthread_mutex_unlock(&result_mutex);
        });
    }

    pool.wait_all();
    pthread_mutex_destroy(&result_mutex);
}

void ParallelGeneratePoolMultiSegment(segment* a, int max_index, const std::string& guess_prefix, std::vector<std::string>& guesses, int& total_guesses, PthreadThreadPool& pool) {
    pthread_mutex_t result_mutex;
    pthread_mutex_init(&result_mutex, nullptr);

    int num_tasks = 4;
    int chunk_size = (max_index + num_tasks - 1) / num_tasks;

    // 主线程直接处理第一个任务块
    {
        int start = 0;
        int end = std::min(chunk_size, max_index);
        std::vector<std::string> local;
        local.reserve(end - start);
        for (int j = start; j < end; ++j) {
            local.emplace_back(guess_prefix + a->ordered_values[j]);
        }

        pthread_mutex_lock(&result_mutex);
        guesses.insert(guesses.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
        total_guesses += (end - start);
        pthread_mutex_unlock(&result_mutex);
    }

    // 其余任务块交由线程池处理
    for (int i = chunk_size; i < max_index; i += chunk_size) {
        int start = i;
        int end = std::min(i + chunk_size, max_index);

        pool.enqueue([=, &guess_prefix, &guesses, &total_guesses, &result_mutex]() {
            std::vector<std::string> local;
            local.reserve(end - start);
            for (int j = start; j < end; ++j) {
                local.emplace_back(guess_prefix + a->ordered_values[j]);
            }

            pthread_mutex_lock(&result_mutex);
            guesses.insert(guesses.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
            total_guesses += (end - start);
            pthread_mutex_unlock(&result_mutex);
        });
    }

    pool.wait_all();
    pthread_mutex_destroy(&result_mutex);
}






void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


//加
// 线程参数结构体
struct ThreadData {
    int start;
    int end;
    segment* a;
    string guess_prefix;
    vector<string>* local_guesses;
    int* local_count;
};

// 单段处理函数（不带前缀）
void* threadTaskSingleSegment(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int i = data->start; i < data->end; i++) {
        data->local_guesses->emplace_back(data->a->ordered_values[i]);
        (*data->local_count)++;
    }

    return nullptr;
}

// 多段处理函数（带前缀）
void* threadTaskMultiSegment(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int i = data->start; i < data->end; i++) {
        string temp = data->guess_prefix + data->a->ordered_values[i];
        data->local_guesses->emplace_back(temp);
        (*data->local_count)++;
    }

    return nullptr;
}

// 优化后的并行处理（单段）
void ParallelGeneratePthreadSingleSegment(segment* a, int max_index, vector<string>& guesses, int& total_guesses) {
    const int num_threads = 3;  // 子线程数
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    vector<string> local_guesses[num_threads];
    int local_counts[num_threads] = {0};

    int num_chunks = num_threads + 1; // 包含主线程的总任务数
    int chunk_size = (max_index + num_chunks - 1) / num_chunks;

    // 主线程处理第一个任务块
    int start_main = 0;
    int end_main = std::min(chunk_size, max_index);
    vector<string> local_main;
    int local_main_count = 0;
    for (int i = start_main; i < end_main; ++i) {
        local_main.emplace_back(a->ordered_values[i]);
        local_main_count++;
    }

    // 子线程处理剩余任务块
    for (int t = 0; t < num_threads; t++) {
        int start = chunk_size * (t + 1);
        int end = std::min(start + chunk_size, max_index);

        thread_data[t].start = start;
        thread_data[t].end = end;
        thread_data[t].a = a;
        thread_data[t].local_guesses = &local_guesses[t];
        thread_data[t].local_count = &local_counts[t];

        pthread_create(&threads[t], nullptr, threadTaskSingleSegment, &thread_data[t]);
    }

    // 主线程合并结果
    guesses.insert(guesses.end(), local_main.begin(), local_main.end());
    total_guesses += local_main_count;

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
        guesses.insert(guesses.end(), local_guesses[t].begin(), local_guesses[t].end());
        total_guesses += local_counts[t];
    }
}


void ParallelGeneratePthreadMultiSegment(segment* a, int max_index, const string& guess, vector<string>& guesses, int& total_guesses) {
    const int num_threads = 3;  // 子线程数
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    vector<string> local_guesses[num_threads];
    int local_counts[num_threads] = {0};

    int num_chunks = num_threads + 1; // 总任务数 = 主线程 + 子线程
    int chunk_size = (max_index + num_chunks - 1) / num_chunks;

    // 主线程处理第一个任务块
    int start_main = 0;
    int end_main = std::min(chunk_size, max_index);
    vector<string> local_main;
    int local_main_count = 0;
    for (int i = start_main; i < end_main; ++i) {
        string temp = guess + a->ordered_values[i];
        local_main.emplace_back(temp);
        local_main_count++;
    }

    // 子线程处理剩余任务块
    for (int t = 0; t < num_threads; t++) {
        int start = chunk_size * (t + 1);
        int end = std::min(start + chunk_size, max_index);

        thread_data[t].start = start;
        thread_data[t].end = end;
        thread_data[t].a = a;
        thread_data[t].guess_prefix = guess;
        thread_data[t].local_guesses = &local_guesses[t];
        thread_data[t].local_count = &local_counts[t];

        pthread_create(&threads[t], nullptr, threadTaskMultiSegment, &thread_data[t]);
    }

    // 主线程合并结果
    guesses.insert(guesses.end(), local_main.begin(), local_main.end());
    total_guesses += local_main_count;

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
        guesses.insert(guesses.end(), local_guesses[t].begin(), local_guesses[t].end());
        total_guesses += local_counts[t];
    }
}



// 使用 OpenMP 并行化的代码（第一个循环）
void ParallelGenerateOpenmpSingleSegment(segment* a, int max_index, vector<string>& guesses, int& total_guesses) {
    // OpenMP 需要对共享资源加锁
    #pragma omp parallel num_threads(8)
    {
        // 定义一个局部变量用于存储线程的临时结果
        vector<string> local_guesses;
        int local_total_guesses = 0;

        // 并行 for 循环
        #pragma omp for
        for (int i = 0; i < max_index; i++) {
            string guess = a->ordered_values[i];
            local_guesses.emplace_back(guess);
            local_total_guesses++;
        }

        // 合并线程的局部结果到全局结果
        #pragma omp critical
        {
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_total_guesses;
        }
    }
}


// 使用 OpenMP 并行化的代码（第二个循环）
void ParallelGenerateOpenmpMultiSegment(segment* a, int max_index, const string& guess, vector<string>& guesses, int& total_guesses) {
    // OpenMP 需要对共享资源加锁
    #pragma omp parallel num_threads(8)
    {
        // 定义一个局部变量用于存储线程的临时结果
        vector<string> local_guesses;
        int local_total_guesses = 0;

        // 并行 for 循环
        #pragma omp for
        for (int i = 0; i < max_index; i++) {
            string temp = guess + a->ordered_values[i];
            local_guesses.emplace_back(temp);
            local_total_guesses++;
        }

        // 合并线程的局部结果到全局结果
        #pragma omp critical
        {
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += local_total_guesses;
        }
    }
}



// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    const int PARALLEL_THRESHOLD = 64;
    static PthreadThreadPool pool(4);  // 仅构造一次
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
        // 使用 pthread/openmp 并行化
        //ParallelGeneratePthreadSingleSegment(a, pt.max_indices[0], guesses, total_guesses);
        //ParallelGeneratePoolSingleSegment(a, pt.max_indices[0], guesses, total_guesses, pool);
        //ParallelGenerateOpenmpSingleSegment(a, pt.max_indices[0], guesses, total_guesses);
//         if (pt.max_indices[0] < 256) {
//             for (int i = 0; i < pt.max_indices[0]; ++i) {
//         string guess = a->ordered_values[i];
//         guesses.emplace_back(guess);
//         total_guesses++;
//     }
//         } else {
//     // 大任务并行处理
//     //ParallelGeneratePoolSingleSegment(a, pt.max_indices[0], guesses, total_guesses, pool);
//     ParallelGenerateOpenmpSingleSegment(a, pt.max_indices[0], guesses, total_guesses);
// }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
         // 并行化
         //ParallelGeneratePthreadMultiSegment(a, pt.max_indices[pt.content.size() - 1], guess, guesses, total_guesses);
         //ParallelGeneratePoolMultiSegment(a, pt.max_indices[pt.content.size() - 1], guess, guesses, total_guesses, pool);
        // ParallelGenerateOpenmpMultiSegment(a, pt.max_indices[pt.content.size() - 1], guess, guesses, total_guesses);

//          if (pt.max_indices[0] < 256) {
//     // 小任务串行执行，避免调度开销
//                     for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
//         {
//             string temp = guess + a->ordered_values[i];
//             // cout << temp << endl;
//             guesses.emplace_back(temp);
//             total_guesses += 1;
//         }
//         } else {
//     // 大任务并行处理
//     //ParallelGeneratePoolMultiSegment(a, pt.max_indices[pt.content.size() - 1], guess, guesses, total_guesses, pool);
//     ParallelGenerateOpenmpMultiSegment(a, pt.max_indices[pt.content.size() - 1], guess, guesses, total_guesses);
// }

    }
}