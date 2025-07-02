#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include "PCFG.h"



#define MAX_GUESS_LEN 64  // 假设单个猜测最大64字节

__global__ void generate_guesses_kernel(
    const char* __restrict__ d_prefix_buffer,
    const int* __restrict__ d_prefix_offsets,
    const char* __restrict__ d_suffix_buffer,
    const int* __restrict__ d_suffix_offsets,
    const int* __restrict__ d_suffix_lengths,
    const int* __restrict__ d_pt_indices,
    int total_guesses,
    char* __restrict__ d_output_buffer,
    int max_guess_len
)
 {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_guesses) return;

    int pt_idx = d_pt_indices[idx];
    const char* prefix = d_prefix_buffer + d_prefix_offsets[pt_idx];
    const char* suffix = d_suffix_buffer + d_suffix_offsets[idx];
    int suffix_len = d_suffix_lengths[idx];

    char* out = d_output_buffer + idx * max_guess_len;

    // 拼接 prefix
    int p = 0;
    while (prefix[p] != '\0' && p < max_guess_len - 1) {
        out[p] = prefix[p];
        ++p;
    }

    // 拼接 suffix
    for (int j = 0; j < suffix_len && p < max_guess_len - 1; ++j, ++p) {
        out[p] = suffix[j];
    }
    out[p] = '\0';
}


void GeneratePTGuessesBatchCUDA(const vector<PT>& pts, model& m, vector<string>& output_guesses) {
    vector<string> prefixes;
    vector<string> suffixes;
    vector<int> pt_indices;

    vector<int> prefix_offsets;
    string prefix_buffer;
    vector<int> suffix_offsets;
    vector<int> suffix_lengths;
    string suffix_buffer;

    int total = 0;

    for (int pt_idx = 0; pt_idx < pts.size(); ++pt_idx) {
        const PT& pt = pts[pt_idx];

        // 1. 生成 prefix
        string prefix;
        for (int seg_i = 0; seg_i < pt.content.size() - 1; ++seg_i) {
            const segment& seg = pt.content[seg_i];
            int idx = pt.curr_indices[seg_i];

            if (seg.type == 1)
                prefix += m.letters[m.FindLetter(seg)].ordered_values[idx];
            else if (seg.type == 2)
                prefix += m.digits[m.FindDigit(seg)].ordered_values[idx];
            else if (seg.type == 3)
                prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
        }
        prefix_offsets.push_back(prefix_buffer.size());
        prefix_buffer += prefix;
        prefix_buffer.push_back('\0');  // null-terminate

        // 2. 获取 suffix 值
        const segment& last_seg = pt.content.back();
        const vector<string>* suffix_values = nullptr;
        if (last_seg.type == 1)
            suffix_values = &m.letters[m.FindLetter(last_seg)].ordered_values;
        else if (last_seg.type == 2)
            suffix_values = &m.digits[m.FindDigit(last_seg)].ordered_values;
        else if (last_seg.type == 3)
            suffix_values = &m.symbols[m.FindSymbol(last_seg)].ordered_values;

        for (const string& sfx : *suffix_values) {
            suffix_offsets.push_back(suffix_buffer.size());
            suffix_lengths.push_back(sfx.size());
            suffix_buffer += sfx;
            pt_indices.push_back(pt_idx);
            ++total;
        }
    }

    // 分配和拷贝数据
    char* d_prefix_buffer;
    char* d_suffix_buffer;
    char* d_output_buffer;
    int* d_prefix_offsets;
    int* d_suffix_offsets;
    int* d_suffix_lengths;
    int* d_pt_indices;

    cudaMalloc(&d_prefix_buffer, prefix_buffer.size());
    cudaMalloc(&d_prefix_offsets, sizeof(int) * prefix_offsets.size());
    cudaMalloc(&d_suffix_buffer, suffix_buffer.size());
    cudaMalloc(&d_suffix_offsets, sizeof(int) * suffix_offsets.size());
    cudaMalloc(&d_suffix_lengths, sizeof(int) * suffix_lengths.size());
    cudaMalloc(&d_pt_indices, sizeof(int) * pt_indices.size());
    cudaMalloc(&d_output_buffer, total * MAX_GUESS_LEN);

    cudaMemcpy(d_prefix_buffer, prefix_buffer.data(), prefix_buffer.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_offsets, prefix_offsets.data(), prefix_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_buffer, suffix_buffer.data(), suffix_buffer.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_offsets, suffix_offsets.data(), suffix_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_lengths, suffix_lengths.data(), suffix_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_indices, pt_indices.data(), pt_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    generate_guesses_kernel<<<blocks, threads>>>(
        d_prefix_buffer, d_prefix_offsets,
        d_suffix_buffer, d_suffix_offsets,
        d_suffix_lengths,
        d_pt_indices, total,
        d_output_buffer, MAX_GUESS_LEN
    );

    // 复制结果回 host
    vector<char> h_output(total * MAX_GUESS_LEN);
    cudaMemcpy(h_output.data(), d_output_buffer, h_output.size(), cudaMemcpyDeviceToHost);

    // 转换为 string
    output_guesses.clear();
    for (int i = 0; i < total; ++i) {
        output_guesses.emplace_back(h_output.data() + i * MAX_GUESS_LEN);
    }

    // 清理
    cudaFree(d_prefix_buffer);
    cudaFree(d_prefix_offsets);
    cudaFree(d_suffix_buffer);
    cudaFree(d_suffix_offsets);
    cudaFree(d_suffix_lengths);
    cudaFree(d_pt_indices);
    cudaFree(d_output_buffer);
}
