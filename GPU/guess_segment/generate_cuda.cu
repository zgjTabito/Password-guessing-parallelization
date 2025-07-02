#include "PCFG.h"
using namespace std;

__global__ void generate_single_segment_kernel_optimized(
    const char* d_buffer, const int* d_offsets,
    int count, char* d_output, int max_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const char* src = d_buffer + d_offsets[idx];
        char* dst = &d_output[idx * max_len];

        int i = 0;
        while (i < max_len - 1 && src[i] != '\0') {
            dst[i] = src[i];
            ++i;
        }
        dst[i] = '\0';
    }
}


__global__ void generate_multi_segment_kernel_offset(
    const char* d_prefix, int prefix_len,
    const char* d_suffix_buffer, const int* d_suffix_offsets,
    int suffix_count, char* d_output, int max_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < suffix_count) {
        char* out_ptr = &d_output[idx * max_len];

        // 拷贝 prefix
        for (int i = 0; i < prefix_len; ++i)
            out_ptr[i] = d_prefix[i];

        // 拷贝 suffix
        const char* src = d_suffix_buffer + d_suffix_offsets[idx];
        int j = 0;
        while (j < max_len - prefix_len - 1 && src[j] != '\0') {
            out_ptr[prefix_len + j] = src[j];
            ++j;
        }
        out_ptr[prefix_len + j] = '\0';
    }
}

void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    int max_len = 128;

    if (pt.content.size() == 1) {
        segment* a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int N = pt.max_indices[0];

        // 构建 host buffer + offsets
        std::vector<int> h_offsets(N);
        int total_bytes = 0;
        for (int i = 0; i < N; ++i) {
            h_offsets[i] = total_bytes;
            total_bytes += a->ordered_values[i].size() + 1;  // for '\0'
        }

        char* h_buffer = new char[total_bytes];
        for (int i = 0; i < N; ++i)
            strcpy(h_buffer + h_offsets[i], a->ordered_values[i].c_str());

        // device malloc and copy
        char* d_buffer;
        cudaMalloc(&d_buffer, total_bytes);
        cudaMemcpy(d_buffer, h_buffer, total_bytes, cudaMemcpyHostToDevice);

        int* d_offsets;
        cudaMalloc(&d_offsets, N * sizeof(int));
        cudaMemcpy(d_offsets, h_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        char* d_output;
        cudaMalloc(&d_output, N * max_len);

        // Launch kernel
        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        generate_single_segment_kernel_optimized<<<numBlocks, blockSize>>>(
            d_buffer, d_offsets, N, d_output, max_len
        );
        cudaDeviceSynchronize();

        // Copy back
        char* h_output = new char[N * max_len];
        cudaMemcpy(h_output, d_output, N * max_len, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            guesses.emplace_back(&h_output[i * max_len]);
            total_guesses++;
        }

        // Cleanup
        delete[] h_buffer;
        delete[] h_output;
        cudaFree(d_buffer);
        cudaFree(d_offsets);
        cudaFree(d_output);

    } else {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment* a;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        if (pt.content.back().type == 3)
            a = &m.symbols[m.FindSymbol(pt.content.back())];

        int N = pt.max_indices[pt.content.size() - 1];

        // 准备 prefix
        char* d_prefix;
        cudaMalloc(&d_prefix, guess.length());
        cudaMemcpy(d_prefix, guess.data(), guess.length(), cudaMemcpyHostToDevice);

        // 将所有 suffix 串联为连续 buffer + offset
        std::vector<int> h_offsets(N);
        int total_bytes = 0;
        for (int i = 0; i < N; ++i) {
            h_offsets[i] = total_bytes;
            total_bytes += a->ordered_values[i].size() + 1;
        }

        char* h_suffix_buffer = new char[total_bytes];
        for (int i = 0; i < N; ++i)
            strcpy(h_suffix_buffer + h_offsets[i], a->ordered_values[i].c_str());

        // 拷贝 suffix buffer 和 offset 到 device
        char* d_suffix_buffer;
        cudaMalloc(&d_suffix_buffer, total_bytes);
        cudaMemcpy(d_suffix_buffer, h_suffix_buffer, total_bytes, cudaMemcpyHostToDevice);

        int* d_suffix_offsets;
        cudaMalloc(&d_suffix_offsets, N * sizeof(int));
        cudaMemcpy(d_suffix_offsets, h_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // 分配输出 buffer
        char* d_output;
        cudaMalloc(&d_output, N * max_len);

        // Launch kernel
        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        generate_multi_segment_kernel_offset<<<numBlocks, blockSize>>>(
            d_prefix, guess.length(),
            d_suffix_buffer, d_suffix_offsets,
            N, d_output, max_len
        );
        cudaDeviceSynchronize();

        // 结果拷回 host
        char* h_output = new char[N * max_len];
        cudaMemcpy(h_output, d_output, N * max_len, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            guesses.emplace_back(&h_output[i * max_len]);
            total_guesses++;
        }

        // 清理
        delete[] h_suffix_buffer;
        delete[] h_output;
        cudaFree(d_output);
        cudaFree(d_suffix_offsets);
        cudaFree(d_suffix_buffer);
        cudaFree(d_prefix);
    }
}
