#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
#include <cstring>
#include <iostream>
#include<omp.h>

typedef uint32_t bit32;

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
 Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits += 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}

/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
 void MD5Hash(string* inputs, uint32x4_t* state1, uint32x4_t* state2) {
    Byte *paddedMessage[8];   // 用于存储 8 个输入口令的填充信息
    int *messageLength = new int[8]; // 每个输入口令的长度
    uint32x4_t x1[16], x2[16]; // 分别存储两路数据块的 SIMD 向量

    // 对 8 个输入口令进行填充
    for (int i = 0; i < 8; i++) {
        paddedMessage[i] = StringProcess(inputs[i], &messageLength[i]);
    }

    // 验证所有填充后的长度是否一致
    for (int i = 1; i < 8; i++) {
        assert(messageLength[i] == messageLength[0]);
    }

    int n_blocks = messageLength[0] / 64;

    // 初始化 MD5 的初始状态
    state1[0] = vdupq_n_u32(0x67452301);
    state1[1] = vdupq_n_u32(0xefcdab89);
    state1[2] = vdupq_n_u32(0x98badcfe);
    state1[3] = vdupq_n_u32(0x10325476);

    state2[0] = vdupq_n_u32(0x67452301);
    state2[1] = vdupq_n_u32(0xefcdab89);
    state2[2] = vdupq_n_u32(0x98badcfe);
    state2[3] = vdupq_n_u32(0x10325476);

    // 逐块更新 state
    for (int i = 0; i < n_blocks; i++) {
        // 加载每个数据块的值到 SIMD 寄存器中
        for (int j = 0; j < 16; j++) {
            uint32_t vals1[4], vals2[4];
            for (int k = 0; k < 4; k++) {
                // 第一组 4 个口令
                vals1[k] = static_cast<uint32_t>(
                    (paddedMessage[k][4 * j + i * 64]) |
                    (paddedMessage[k][4 * j + 1 + i * 64] << 8) |
                    (paddedMessage[k][4 * j + 2 + i * 64] << 16) |
                    (paddedMessage[k][4 * j + 3 + i * 64] << 24)
                );

                // 第二组 4 个口令
                vals2[k] = static_cast<uint32_t>(
                    (paddedMessage[k + 4][4 * j + i * 64]) |
                    (paddedMessage[k + 4][4 * j + 1 + i * 64] << 8) |
                    (paddedMessage[k + 4][4 * j + 2 + i * 64] << 16) |
                    (paddedMessage[k + 4][4 * j + 3 + i * 64] << 24)
                );
            }

            // 使用 vld1q_u32 直接加载值
            x1[j] = vld1q_u32(vals1);  // 加载四个值到 SIMD 向量
            x2[j] = vld1q_u32(vals2);  // 加载四个值到 SIMD 向量
        }

        // MD5 的四个状态变量
        uint32x4_t a1 = state1[0], b1 = state1[1], c1 = state1[2], d1 = state1[3];
        uint32x4_t a2 = state2[0], b2 = state2[1], c2 = state2[2], d2 = state2[3];

/*********** Round 1 for a1, b1, c1, d1 ***********/
FF(a1, b1, c1, d1, x1[ 0], s11, 0xd76aa478);
FF(d1, a1, b1, c1, x1[ 1], s12, 0xe8c7b756);
FF(c1, d1, a1, b1, x1[ 2], s13, 0x242070db);
FF(b1, c1, d1, a1, x1[ 3], s14, 0xc1bdceee);
FF(a1, b1, c1, d1, x1[ 4], s11, 0xf57c0faf);
FF(d1, a1, b1, c1, x1[ 5], s12, 0x4787c62a);
FF(c1, d1, a1, b1, x1[ 6], s13, 0xa8304613);
FF(b1, c1, d1, a1, x1[ 7], s14, 0xfd469501);
FF(a1, b1, c1, d1, x1[ 8], s11, 0x698098d8);
FF(d1, a1, b1, c1, x1[ 9], s12, 0x8b44f7af);
FF(c1, d1, a1, b1, x1[10], s13, 0xffff5bb1);
FF(b1, c1, d1, a1, x1[11], s14, 0x895cd7be);
FF(a1, b1, c1, d1, x1[12], s11, 0x6b901122);
FF(d1, a1, b1, c1, x1[13], s12, 0xfd987193);
FF(c1, d1, a1, b1, x1[14], s13, 0xa679438e);
FF(b1, c1, d1, a1, x1[15], s14, 0x49b40821);

/*********** Round 2 for a1, b1, c1, d1 ***********/
GG(a1, b1, c1, d1, x1[ 1], s21, 0xf61e2562);
GG(d1, a1, b1, c1, x1[ 6], s22, 0xc040b340);
GG(c1, d1, a1, b1, x1[11], s23, 0x265e5a51);
GG(b1, c1, d1, a1, x1[ 0], s24, 0xe9b6c7aa);
GG(a1, b1, c1, d1, x1[ 5], s21, 0xd62f105d);
GG(d1, a1, b1, c1, x1[10], s22, 0x02441453);
GG(c1, d1, a1, b1, x1[15], s23, 0xd8a1e681);
GG(b1, c1, d1, a1, x1[ 4], s24, 0xe7d3fbc8);
GG(a1, b1, c1, d1, x1[ 9], s21, 0x21e1cde6);
GG(d1, a1, b1, c1, x1[14], s22, 0xc33707d6);
GG(c1, d1, a1, b1, x1[ 3], s23, 0xf4d50d87);
GG(b1, c1, d1, a1, x1[ 8], s24, 0x455a14ed);
GG(a1, b1, c1, d1, x1[13], s21, 0xa9e3e905);
GG(d1, a1, b1, c1, x1[ 2], s22, 0xfcefa3f8);
GG(c1, d1, a1, b1, x1[ 7], s23, 0x676f02d9);
GG(b1, c1, d1, a1, x1[12], s24, 0x8d2a4c8a);

/*********** Round 3 for a1, b1, c1, d1 ***********/
HH(a1, b1, c1, d1, x1[ 5], s31, 0xfffa3942);
HH(d1, a1, b1, c1, x1[ 8], s32, 0x8771f681);
HH(c1, d1, a1, b1, x1[11], s33, 0x6d9d6122);
HH(b1, c1, d1, a1, x1[14], s34, 0xfde5380c);
HH(a1, b1, c1, d1, x1[ 1], s31, 0xa4beea44);
HH(d1, a1, b1, c1, x1[ 4], s32, 0x4bdecfa9);
HH(c1, d1, a1, b1, x1[ 7], s33, 0xf6bb4b60);
HH(b1, c1, d1, a1, x1[10], s34, 0xbebfbc70);
HH(a1, b1, c1, d1, x1[13], s31, 0x289b7ec6);
HH(d1, a1, b1, c1, x1[ 0], s32, 0xeaa127fa);
HH(c1, d1, a1, b1, x1[ 3], s33, 0xd4ef3085);
HH(b1, c1, d1, a1, x1[ 6], s34, 0x04881d05);
HH(a1, b1, c1, d1, x1[ 9], s31, 0xd9d4d039);
HH(d1, a1, b1, c1, x1[12], s32, 0xe6db99e5);
HH(c1, d1, a1, b1, x1[15], s33, 0x1fa27cf8);
HH(b1, c1, d1, a1, x1[ 2], s34, 0xc4ac5665);

/*********** Round 4 for a1, b1, c1, d1 ***********/
II(a1, b1, c1, d1, x1[ 0], s41, 0xf4292244);
II(d1, a1, b1, c1, x1[ 7], s42, 0x432aff97);
II(c1, d1, a1, b1, x1[14], s43, 0xab9423a7);
II(b1, c1, d1, a1, x1[ 5], s44, 0xfc93a039);
II(a1, b1, c1, d1, x1[12], s41, 0x655b59c3);
II(d1, a1, b1, c1, x1[ 3], s42, 0x8f0ccc92);
II(c1, d1, a1, b1, x1[10], s43, 0xffeff47d);
II(b1, c1, d1, a1, x1[ 1], s44, 0x85845dd1);
II(a1, b1, c1, d1, x1[ 8], s41, 0x6fa87e4f);
II(d1, a1, b1, c1, x1[15], s42, 0xfe2ce6e0);
II(c1, d1, a1, b1, x1[ 6], s43, 0xa3014314);
II(b1, c1, d1, a1, x1[13], s44, 0x4e0811a1);
II(a1, b1, c1, d1, x1[ 4], s41, 0xf7537e82);
II(d1, a1, b1, c1, x1[11], s42, 0xbd3af235);
II(c1, d1, a1, b1, x1[ 2], s43, 0x2ad7d2bb);
II(b1, c1, d1, a1, x1[ 9], s44, 0xeb86d391);

/*********** Round 1 for a2, b2, c2, d2 ***********/
FF(a2, b2, c2, d2, x2[ 0], s11, 0xd76aa478);
FF(d2, a2, b2, c2, x2[ 1], s12, 0xe8c7b756);
FF(c2, d2, a2, b2, x2[ 2], s13, 0x242070db);
FF(b2, c2, d2, a2, x2[ 3], s14, 0xc1bdceee);
FF(a2, b2, c2, d2, x2[ 4], s11, 0xf57c0faf);
FF(d2, a2, b2, c2, x2[ 5], s12, 0x4787c62a);
FF(c2, d2, a2, b2, x2[ 6], s13, 0xa8304613);
FF(b2, c2, d2, a2, x2[ 7], s14, 0xfd469501);
FF(a2, b2, c2, d2, x2[ 8], s11, 0x698098d8);
FF(d2, a2, b2, c2, x2[ 9], s12, 0x8b44f7af);
FF(c2, d2, a2, b2, x2[10], s13, 0xffff5bb1);
FF(b2, c2, d2, a2, x2[11], s14, 0x895cd7be);
FF(a2, b2, c2, d2, x2[12], s11, 0x6b901122);
FF(d2, a2, b2, c2, x2[13], s12, 0xfd987193);
FF(c2, d2, a2, b2, x2[14], s13, 0xa679438e);
FF(b2, c2, d2, a2, x2[15], s14, 0x49b40821);

/*********** Round 2 for a2, b2, c2, d2 ***********/
GG(a2, b2, c2, d2, x2[ 1], s21, 0xf61e2562);
GG(d2, a2, b2, c2, x2[ 6], s22, 0xc040b340);
GG(c2, d2, a2, b2, x2[11], s23, 0x265e5a51);
GG(b2, c2, d2, a2, x2[ 0], s24, 0xe9b6c7aa);
GG(a2, b2, c2, d2, x2[ 5], s21, 0xd62f105d);
GG(d2, a2, b2, c2, x2[10], s22, 0x02441453);
GG(c2, d2, a2, b2, x2[15], s23, 0xd8a1e681);
GG(b2, c2, d2, a2, x2[ 4], s24, 0xe7d3fbc8);
GG(a2, b2, c2, d2, x2[ 9], s21, 0x21e1cde6);
GG(d2, a2, b2, c2, x2[14], s22, 0xc33707d6);
GG(c2, d2, a2, b2, x2[ 3], s23, 0xf4d50d87);
GG(b2, c2, d2, a2, x2[ 8], s24, 0x455a14ed);
GG(a2, b2, c2, d2, x2[13], s21, 0xa9e3e905);
GG(d2, a2, b2, c2, x2[ 2], s22, 0xfcefa3f8);
GG(c2, d2, a2, b2, x2[ 7], s23, 0x676f02d9);
GG(b2, c2, d2, a2, x2[12], s24, 0x8d2a4c8a);

/*********** Round 3 for a2, b2, c2, d2 ***********/
HH(a2, b2, c2, d2, x2[ 5], s31, 0xfffa3942);
HH(d2, a2, b2, c2, x2[ 8], s32, 0x8771f681);
HH(c2, d2, a2, b2, x2[11], s33, 0x6d9d6122);
HH(b2, c2, d2, a2, x2[14], s34, 0xfde5380c);
HH(a2, b2, c2, d2, x2[ 1], s31, 0xa4beea44);
HH(d2, a2, b2, c2, x2[ 4], s32, 0x4bdecfa9);
HH(c2, d2, a2, b2, x2[ 7], s33, 0xf6bb4b60);
HH(b2, c2, d2, a2, x2[10], s34, 0xbebfbc70);
HH(a2, b2, c2, d2, x2[13], s31, 0x289b7ec6);
HH(d2, a2, b2, c2, x2[ 0], s32, 0xeaa127fa);
HH(c2, d2, a2, b2, x2[ 3], s33, 0xd4ef3085);
HH(b2, c2, d2, a2, x2[ 6], s34, 0x04881d05);
HH(a2, b2, c2, d2, x2[ 9], s31, 0xd9d4d039);
HH(d2, a2, b2, c2, x2[12], s32, 0xe6db99e5);
HH(c2, d2, a2, b2, x2[15], s33, 0x1fa27cf8);
HH(b2, c2, d2, a2, x2[ 2], s34, 0xc4ac5665);

/*********** Round 4 for a2, b2, c2, d2 ***********/
II(a2, b2, c2, d2, x2[ 0], s41, 0xf4292244);
II(d2, a2, b2, c2, x2[ 7], s42, 0x432aff97);
II(c2, d2, a2, b2, x2[14], s43, 0xab9423a7);
II(b2, c2, d2, a2, x2[ 5], s44, 0xfc93a039);
II(a2, b2, c2, d2, x2[12], s41, 0x655b59c3);
II(d2, a2, b2, c2, x2[ 3], s42, 0x8f0ccc92);
II(c2, d2, a2, b2, x2[10], s43, 0xffeff47d);
II(b2, c2, d2, a2, x2[ 1], s44, 0x85845dd1);
II(a2, b2, c2, d2, x2[ 8], s41, 0x6fa87e4f);
II(d2, a2, b2, c2, x2[15], s42, 0xfe2ce6e0);
II(c2, d2, a2, b2, x2[ 6], s43, 0xa3014314);
II(b2, c2, d2, a2, x2[13], s44, 0x4e0811a1);
II(a2, b2, c2, d2, x2[ 4], s41, 0xf7537e82);
II(d2, a2, b2, c2, x2[11], s42, 0xbd3af235);
II(c2, d2, a2, b2, x2[ 2], s43, 0x2ad7d2bb);
II(b2, c2, d2, a2, x2[ 9], s44, 0xeb86d391);

        // 更新 state
        state1[0] = vaddq_u32(state1[0], a1);
        state1[1] = vaddq_u32(state1[1], b1);
        state1[2] = vaddq_u32(state1[2], c1);
        state1[3] = vaddq_u32(state1[3], d1);

        state2[0] = vaddq_u32(state2[0], a2);
        state2[1] = vaddq_u32(state2[1], b2);
        state2[2] = vaddq_u32(state2[2], c2);
        state2[3] = vaddq_u32(state2[3], d2);
    }

    // 释放动态分配的内存
    for (int i = 0; i < 8; i++) {
        delete[] paddedMessage[i];
    }
    delete[] messageLength;
}
