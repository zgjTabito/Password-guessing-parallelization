#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>

using namespace std;

typedef unsigned char Byte;
typedef unsigned int bit32;

// MD5的一系列参数
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// SIMD并行版的基本运算（全部用uint32x2_t）
#define F(x, y, z) vorr_u32(vand_u32(x, y), vand_u32(vmvn_u32(x), z))
#define G(x, y, z) vorr_u32(vand_u32(x, z), vand_u32(y, vmvn_u32(z)))
#define H(x, y, z) veor_u32(veor_u32(x, y), z)
#define I(x, y, z) veor_u32(y, vorr_u32(x, vmvn_u32(z)))

// 左循环位移（2元素向量版）
#define VROTATE_LEFT_U32(data, shift) \
    vorr_u32(vshl_n_u32(data, shift), vshr_n_u32(data, 32 - shift))

// FF operation
#define FF(a, b, c, d, x, s, ac) { \
    uint32x2_t tmp = F(b, c, d); \
    a = vadd_u32(a, tmp); \
    a = vadd_u32(a, x); \
    a = vadd_u32(a, vdup_n_u32(ac)); \
    a = VROTATE_LEFT_U32(a, s); \
    a = vadd_u32(a, b); \
}

// GG operation
#define GG(a, b, c, d, x, s, ac) { \
    uint32x2_t tmp = G(b, c, d); \
    a = vadd_u32(a, tmp); \
    a = vadd_u32(a, x); \
    a = vadd_u32(a, vdup_n_u32(ac)); \
    a = VROTATE_LEFT_U32(a, s); \
    a = vadd_u32(a, b); \
}

// HH operation
#define HH(a, b, c, d, x, s, ac) { \
    uint32x2_t tmp = H(b, c, d); \
    a = vadd_u32(a, tmp); \
    a = vadd_u32(a, x); \
    a = vadd_u32(a, vdup_n_u32(ac)); \
    a = VROTATE_LEFT_U32(a, s); \
    a = vadd_u32(a, b); \
}

// II operation
#define II(a, b, c, d, x, s, ac) { \
    uint32x2_t tmp = I(b, c, d); \
    a = vadd_u32(a, tmp); \
    a = vadd_u32(a, x); \
    a = vadd_u32(a, vdup_n_u32(ac)); \
    a = VROTATE_LEFT_U32(a, s); \
    a = vadd_u32(a, b); \
}

// 现在的函数声明，也要改成
void MD5Hash(string* inputs, uint32x2_t* state);
