#include <cstring>
#include <emmintrin.h> // SSE2
#include <iostream>
#include <string>
#include <tmmintrin.h> // SSSE3 for _mm_shuffle_epi8


using namespace std;



// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
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

// SIMD 并行化的 MD5 基本函数，改为 AVX 256 位指令
#define F(x, y, z)                                                             \
  _mm256_or_si256(_mm256_and_si256(x, y),                                      \
                  _mm256_and_si256(_mm256_xor_si256(x, _mm256_set1_epi32(-1)), z))
#define G(x, y, z)                                                             \
  _mm256_or_si256(_mm256_and_si256(x, z),                                      \
                  _mm256_and_si256(y, _mm256_xor_si256(z, _mm256_set1_epi32(-1))))
#define H(x, y, z) _mm256_xor_si256(_mm256_xor_si256(x, y), z)
#define I(x, y, z)                                                             \
  _mm256_xor_si256(y, _mm256_or_si256(x, _mm256_xor_si256(z, _mm256_set1_epi32(-1))))

#define VROTATE_LEFT_U32(data, shift)                                          \
  _mm256_or_si256(_mm256_slli_epi32(data, shift), _mm256_srli_epi32(data, 32 - shift))

// MD5 的四个主要操作函数，改为支持八路并行
#define FF(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    __m256i Fv = F(b, c, d);                                                   \
    a = _mm256_add_epi32(a, Fv);                                               \
    a = _mm256_add_epi32(a, x);                                                \
    a = _mm256_add_epi32(a, _mm256_set1_epi32(ac));                            \
    a = VROTATE_LEFT_U32(a, s);                                                \
    a = _mm256_add_epi32(a, b);                                                \
  }

#define GG(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    __m256i Gv = G(b, c, d);                                                   \
    a = _mm256_add_epi32(a, Gv);                                               \
    a = _mm256_add_epi32(a, x);                                                \
    a = _mm256_add_epi32(a, _mm256_set1_epi32(ac));                            \
    a = VROTATE_LEFT_U32(a, s);                                                \
    a = _mm256_add_epi32(a, b);                                                \
  }

#define HH(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    __m256i Hv = H(b, c, d);                                                   \
    a = _mm256_add_epi32(a, Hv);                                               \
    a = _mm256_add_epi32(a, x);                                                \
    a = _mm256_add_epi32(a, _mm256_set1_epi32(ac));                            \
    a = VROTATE_LEFT_U32(a, s);                                                \
    a = _mm256_add_epi32(a, b);                                                \
  }

#define II(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    __m256i Iv = I(b, c, d);                                                   \
    a = _mm256_add_epi32(a, Iv);                                               \
    a = _mm256_add_epi32(a, x);                                                \
    a = _mm256_add_epi32(a, _mm256_set1_epi32(ac));                            \
    a = VROTATE_LEFT_U32(a, s);                                                \
    a = _mm256_add_epi32(a, b);                                                \
  }

void MD5Hash(string *inputs, __m256i *state);