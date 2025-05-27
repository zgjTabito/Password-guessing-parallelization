#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
 string input = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    
    string inputs[4] = { input, "", "", "" }; // 只测试第一个口令

    __m128i state[4]; // SIMD hash 输出
    MD5Hash(inputs, state); // 调用并行 hash 函数

    for (int i = 0; i < 4; ++i) {
        uint32_t val = _mm_cvtsi128_si32(state[i]); // 提取最低的 32 位（即前 32 位）
        cout << setw(8) << setfill('0') << hex << val;
    }
    cout << endl;
    
}