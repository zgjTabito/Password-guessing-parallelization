// #include "PCFG.h"
// #include <chrono>
// #include <fstream>
// #include "md5.h"
// #include <iomanip>
// using namespace std;
// using namespace chrono;

// // 编译指令如下：
// // g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// // 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
// int main()
// {
//     bit32 state[4];
//     MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", state);
//     for (int i1 = 0; i1 < 4; i1 += 1)
//     {
//         cout << std::setw(8) << std::setfill('0') << hex << state[i1];
//     }
//     cout << endl;
// }

#include "md5.h"
#include <iomanip>
#include <iostream>
using namespace std;


int main()
{
    uint32x4_t* state; // 使用 NEON 的 uint32x4_t 类型
    string inputs[4];
    inputs[0]="bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    inputs[1]="bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvaaa";
    inputs[2]="";
    inputs[3]="";
    MD5Hash(inputs, state);

    // 输出 MD5 哈希结果
    
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i][0]; // 提取 NEON 向量的第一个元素
        cout << std::setw(8) << std::setfill('0') << hex << value;
    }
    cout << endl;

    return 0;
}