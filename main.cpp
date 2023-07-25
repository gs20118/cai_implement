#include "cai.h"
#include <ctime>

int main() {
    clock_t start, finish;
    double duration;
    start = clock();

    for(int ITER = 0; ITER<100; ITER++)
    {
        cai::Tensor<int> t = cai::Tensor<int>({10, 1000, 1000});
        for(int i=0; i<10; i++){
            cai::Tensor<int> nt = cai::Tensor<int>({1000, 1000}, i);
            for(int j=0; j<1000; j++){
                nt.item(j, j) = 0;
            }

            cai::Tensor<int> temp = nt.copy();
            t[i] = temp;
        }
    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << duration << "s" << std::endl;
    return 0;
}
