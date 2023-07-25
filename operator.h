#ifndef CLIONPROJECT_OPERATOR_H
#define CLIONPROJECT_OPERATOR_H
#include "cailib.h"

namespace cai::Operation
{
    template<typename T, typename... Ts>
    class Operator{
        virtual Tensor<T> forward(Ts... v);
        virtual std::tuple<Ts...> backward(Tensor<T> l);
    };

    template<typename T>
    class Add : Operator<T, Tensor<T>, Tensor<T>>{
        Tensor<T> t1, t2;
        Tensor<T> forward(Tensor<T> t1, Tensor<T> t2){
            this->t1 = t1;
            this->t2 = t2;
            return t1 + t2;
        }
        std::tuple<Tensor<T>, Tensor<T>> backward(Tensor<T> l){
            t1.backward(l);
            t2.backward(l);
        }
    };
}

#endif //CLIONPROJECT_OPERATOR_H
