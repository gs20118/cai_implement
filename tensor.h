#ifndef CLIONPROJECT_TENSOR_H
#define CLIONPROJECT_TENSOR_H

#include "cailib.h"
#include "processor.h"
#include "range.h"
#include "operator.h"

namespace cai
{
    template<typename T>
    class Tensor
    {
    private:
        T* data = nullptr;
        T * grad = nullptr;

        int dim;
        int offset;
        int *stride;
        int *shape;
        bool conjugate = true;
        bool required_grad = false;

    public:
        bool indexed = false;
        Operation::Operator<T> * oper = nullptr;

        //생성자들
        Tensor();
        ~Tensor();
        Tensor(T v);
        Tensor(std::initializer_list<int> sh, T val = 0);
        Tensor(std::initializer_list<T> val, std::initializer_list<int> sh);
        Tensor(std::vector<int> sh, T val = 0);
        Tensor(std::vector<T> val, std::vector<int> sh);
        Tensor(int dim, int *stride, int *shape, T val =0);
        Tensor(int dim, int *stride, int *shape, T* data, int offset);
        Tensor(const Tensor& other);

        //set, get
        void set_data(T *data);
        void set_conj();
        void set(Tensor &o);
        void set(T v);
        void set_grad();

        int get_size();

        T& item();
        T& item(std::vector<int>& pos);
        template<typename... Ints>
        T& item(Ints... b);

        Tensor get(std::vector<cai::Range>& pos);
        template<typename... Ranges>
        Tensor get(Ranges... b);

        template<typename... Ranges>
        Tensor operator[](Ranges... b);

        void nexti(std::vector<int>& v);
        std::vector<int> initi();

        template<typename... Ints>
        Tensor reshape(Ints... b);
        Tensor reshape(std::vector<int> resh);

        Tensor copy();

        int index(std::vector<int>& pos);
        template<typename... Ints>
        int index(Ints... b);

        //출력
        void print();
        void print_all();
        void print(std::ostream& o);
        std::string toString();
        std::string toString_(std::vector<int> pos);


        Tensor& operator=(Tensor &o);
        Tensor& operator=(T v);
        Tensor& operator+(Tensor &o);
        Tensor& operator+(T v);
        Tensor& operator+=(Tensor &o);
        Tensor& operator+=(T v);
        bool sameShape(Tensor &o);
    };

}

/**** 사용할 함수 ****/



namespace func{
    template<typename T>
    T* newArray(int n, T *arr)
    {
        T* ret = new T[n];
        for(int i=0; i<n; i++) ret[i] = arr[i];
        return ret;
    }

    template<typename T>
    T* newArray(int n, T val)
    {
        T* ret = new T[n];
        for(int i=0; i<n; i++) ret[i] = val;
        return ret;
    }

    template<typename T>
    std::vector<T> vec_(T a){
        std::vector<T> ret = std::vector<T>();
        ret.push_back(a);
        return ret;
    }

    template<typename T, typename... Ts>
    std::vector<T> vec_(T a, Ts... b){
        std::vector<T> ret = vec_<T>(b...);
        ret.push_back(a);
        return ret;
    }

    template<typename T, typename... Ts>
    std::vector<T> vec(T a, Ts... b){
        std::vector<T> ret = vec_<T>(a, b...);
        std::reverse(ret.begin(), ret.end());
        return ret;
    }
}



//==========================================================================

/**** Tensor 생성자/소멸자 정의 ****/



//dim = 0 용
template<typename T>
cai::Tensor<T>::Tensor() {
    this->dim = 0;
    set_data(func::newArray<T>(dim, 0));
    stride = nullptr;
    shape = nullptr;
    offset = 0;
    conjugate = true;
}

template<typename T>
cai::Tensor<T>::~Tensor() {
    delete[] stride;
    delete[] shape;
    cai::Processor::unref<T>(this->data);
}

template<typename T>
cai::Tensor<T>::Tensor(T v)
:Tensor(){
    data[0] = v;
}



//사용자 용
template<typename T>
cai::Tensor<T>::Tensor(std::initializer_list<int> sh, T val)
:Tensor(std::vector<int>(sh.begin(), sh.end()), val) {}

template<typename T>
cai::Tensor<T>::Tensor(std::initializer_list<T> val, std::initializer_list<int> sh)
:Tensor(std::vector<T>(val.begin(), val.end()), std::vector<int>(sh.begin(), sh.end())) {}



//내부 구현 용


template<typename T>
cai::Tensor<T>::Tensor(int dim, int *stride, int *shape, T* data, int offset)
{
    this->dim = dim;
    this->stride = stride;
    this->shape = shape;
    set_conj();
    set_data(data);
    this->offset = offset;
}

template<typename T>
cai::Tensor<T>::Tensor(int dim, int *stride, int *shape, T val){
    this->dim = dim;
    this->offset = 0;
    this->stride = stride;
    this->shape = shape;
    conjugate = true;
    set_data(func::newArray<T>(get_size(), val));
}

template<typename T>
cai::Tensor<T>::Tensor(const Tensor &o)
:Tensor(o.dim,func::newArray(o.dim, o.stride),
        func::newArray(o.dim, o.shape),o.data, o.offset){}

template<typename T>
cai::Tensor<T>::Tensor(std::vector<int> sh, T val)
{
    int cnt = 0, num;
    dim  = sh.size();
    shape = new int[dim];
    stride = new int[dim];
    offset = 0;

    for(int & x : sh){
        shape[cnt] = x;
        cnt++;
    }

    stride[dim-1] = 1;
    for(int i=dim-2; i>=0; i--)
        stride[i] = stride[i+1] * shape[i+1];

    set_data(func::newArray(get_size(), val));
    conjugate = true;
}

template<typename T>
cai::Tensor<T>::Tensor(std::vector<T> val, std::vector<int> sh)
:Tensor(sh)
{
    int cnt = 0, num = get_size();
    if(num != val.size()){
        throw std::range_error( "you can't make " + std::to_string(sh.size()) + " to " + std::to_string(num) +  "\n");
    }
    for(T & x : val){
        data[cnt] = x;
        cnt++;
    }
}



//==========================================================================

/**** Tensor Operation 정의 ****/


template<typename T>
cai::Tensor<T>& cai::Tensor<T>::operator=(cai::Tensor<T> &o) {
    if(!indexed){
        dim = o.dim;
        shape = func::newArray(dim, o.shape);
        stride = func::newArray(dim, o.stride);
        offset = o.offset;
        conjugate = o.conjugate;
        set_data(o.data);
    }
    else{
        set(o);
    }
    return *this;
}

template<typename T>
cai::Tensor<T>& cai::Tensor<T>::operator=(T val) {
    if(indexed == false){
        this->dim = 0;
        set_data(func::newArray<T>(1, val));
        stride = nullptr;
        shape = nullptr;
        offset = 0;
        conjugate = true;
    }
    else{
        set(val);
    }
    return *this;
}

template<typename T>
bool cai::Tensor<T>::sameShape(Tensor<T> &o) {
    if( dim!=o.dim ) return false;
    for(int i=0; i<dim; i++) if(shape[i]!=o.shape[i]) return false;
    return true;
}



//==========================================================================

/**** Tensor setter, getter 정의 ****/




//set
template<typename T>
void cai::Tensor<T>::set_data(T* data){
    if(this->data != nullptr) cai::Processor::unref(this->data);
    this->data = data;
    cai::Processor::ref(this->data);
}

template<typename T>
void cai::Tensor<T>::set_conj() {
    int chk = 1;
    for (int i = dim-1; i>=0; i--)
    {
        if(chk != stride[i]) {
            conjugate = false;
            return;
        }
        chk *= shape[i];
    }
}

template<typename T>
void cai::Tensor<T>::set(Tensor<T> &o)
{
    if(!sameShape(o)) throw std::length_error("the shape is not equal");
    std::vector<int> v = initi();
    int num = get_size();
    for(int i=0; i<num; i++) {
        nexti(v);
        item(v) = o.item(v);
    }
}

template<typename T>
void cai::Tensor<T>::set(T val)
{
    std::vector<int> v = initi();
    int num = get_size();
    for(int i=0; i<num; i++) {
        nexti(v);
        item(v) = val;
    }
}



template<typename  T>
int cai::Tensor<T>::get_size() {
    int ret = 1;
    for(int i=0; i<dim;i++)
        ret *= shape[i];
    return ret;
}


//item
template<typename T>
T& cai::Tensor<T>::item(){
    if(dim!=0){
        throw std::domain_error("Can't get the item if dimension isn't 0");
    }
    return data[0];
}

template<typename T>
T& cai::Tensor<T>::item(std::vector<int>& pos){
    return data[index(pos)];
}

template<typename T>
template<typename... Ints>
T& cai::Tensor<T>::item(Ints... b){
    auto a = func::vec(b...);
    return item(a);
}



//index
template<typename T>
int cai::Tensor<T>::index(std::vector<int>& pos){
    int idx = offset;
    if(pos.size() != dim){
        throw std::domain_error("there isn't index for indicator which dimension isn't equal to tensor\n"
                                + std::to_string(pos.size()) + " is not equal to " + std::to_string(dim) + "\n");
    }
    for(int i=0; i<dim; i++){
        if(pos[i]>=shape[i]){
            throw std::range_error("index over shape in dim:" + std::to_string(dim) + "\n");
        }
        idx += pos[i] * stride[i];
    }
    return idx;
}

template<typename T>
template<typename... Ints>
int cai::Tensor<T>::index(Ints... b){
    auto &a = func::vec(b...);
    return index(a);
}

template<typename T>
void cai::Tensor<T>::nexti(std::vector<int>& v){
    if(dim>=1) v[dim-1] ++;
    int n = dim-1;
    while(n>=1){
        if(v[n] < shape[n]) break;
        v[n] = 0 ;
        v[n-1] ++;
        n--;
    }
}

template<typename T>
std::vector<int> cai::Tensor<T>::initi(){
    std::vector<int> ret = std::vector<int>(dim);
    if(dim>=1) ret[dim-1] = -1;
    return ret;
}


//get
template<typename T>
cai::Tensor<T> cai::Tensor<T>::get(std::vector<Range>& pos){
    if(pos.size() > dim){
        throw std::range_error("index is over " + std::to_string(dim) + "\n");
    }

    int newDim = dim;
    for(Range & p : pos){
        if(p.v) newDim--;
    }

    int *newShape = new int[newDim];
    int *newStride = new int[newDim];
    int newOffset = offset;
    int cnt = 0;

    for(int i=0; i<dim; i++){
        if(i<pos.size()){
            newOffset += pos[i].s * stride[i];
            if(!pos[i].v){
                if(pos[i].e == -1) pos[i].e = shape[i];
                newShape[cnt] = pos[i].e - pos[i].s;
                newStride[cnt] = stride[i];
                cnt++;
            }
        }
        else{
            newShape[cnt] = shape[i];
            newStride[cnt] = stride[i];
            cnt++;
        }
    }

    Tensor ret = Tensor<T>(newDim, newStride, newShape, data, newOffset);
    return ret;
}

template<typename T>
template<typename... Ranges>
cai::Tensor<T> cai::Tensor<T>::get(Ranges... b){
    auto a = func::vec<cai::Range>(b...);
    return get(a);
}

template<typename T>
template<typename... Ranges>
cai::Tensor<T> cai::Tensor<T>::operator[](Ranges... b) {
    cai::Tensor<T> ret =  get(b...);
    ret.indexed = true;
    return ret;
}

//set
template<typename T>
void cai::Tensor<T>::set_grad(){
    required_grad = true;
    grad = new T[get_size()];
}

//reshape
template<typename T>
template<typename... Ints>
cai::Tensor<T> cai::Tensor<T>::reshape(Ints... b){
    auto & a = func::vec(b...);
    return reshape(a);
}

template<typename T>
cai::Tensor<T> cai::Tensor<T>::reshape(std::vector<int> resh){
    if(conjugate == false){
        throw std::domain_error( "you can't reshape, the tensor is not conjugate \n");
    }

    int num = 1;
    for(int &x : resh) num *= x;

    if(get_size() != num){
        throw std::range_error( "you can't reshape, the number of item is different \n");
    }

    int newDim = resh.size();
    int* newShape = resh.data();
    int* newStride = new int[newDim];
    int cnt = 1;
    for(int i=newDim-1; i>=0; i--){
        if(i != newDim-1) cnt *= newShape[i+1];
        newStride[i] = cnt;
    }

    Tensor<T> ret = Tensor(newDim , newShape, newStride);
    ret.set_data(data);
    ret.offset = 0;
    return ret;
}

template<typename T>
cai::Tensor<T> cai::Tensor<T>::copy(){
    int newDim = dim;
    int* newShape = func::newArray(dim, shape);
    int* newStride = new int[newDim];
    int cnt = 1;
    for(int i=newDim-1; i>=0; i--){
        newStride[i] = cnt;
        cnt *= newShape[i];
    }

    T* newData = new T[cnt];
    std::vector<int> pos = initi();
    for(int i=0; i<cnt; i++){
        nexti(pos);
        newData[i] = item(pos);
    }
    Tensor<T> ret = Tensor(newDim , newStride, newShape,newData, 0);
    return ret;
}



//==========================================================================

/**** Tensor 출력 정의 ****/



template<typename T>
std::ostream& operator<<(std::ostream& o, cai::Tensor<T>& t){
    t.print(o);
    return o;
}

template<typename T>
void cai::Tensor<T>::print(){
    print(std::cout);
}

template<typename T>
void cai::Tensor<T>::print_all(){
    std::cout << "dim : " <<  dim << std::endl;
    std::cout << "shape : (" ;
    for(int i=0; i<dim; i++){
        std::cout<< shape[i] ;
        if(i!=dim -1) std:: cout<< ", ";
        else std::cout<<")\n";
    }
    std::cout << "stride : (" ;
    for(int i=0; i<dim; i++){
        std::cout<< stride[i] ;
        if(i!=dim -1) std:: cout<< ", ";
        else std::cout<<")\n";
    }
    std::cout << "offset : " << offset << std::endl;
}

template<typename T>
void cai::Tensor<T>::print(std::ostream &o) {
    o<<toString();
}

template<typename T>
std::string cai::Tensor<T>::toString_(std::vector<int> pos) {
    if(pos.size() == dim){
        return std::to_string(item(pos));
    }

    std::string ret = "[";
    int ldim = dim - pos.size();
    for(int i=0; i<shape[pos.size()]; i++){
        std::vector<int> temp(pos);
        temp.push_back(i);
        ret += toString_(temp);
        if( i!= shape[pos.size()]-1) {
            ret += ", ";
            if (ldim >= 2) ret += "\n";
            if (ldim >= 3) ret += "\n";
        }
    }
    ret += "]";

    return ret;
}

template<typename T>
std::string cai::Tensor<T>::toString() {
    std::vector<int> temp;
    return toString_(temp);
}



//==========================================================================

template class cai::Tensor<int>;
template class cai::Tensor<float>;
template class cai::Tensor<double>;
#endif
