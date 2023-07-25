#ifndef CLIONPROJECT_PROCESSOR_H
#define CLIONPROJECT_PROCESSOR_H

#include "cailib.h"

namespace cai
{
    class Processor{
    private:
        static std::map<intptr_t , int> rfcnt;
        static int cnt;

    public:
        template<class T>
        static int getCnt(T* adrs){
            intptr_t num = (intptr_t)adrs;
            if(rfcnt.find(num) == rfcnt.end()) return 0;
            else return rfcnt[num];
        }

        template<class T>
        static void ref(T* adrs){
            intptr_t num = (intptr_t)adrs;
            if(rfcnt.find(num) == rfcnt.end())  {
                rfcnt[num] = 1;
                cnt++;
            }
            else rfcnt[num] ++;
        }

        template<class T>
        static void unref(T* adrs){
            intptr_t num = (intptr_t)adrs;
            if(rfcnt.find(num) == rfcnt.end()){
                throw std::domain_error("there is no data which pointer is " + std::to_string(num));
            }
            else if(rfcnt[num] == 1){
                delete[] adrs;
                cnt--;
                rfcnt.erase(num);
                //std::cout << cnt << " storage left" << std::endl;
            }
            else {
                rfcnt[num] --;
            }
        }
    };

    std::map<intptr_t, int> Processor::rfcnt = std::map<intptr_t, int>();
    int Processor::cnt = 0;
}

#endif //CLIONPROJECT_PROCESSOR_H
