#pragma once
//#include <mlpack/core.hpp>
#include <vector>
#include <algorithm>

namespace PROJECT_NAME {
namespace simulation {
    template<typename DYNAMIC_CPDF>
    class Markov {
        public:
        Markov(DYNAMIC_CPDF dynamic_m = DYNAMIC_CPDF())
           : dyn_m(dynamic_m), ini_m(initial_m) {}
        
        auto Initialize(DYNAMIC_CPDF::MARGINAL_TYPE init_m){
            state = init_m.Random();
        }

        auto Step(){
            state = dyn_m.Random(state);
            return state;
        }

        auto Step(size_t N){
            std::vector<DYNAMIC_CPDF::RV_TYPE> traj;
            std::generate_n(traj, N, [](){return step();});
            return traj;
        }

        private:
        DYNAMIC_CPDF dyn_m;
        DYNAMIC_CPDF::RV_TYPE state;
    };
} // simulation
} // PROJECT_NAME


