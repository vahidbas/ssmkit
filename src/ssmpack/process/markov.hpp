#pragma once
//#include <mlpack/core.hpp>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>

namespace PROJECT_NAME {
namespace simulation {
    template<typename CPDF>
    class Markov {

        using URCPDF = typename std::remove_reference<CPDF>::type;
        static_assert(std::is_same<typename URCPDF::RV_TYPE, typename URCPDF::CV_TYPE>::value,
        "condition and random variables of cpdf should be same for first order Markov process");

        public:

        // dependent types
        using STATE_TYPE = typename URCPDF::RV_TYPE;

        Markov(URCPDF cpdf = URCPDF()) : cpdf_(cpdf) {}

        template<typename INIT_PDF>
        STATE_TYPE Initialize(INIT_PDF init_pdf)
        {
            auto init_par = init_pdf.ParticleSample();
            state_ = init_par.point;
            log_likelihood_ = std::log(init_par.weight);
            return state_;
        }

        STATE_TYPE Random()
        {
            auto par = cpdf_.ParticleSample(state_);
            state_ = par.point;
            log_likelihood_ += std::log(par.weight);
            return state_;
        }

        template<typename INIT_PDF>
        double Random(std::vector<STATE_TYPE>& samples, INIT_PDF init_pdf, size_t n)
        {
            static_assert(std::is_same<typename INIT_PDF::RV_TYPE, STATE_TYPE>::value,
            "random variable of initial pdf is different from the state variable") ;

            samples.clear();
            samples.resize(n+1);
            
            samples.push_back(Initialize(init_pdf));
            std::generate_n(samples.begin()+1, n, [this](){return Random();});
            return log_likelihood_;
        }

        private:
        CPDF cpdf_;
        STATE_TYPE state_;
        double log_likelihood_;
    };

    template<typename T>
    Markov<T> makeMarkov(T &&cpdf)
    {
        return Markov<T>(std::forward<T>(cpdf));
    }
} // simulation
} // PROJECT_NAME


