#pragma once
#include <type_traits>
#include <mlpack/core.hpp>
#include "external/callable/callable.hpp"

namespace PROJECT_NAME{
namespace distribution{
    using Gaussian = mlpack::distribution::GaussianDistribution;

    template <typename MEAN_FUNC, typename COV_FUNC>
    class ConditionalGaussian {
        using MARGINAL_TYPE = Gaussian;
        using CV_TYPE = typename callable_traits<MEAN_FUNC>::template argument_type<0>;
        using RV_TYPE = typename std::result_of<decltype(&Gaussian::Random)(Gaussian)>::type;
        
        public:
        ConditionalGaussian(MEAN_FUNC &&mf, COV_FUNC &&cf):
            mean_func(mf), cov_func(cf) {}


        RV_TYPE Random(CV_TYPE cnd){
            dist.Mean() = std::move(mean_func(cnd));
            dist.Covariance(cov_func(cnd));
            return dist.Random();
        }

        private:
        Gaussian dist;
        MEAN_FUNC mean_func;
        COV_FUNC cov_func;
    };

    
   template<typename F1, typename F2>
   ConditionalGaussian<F1,F2> makeConditionalGaussian(F1 &&f1, F2 &&f2)
   { return ConditionalGaussian<F1,F2>(std::forward<F1>(f1), std::forward<F2>(f2)); }
} // namespace ditribution
} // namespace PROJECT_NAME
