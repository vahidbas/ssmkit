#pragma once

#include <mlpack/core.hpp>

namespace PROJECT_NAME{
namespace distribution{
    using Gaussian = mlpack::distribution::GaussianDistribution;

    template <typename MEAN_FUNC, typename COV_FUNC>
    class ConditionalGaussian {
        using RV_TYPE = decltype(dist.Random());
        using MARGINAL_TYPE = Gaussian;

        ConditionalGaussian(MEAN_FUNC &mf, COV_FUNC &cf):
            mean_func(mf), cov_func(cf) {}

        auto Random(SomeType cnd){
            dist.Mean() = std::move(mean_func(cnd));
            dist.Covariance(cov_func(cnd));
            return dist.Random();
        }

        private:
        Gaussian dist;
        MEAN_FUNC mean_func;
        COV_FUNC cov_func;
    };
} // namespace ditribution
} // namespace PROJECT_NAME
