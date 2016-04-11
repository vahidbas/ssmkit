#pragma once
#include<armadillo>

namespace PROJECT_NAME {
namespace model {
    template<size_t N>
    struct LinearGaussian {
        using PARAM_TYPE =  std::tuple<arma::vec::fixed<N>, arma::mat::fixed<N,N>>;
        using CV_TYPE = arma::vec::fixed<N>;
        
        PARAM_TYPE operator()(CV_TYPE & x)
        {
            return std::make_tuple(dynamic*x, covariance);
        }

        arma::mat::fixed<N,N> dynamic;
        arma::mat::fixed<N,N> covariance;
    };
} // namespace model
} // namespace PROJECT_NAME
