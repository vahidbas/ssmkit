#include <boost/test/unit_test.hpp>

#include "filter/kalman.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(kalman_filter_test);

BOOST_AUTO_TEST_CASE(filter_test)
{

    auto F = arma::eye<arma::mat>(2,2);
    auto H = arma::eye<arma::mat>(1,2);
    auto B = arma::ones<arma::mat>(2,1);

    mlpack::distribution::GaussianDistribution w(arma::zeros<arma::vec>(2), arma::eye<arma::mat>(2,2));
    mlpack::distribution::GaussianDistribution v(arma::zeros<arma::vec>(1), arma::eye<arma::mat>(1,1));

    filter::Kalman kf(F,H,w,v); 

    mlpack::distribution::GaussianDistribution x0(arma::zeros<arma::vec>(2), arma::eye<arma::mat>(2,2));

    kf.initialize(x0);
    kf.predict();
    arma::vec obs {1};
    kf.filter(obs);
}

BOOST_AUTO_TEST_SUITE_END();
