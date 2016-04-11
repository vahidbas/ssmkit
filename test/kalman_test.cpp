#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/filter/kalman.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(kalman_filter_test);

BOOST_AUTO_TEST_CASE(one_step_test)
{
    constexpr double diff_tol = 0.0001;

    arma::mat F {{1, 0}, {0, 1}};
    arma::mat H ("1 0");
    arma::mat B ("1; 1");

    mlpack::distribution::GaussianDistribution w(arma::zeros<arma::vec>(2), arma::eye<arma::mat>(2,2));
    mlpack::distribution::GaussianDistribution v(arma::zeros<arma::vec>(1), arma::eye<arma::mat>(1,1));

    mlpack::distribution::GaussianDistribution x0(arma::zeros<arma::vec>(2), arma::eye<arma::mat>(2,2));

    filter::Kalman kf(F,H,B,w,v); 

    // whithout control input
    kf.initialize(x0);
    BOOST_REQUIRE(arma::approx_equal(arma::vec({0,0}),kf.state().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{1,0},{0,1}}),kf.state().Covariance(), "absdiff", diff_tol));
    
    kf.predict(); 
    BOOST_REQUIRE(arma::approx_equal(arma::vec({0,0}),kf.predicted().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{2,0},{0,2}}),kf.predicted().Covariance(), "absdiff", diff_tol));

    kf.filter(arma::vec({1}));
    BOOST_REQUIRE(arma::approx_equal(arma::vec({0.6667,0}),kf.state().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{0.6667,0},{0,2}}),kf.state().Covariance(), "absdiff", diff_tol));

    // with control input
    kf.initialize(x0);
    BOOST_REQUIRE(arma::approx_equal(arma::vec({0,0}),kf.state().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{1,0},{0,1}}),kf.state().Covariance(), "absdiff", diff_tol));
    
    kf.predict(arma::vec({2})); 
    BOOST_REQUIRE(arma::approx_equal(arma::vec({2,2}),kf.predicted().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{2,0},{0,2}}),kf.predicted().Covariance(), "absdiff", diff_tol));

    kf.filter(arma::vec({3}));
    BOOST_REQUIRE(arma::approx_equal(arma::vec({2.6667,2}),kf.state().Mean(), "absdiff", diff_tol));
    BOOST_REQUIRE(arma::approx_equal(arma::mat({{0.6667,0},{0,2}}),kf.state().Covariance(), "absdiff", diff_tol));
}

BOOST_AUTO_TEST_SUITE_END();
