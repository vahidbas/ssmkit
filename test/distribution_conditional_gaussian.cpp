#include <boost/test/unit_test.hpp>
#include <iostream>

#include "distribution/conditional_gaussian.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(conditional_gaussian_distribution_test);

BOOST_AUTO_TEST_CASE(one_step_test)
{
    auto j = distribution::makeConditionalGaussian([](double x){return arma::vec("0 0")+x;},
    [](double x){return arma::mat("1 0; 0 1");});

    std::cout << j.Random(10) << std::endl;
    std::cout << j.Random(0) << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
