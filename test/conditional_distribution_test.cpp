#include <boost/test/unit_test.hpp>
#include <iostream>

#include "distribution/gaussian.hpp"
#include "distribution/conditional_distribution.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(conditional_distribution_test);

std::tuple<arma::vec, arma::mat> pfunc(double x)
{
   std::tuple<arma::vec, arma::mat> p;
   std::get<0>(p) = {x, x};
   std::get<1>(p) = {{1, 0},{0, 1}};
   return p;
}
BOOST_AUTO_TEST_CASE(one_step_test)
{
    distribution::Gaussian g(2);
    auto cpdf = makeParametericConditionalDistribution(g,pfunc);
    std::cout << cpdf.Random(100) << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
