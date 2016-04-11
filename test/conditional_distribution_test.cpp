#include <boost/test/unit_test.hpp>
#include <iostream>

#include "model/linear_gaussian.hpp"
#include "distribution/gaussian.hpp"
#include "distribution/conditional_distribution.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(conditional_distribution_test);

BOOST_AUTO_TEST_CASE(one_step_test)
{
    model::LinearGaussian<2> f { {{1, 0}, {0, 1}}, {{1, 0},{0, 1}} };
    distribution::Gaussian<2> g;
    auto cpdf = makeParametericConditionalDistribution(g,f);
    std::cout << cpdf.Random({100, 0}) << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
