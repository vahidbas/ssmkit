#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/model/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional_distribution.hpp"
#include "ssmpack/process/markov.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(markov_test);

BOOST_AUTO_TEST_CASE(test1)
{
    model::LinearGaussian<2> f { {{1, 0}, {0, 1}}, {{0.1, 0},{0, 0.1}} };
    distribution::Gaussian<2> g;
    auto cpdf = makeParametericConditionalDistribution(g,f);
    
    auto markov_p = simulation::makeMarkov(cpdf);

    std::vector<typename decltype(markov_p)::STATE_TYPE> v;
    std::cout << markov_p.Random(v, g, 10) << std::endl;

    std::for_each(v.begin(), v.end(),
             [](typename decltype(markov_p)::STATE_TYPE &p)
             {std::cout << p << std::endl;});
}

BOOST_AUTO_TEST_SUITE_END();
