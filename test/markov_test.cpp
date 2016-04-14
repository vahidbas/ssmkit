#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/model/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional_distribution.hpp"
#include "ssmpack/process/markov.hpp"

#include "ssmpack/process/hierachy.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(markov_test);

BOOST_AUTO_TEST_CASE(test1) {
  model::LinearGaussian<2,2> f{{{1, 1}, {0, 1}}, {{0.0001, 0}, {0, 0.0001}}};
  distribution::Gaussian<2> g;
  distribution::Conditional<decltype(g), decltype(f)> cpdf(g,f);  process::Markov<decltype(cpdf)> markov(cpdf);

auto r = markov.random();

double l = markov.likelihood(r);

process::Hierachy<decltype(markov), decltype(markov)> tt;

//  std::vector<typename decltype(markov_p)::TStateVAR> v;
//  arma::arma_rng::set_seed_random();
//  std::cout << markov_p.random(v, g, 10) << std::endl;
//
//  std::for_each(v.begin(), v.end(),
//                [](typename decltype(markov_p)::TStateVAR &p) {
//                  std::cout << p << std::endl;
//                });
}

BOOST_AUTO_TEST_SUITE_END();
