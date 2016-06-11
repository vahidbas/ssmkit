#include <boost/test/unit_test.hpp>

#include "ssmpack/random/generator.hpp"

#include <thread>
#include <array>
#include <random>
#include <algorithm>

using namespace ssmpack;
using namespace std;

BOOST_AUTO_TEST_SUITE(generator_test);

// thread task
void sample(double *r){
  uniform_real_distribution<double> dist;
  *r = dist(ssmpack::random::Generator::get().getGenerator());
}

BOOST_AUTO_TEST_CASE(multithread) {
  constexpr unsigned int n = 100;
  // array to store samples
  array<double, n> samples;
  array<thread, n> thrd;
  unsigned int repeat = 20;
  
  for (int i=0; i<repeat; i++){
    auto sb = samples.begin();
    generate_n(thrd.begin(), n, [&sb]() { return thread(sample, sb++); });
    for_each(thrd.begin(), thrd.end(), [](auto &t) { t.join(); });

    // check if there is any replicate
    bool ch =
        all_of(samples.begin(), samples.end(), [&samples](const auto &sb) {
          return count_if(samples.begin(), samples.end(), [&sb](const auto &s) {
                   return s == sb;
                 }) > 1 ? true : false;
        });
    BOOST_CHECK(!ch);
  }
}

BOOST_AUTO_TEST_SUITE_END();
