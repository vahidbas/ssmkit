#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmkit/filter/particle.hpp"
#include "ssmkit/filter/resampler/systematic.hpp"
#include "ssmkit/filter/resampler/criterion/ess.hpp"
#include "ssmkit/map/linear_gaussian.hpp"
#include "ssmkit/distribution/gaussian.hpp"
#include "ssmkit/distribution/conditional.hpp"
#include "ssmkit/process/markov.hpp"
#include "ssmkit/process/memoryless.hpp"
#include "ssmkit/process/hierarchical.hpp"

#include <tuple>

using namespace ssmkit;

BOOST_AUTO_TEST_SUITE(filter_particle);

BOOST_AUTO_TEST_CASE(builder)
{
  unsigned int state_dim = 8;
  unsigned int measu_dim = 3;
  unsigned int num_particle = 100;

  auto dynamic_model = map::LinearGaussian(arma::eye<arma::mat>(state_dim, state_dim),
                                               arma::eye<arma::mat>(state_dim, state_dim));
  auto measurement_model = map::LinearGaussian(arma::eye<arma::mat>(measu_dim, state_dim),
                                                   arma::eye<arma::mat>(measu_dim, measu_dim));

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(state_dim), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(measu_dim), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(state_dim));
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto resampler = filter::resampler::makeSystematic(
      filter::resampler::criterion::ESS(num_particle * 0.8));

  auto pfilter = filter::makeParticle(joint_process, resampler, num_particle);

  BOOST_CHECK_EQUAL(pfilter.getWeights().n_rows, num_particle);
  BOOST_CHECK_EQUAL(pfilter.getStateParticles().n_cols, num_particle);
  BOOST_CHECK_EQUAL(pfilter.getStateParticles().n_rows, state_dim);
}

BOOST_AUTO_TEST_CASE(initializer)
{
  unsigned int state_dim = 3;
  unsigned int measu_dim = 5;
  unsigned int num_particle = 100;

  auto dynamic_model = map::LinearGaussian(arma::eye<arma::mat>(state_dim, state_dim),
                                               arma::eye<arma::mat>(state_dim, state_dim));
  auto measurement_model = map::LinearGaussian(arma::eye<arma::mat>(measu_dim, state_dim),
                                                   arma::eye<arma::mat>(measu_dim, measu_dim));

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(state_dim), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(measu_dim), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(state_dim));
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto resampler = filter::resampler::makeSystematic(
      filter::resampler::criterion::ESS(num_particle * 0.8));

  auto pfilter = filter::makeParticle(joint_process, resampler, num_particle);
  auto cstate = pfilter.initialize();

  BOOST_REQUIRE_EQUAL(pfilter.getWeights().n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_rows, state_dim);

  BOOST_CHECK_EQUAL(std::get<1>(cstate).n_rows, num_particle);
  BOOST_CHECK_EQUAL(std::get<0>(cstate).n_cols, num_particle);
  BOOST_CHECK_EQUAL(std::get<0>(cstate).n_rows, state_dim);
  
  BOOST_CHECK(arma::all(std::get<1>(cstate) == pfilter.getWeights()));
  BOOST_CHECK(arma::all(arma::vectorise(std::get<0>(cstate)) == 
                        arma::vectorise(pfilter.getStateParticles())));

  // check if weights are normalized
  BOOST_CHECK_CLOSE(arma::accu(std::get<1>(cstate)), 1.0, 0.001);
}

BOOST_AUTO_TEST_CASE(predict)
{
  unsigned int state_dim = 4;
  unsigned int measu_dim = 2;
  unsigned int num_particle = 100;

  auto dynamic_model = map::LinearGaussian(arma::eye<arma::mat>(state_dim, state_dim),
                                               arma::eye<arma::mat>(state_dim, state_dim));
  auto measurement_model = map::LinearGaussian(arma::eye<arma::mat>(measu_dim, state_dim),
                                                   arma::eye<arma::mat>(measu_dim, measu_dim));

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(state_dim), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(measu_dim), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(state_dim));
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto resampler = filter::resampler::makeSystematic(
      filter::resampler::criterion::ESS(num_particle * 0.8));

  auto pfilter = filter::makeParticle(joint_process, resampler, num_particle);
  auto i_state = pfilter.initialize();
  // prediction
  pfilter.predict();

  BOOST_REQUIRE_EQUAL(pfilter.getWeights().n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_rows, state_dim);

  BOOST_REQUIRE_EQUAL(std::get<1>(i_state).n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(i_state).n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(i_state).n_rows, state_dim);
  // weights should not change after prediction
  BOOST_REQUIRE(arma::all(std::get<1>(i_state) == pfilter.getWeights()));
  // state should change after prediction
  BOOST_CHECK(!arma::all(arma::vectorise(std::get<0>(i_state)) == 
                        arma::vectorise(pfilter.getStateParticles())));

  // check if weights are normalized
  BOOST_CHECK_CLOSE(arma::accu(pfilter.getWeights()), 1.0, 0.001);
}

BOOST_AUTO_TEST_CASE(correct)
{
  unsigned int state_dim = 4;
  unsigned int measu_dim = 2;
  unsigned int num_particle = 100;

  auto dynamic_model = map::LinearGaussian(arma::eye<arma::mat>(state_dim, state_dim),
                                               arma::eye<arma::mat>(state_dim, state_dim));
  auto measurement_model = map::LinearGaussian(arma::eye<arma::mat>(measu_dim, state_dim),
                                                   arma::eye<arma::mat>(measu_dim, measu_dim));

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(state_dim), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(measu_dim), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(state_dim));
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto resampler = filter::resampler::makeSystematic(
      filter::resampler::criterion::ESS(num_particle * 0.8));

  auto pfilter = filter::makeParticle(joint_process, resampler, num_particle);
  auto i_state = pfilter.initialize();
  // prediction
  pfilter.predict();
  auto c_state = pfilter.correct(arma::vec({0, 0}));

  BOOST_REQUIRE_EQUAL(pfilter.getWeights().n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(pfilter.getStateParticles().n_rows, state_dim);

  BOOST_REQUIRE_EQUAL(std::get<1>(i_state).n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(i_state).n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(i_state).n_rows, state_dim);
  BOOST_REQUIRE_EQUAL(std::get<1>(c_state).n_rows, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(c_state).n_cols, num_particle);
  BOOST_REQUIRE_EQUAL(std::get<0>(c_state).n_rows, state_dim);
  
  // check if internal values equal to returned ones
  BOOST_CHECK(arma::all(std::get<1>(c_state) == pfilter.getWeights()));
  BOOST_CHECK(arma::all(arma::vectorise(std::get<0>(c_state)) ==
                        arma::vectorise(pfilter.getStateParticles())));

  // weights should change after correct
  BOOST_CHECK(!arma::all(std::get<1>(i_state) == std::get<1>(c_state)));
  // state should change after prediction
  BOOST_CHECK(!arma::all(arma::vectorise(std::get<0>(i_state)) == 
                        arma::vectorise(std::get<0>(c_state))));

  // check if weights are normalized
  BOOST_CHECK_CLOSE(arma::accu(pfilter.getWeights()), 1.0, 0.001);
}

BOOST_AUTO_TEST_SUITE_END();
