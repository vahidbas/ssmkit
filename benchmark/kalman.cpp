#include <benchmark/benchmark.h>


#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/filter/kalman.hpp"

using namespace ssmpack;

auto make(){
  double delta = 0.1; // sample time
  arma::mat dynamic_matrix{
      {1, 0, delta, 0}, {0, 1, 0, delta}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  
  arma::mat dynamic_noise{
      {0.1, 0, 0, 0}, {0, 0.1, 0, 0}, {0, 0, 0.1, 0}, {0, 0, 0, 0.1}};
  
  arma::mat measurement_matrix{{1, 0, 0, 0}, {0, 1, 0, 0}};
  arma::mat measurement_noise{{0.1, 0}, {0, 0.1}};
  
  auto dynamic_model = map::LinearGaussian(dynamic_matrix, dynamic_noise);
  auto measurement_model =
      map::LinearGaussian(measurement_matrix, measurement_noise);
  
  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(4), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(2), measurement_model);
  
  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(4));
  auto measurement_process =
      process::makeMemoryless(measurement_cpdf);
  
  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);
  
  auto kalman = filter::makeKalman(joint_process);
  kalman.initialize();
  return kalman;
}


auto kalman = make();

arma::vec meas {0, 0};

static void ssmpack_kalman_predict(benchmark::State& state) {
  while (state.KeepRunning())
    kalman.predict();
}
//Register the function as a benchmark
BENCHMARK(ssmpack_kalman_predict);

// Define another benchmark
static void ssmpack_kalman_correct(benchmark::State& state) {
  while (state.KeepRunning())
    kalman.correct(meas);
}
BENCHMARK(ssmpack_kalman_correct);

#ifdef WITH_OpenCV

#include "opencv2/video/tracking.hpp"
using namespace cv;
auto make_opencv(){
  return KalmanFilter(4, 2, 0);
}
auto kalman_cv = make_opencv();
Mat_<float> meas_cv(2,1);

static void opencv_kalman_predict(benchmark::State& state) {
  while (state.KeepRunning())
    kalman_cv.predict();
}
//Register the function as a benchmark
BENCHMARK(opencv_kalman_predict);

// Define another benchmark
static void opencv_kalman_correct(benchmark::State& state) {
  while (state.KeepRunning())
    kalman_cv.correct(meas_cv);
}
BENCHMARK(opencv_kalman_correct);

#endif

BENCHMARK_MAIN();
