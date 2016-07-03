/**
 * @file ess.hpp
 * @author Vahid Bastani
 *
 * Effective Sample Size (ESS) resampling criterion
 */
#ifndef SSMPACK_FILTER_RESAMPLER_CRITERION_ESS
#define SSMPACK_FILTER_RESAMPLER_CRITERION_ESS

#include <armadillo>

namespace ssmkit {
namespace filter {
namespace resampler {
namespace criterion {

struct ESS{
  double th;
  /**
   * @param threshold: minimum number of effective sample.
   */
  ESS(double threshold) : th{threshold} {}
  ESS() = delete;
  /**
   * return true if number of effective samples with weights w falls bellow the
   * threshold
   */
  bool operator()(const arma::vec &w){
    return arma::sum(w)/arma::sum(arma::square(w)) < th;
  }

};
} // namespace criterion
} // namespace resampler
} // namespace filter
} // namespace ssmkit
#endif // SSMPACK_FILTER_RESAMPLER_CRITERION_ESS
