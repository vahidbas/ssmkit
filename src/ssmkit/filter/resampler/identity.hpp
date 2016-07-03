/**
 * @file identity.hpp
 * @author Vahid Bastani
 *
 * A resampler that returns same particle and weights
 */
#ifndef SSMPACK_FILTER_RESAMPLER_IDENTITY
#define SSMPACK_FILTER_RESAMPLER_IDENTITY

namespace ssmkit {
namespace filter {
namespace resampler {

/** Returns same particles and weights
 */
struct Identity {
  template<class Particles, class Weights>
  void operator()(Particles &pars, Weights &w) {}
};

} // namespace resampler
} // namespace filter
} // namespace ssmkit
#endif //SSMPACK_FILTER_RESAMPLER_IDENTITY
