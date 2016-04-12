/**
 * @file measurement.hpp
 * @author Vahid Bastani
 *
 * Generic class for measurement from stochastic process
 */
#ifndef SSMPACK_PROCESS_MEASUREMENT_HPP__
#define SSMPACK_PROCESS_MEASUREMENT_HPP__

#include <type_traits>
#include <tuple>

namespace ssmpack {
namespace process {

/**
 * A measurement process
 *
 */
template <typename LATENT_PROC, typename CPDF>
class Measurement {
 private:
  using UR_PROC = typename std::remove_reference<LATENT_PROC>::type;
  using UR_CPDF = typename std::remove_reference<CPDF>::type;

 public:
  using MV_TYPE = typename UR_CPDF::RV_TYPE;
  using LV_TYPE = typename UR_PROC::STATE_TYPE;
  using MV_LV_TYPE = std::tuple<MV_TYPE,LV_TYPE>;

  private:
  //! Conditional probability density function (measurement model)
  UR_CPDF cpdf_;
  //! Latent stochastic process
  UR_PROC process_;

 public:
  /**
   * Creates a measurement process given measurement model as  conditional
   * probability density function and the latent state process model.
   */
  Measurement(const UR_PROC &process, const UR_CPDF &measurement_cpdf)
      : process_(process), cpdf_(measurement_cpdf) {}

  /**
   * Initialize the process using initial latent state probability density
   * function.
   */
  template<typename INIT_PDF>
  LV_TYPE Initialize(INIT_PDF &&init_pdf)
  {
    return process_.Initialize<INIT_PDF>(init_pdf);
  }

  /**
   * Sample one measurement and move the process forward one step.
   */
   MV_LV_TYPE Random() {
     auto state = process_.Random();
     auto measurement = cpdf_.Random(state);
     return std::make_tuple(measurement,state);
   }
};

//! A simple constructor for measurement process
template <typename LATENT_PROC, typename CPDF>
Measurement<LATENT_PROC, CPDF> makeMeasurement(LATENT_PROC &&process,
                                               CPDF &&measurement_cpdf) {
  return Measurement<LATENT_PROC, CPDF>(std::forward<LATENT_PROC>(process),
                                        std::forward<CPDF>(measurement_cpdf));
}

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MEASUREMENT_HPP__
