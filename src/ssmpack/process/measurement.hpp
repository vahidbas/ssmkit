/**
 * @file measurement.hpp
 * @author Vahid Bastani
 *
 * Generic class for measurement from stochastic process
 */
#ifndef SSMPACK_PROCESS_MEASUREMENT_HPP
#define SSMPACK_PROCESS_MEASUREMENT_HPP

#include <type_traits>
#include <tuple>

namespace ssmpack {
namespace process {

/**
 * A measurement process
 *
 */
template <typename TLatentPRC, typename TMeasurementCPDF>
class Measurement {
 private:
  using TLatentPRC_ = typename std::remove_reference<TLatentPRC>::type;
  using TMeasurementCPDF_ = typename std::remove_reference<TMeasurementCPDF>::type;

 public:
 //! Type of the measurement variable.
  using TMeasurementVAR = typename TMeasurementCPDF_::TRandomVAR;
  //! Type of the letent state variable.
  using TLatentVAR = typename TLatentPRC_::TStateVAR;
  //! Type of joint measurment-latent state variable.
  using TJointVAR = std::tuple<TMeasurementVAR,TLatentVAR>;

  private:
  //! Conditional probability density function (measurement model)
  TMeasurementCPDF_ cpdf_;
  //! Latent stochastic process
  TLatentPRC_ process_;

 public:
  /**
   * Creates a measurement process given measurement model as  conditional
   * probability density function and the latent state process model.
   */
  Measurement(const TLatentPRC_ &process, const TMeasurementCPDF_ &measurement_cpdf)
      : process_(process), cpdf_(measurement_cpdf) {}

  /**
   * Initialize the process using initial latent state probability density
   * function.
   */
  template<typename INIT_PDF>
  TLatentVAR initialize(INIT_PDF &&init_pdf)
  {
    return process_.initialize<INIT_PDF>(init_pdf);
  }

  /**
   * Sample one measurement and move the process forward one step.
   */
   TJointVAR random() {
     auto state = process_.random();
     auto measurement = cpdf_.random(state);
     return std::make_tuple(measurement,state);
   }
};

/**
 * A simple constructor for measurement process. For easy deduction of he types
 * of latent random process (TLatentPRC) and measurement model cpdf
 * (TMeasurementCPDF).
 */
template <typename TLatentPRC, typename TMeasurementCPDF>
Measurement<TLatentPRC, TMeasurementCPDF> makeMeasurement(TLatentPRC &&process,
                                               TMeasurementCPDF &&measurement_cpdf) {
  return Measurement<TLatentPRC, TMeasurementCPDF>(std::forward<TLatentPRC>(process),
                                        std::forward<TMeasurementCPDF>(measurement_cpdf));
}

} // namespace process
} // namespace ssmpack

#endif //SSMPACK_PROCESS_MEASUREMENT_HPP
