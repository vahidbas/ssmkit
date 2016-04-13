/**
 * @file markov.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process.
 */
#ifndef SSMPACK_PROCESS_MARKOV_HPP
#define SSMPACK_PROCESS_MARKOV_HPP

#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ssmpack {
namespace process {

/**
 * A first-order Markov process
 */
template <typename TTransitionCPDF>
class Markov {
 private:
  //! Type of the conditional distribution without reference
  using TTransitionCPDF_ =
      typename std::remove_reference<TTransitionCPDF>::type;
  // Check if the types condition and random variable of the TTransitionCPDF are
  // the same
  static_assert(
      std::is_same<typename TTransitionCPDF_::TRandomVAR,
                   typename TTransitionCPDF_::TConditionVAR>::value,
      "condition and random variables of cpdf should be same for first order "
      "Markov process");

 public:
  //! Type of the state variable
  using TStateVAR = typename TTransitionCPDF_::TRandomVAR;

 private:
  //! Conditional probability density function
  TTransitionCPDF_ cpdf_;
  //! State variable
  TStateVAR state_;
  //! Log likelihood of current sequence
  double log_likelihood_;

 public:
  /**
   * Creates a Markov process with given conditional probability density
   * function (cpdf).
   */
  Markov(TTransitionCPDF_ cpdf = TTransitionCPDF_()) : cpdf_(cpdf) {}

  /**
   * Resets the state of process. It also resets likelihood to zero
   */
  void reset(const TStateVAR &state) {
    state_ = state;
    log_likelihood_ = 0;
  }

  /**
   * Initialize the process using initial state probability density function
   * (init_pdf)
   */
  template <typename TInitialPDF_>
  TStateVAR initialize(TInitialPDF_ &init_pdf) {
    auto init_par = init_pdf.particle();
    state_ = init_par.point;
    log_likelihood_ = std::log(init_par.weight);
    return state_;
  }

  /**
   * Sample one state variable from process and move the state
   */
  TStateVAR random() {
    auto par = cpdf_.particle(state_);
    state_ = par.point;
    log_likelihood_ += std::log(par.weight);
    return state_;
  }

  /**
   * Sample a sequence of n states from process with initial PDF (init_pdf) and
   * return the likelihood of sequence. Sampling moves the process
   */
  template <typename TInitialPDF_>
  double random(std::vector<TStateVAR> &samples, TInitialPDF_ &init_pdf,
                size_t n) {
    static_assert(
        std::is_same<typename TInitialPDF_::TRandomVAR, TStateVAR>::value,
        "random variable of initial pdf is different from the state variable");

    samples.clear();
    samples.resize(n);

    samples[0] = initialize(init_pdf);
    std::generate_n(samples.begin() + 1, n - 1, [this]() { return random(); });
    return log_likelihood_;
  }

  /**
   * Sample a sequence of n states from process and return the likelihood of
   * the sequence. Sampling moves the process.
   */
  double random(std::vector<TStateVAR> &samples, size_t n) {
    samples.clear();
    samples.resize(n);
    double initial_loglik = log_likelihood_;
    std::generate_n(samples.begin(), n, [this]() { return random(); });
    return log_likelihood_ - initial_loglik;
  }
};

//! A simple constructor for Markov process
template <typename TTransitionCPDF>
Markov<TTransitionCPDF> makeMarkov(TTransitionCPDF &&cpdf) {
  return Markov<TTransitionCPDF>(std::forward<TTransitionCPDF>(cpdf));
}

} // namespace process
} // namespace ssmpack

#endif // SSMPACK_PROCESS_MARKOV_HPP
