/**
 * @file markov.hpp
 * @author Vahid Bastani
 *
 * Generic class for first-order Markovian stochastic process
 */
#ifndef SSMPACK_PROCESS_MARKOV_HPP__
#define SSMPACK_PROCESS_MARKOV_HPP__

#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ssmpack {
namespace process {

/**
 * A first-order Markov process
 */
template <typename CPDF>
class Markov {
 private:
  //! Type of the conditional distribution without reference
  using URCPDF = typename std::remove_reference<CPDF>::type;
  // Check if the types condition and random variable of the CPDF are the same
  static_assert(
      std::is_same<typename URCPDF::RV_TYPE, typename URCPDF::CV_TYPE>::value,
      "condition and random variables of cpdf should be same for first order "
      "Markov process");

 public:
  //! Type of the state variable
  using STATE_TYPE = typename URCPDF::RV_TYPE;

 private:
  //! Conditional probability density function
  CPDF cpdf_;
  //! State variable
  STATE_TYPE state_;
  //! Log likelihood of current sequence
  double log_likelihood_;

 public:
  /**
   * Creates a Markov process with given conditional probability density
   * function (cpdf).
   */
  Markov(URCPDF cpdf = URCPDF()) : cpdf_(cpdf) {}

  /**
   * Resets the state of process. It also resets likelihood to zero
   */
  void Reset(const STATE_TYPE & state){
    state_ = state;
    log_likelihood_ = 0;
  }

  /**
   * Initialize the process using initial state probability density function
   * (init_pdf)
   */
  template <typename INIT_PDF>
  STATE_TYPE Initialize(INIT_PDF &init_pdf) {
    auto init_par = init_pdf.ParticleSample();
    state_ = init_par.point;
    log_likelihood_ = std::log(init_par.weight);
    return state_;
  }

  /**
   * Sample one state variable from process and move the state
   */
  STATE_TYPE Random() {
    auto par = cpdf_.ParticleSample(state_);
    state_ = par.point;
    log_likelihood_ += std::log(par.weight);
    return state_;
  }

  /**
   * Sample a sequence of n states from process with initial PDF (init_pdf) and
   * return the likelihood of sequence. Sampling moves the process
   */
  template <typename INIT_PDF>
  double Random(std::vector<STATE_TYPE> &samples, INIT_PDF &init_pdf,
                size_t n) {
    static_assert(
        std::is_same<typename INIT_PDF::RV_TYPE, STATE_TYPE>::value,
        "random variable of initial pdf is different from the state variable");

    samples.clear();
    samples.resize(n);

    samples[0] = Initialize(init_pdf);
    std::generate_n(samples.begin() + 1, n - 1, [this]() { return Random(); });
    return log_likelihood_;
  }

  /**
   * Sample a sequence of n states from process and return the likelihood of
   * the sequence. Sampling moves the process.
   */
  double Random(std::vector<STATE_TYPE> &samples, size_t n) {
    samples.clear();
    samples.resize(n);
    double initial_loglik = log_likelihood_;
    std::generate_n(samples.begin(), n, [this]() { return Random(); });
    return log_likelihood_ - initial_loglik;
  }
};

//! A simple constructor for Markov process
template <typename T>
Markov<T> makeMarkov(T &&cpdf) {
  return Markov<T>(std::forward<T>(cpdf));
}

} // namespace process
} // namespace ssmpack

#endif //SSMPACK_PROCESS_MARKOV_HPP__
