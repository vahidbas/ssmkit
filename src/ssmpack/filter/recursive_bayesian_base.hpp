#pragma once

#include <vector>
#include <algorithm>
#include <tuple>

namespace ssmpack {
namespace filter {

template <class TFilter>
class RecursiveBayesianBase {
  public:
  template <class TMeasurement, class... TArgs>
  decltype(auto) step(const TMeasurement &measurement, const TArgs &... args) {
    static_cast<TFilter *>(this)->predict();
    return static_cast<TFilter *>(this)->correct(measurement);
  }

  template <class TMeasurement, class TControl = std::tuple<>>
  decltype(auto)
  filter(const std::vector<TMeasurement> &measurements,
         const std::vector<TControl> &controls = std::vector<std::tuple<>>()) {

    using TState = decltype(step(measurements.at(0)));
    std::vector<TState> state_seq(measurements.size() + 1);
    state_seq[0] = static_cast<TFilter *>(this)->initialize();
    auto pm = measurements.cbegin();

    if (controls.size() == 0) {
      std::generate_n(state_seq.begin() + 1, state_seq.size(),
                      [this, &pm]() { return step(*(pm++)); });
    } else {
      // else what?
    }

    return state_seq;

  }

};

} // namespace filter
} // namespace ssmpack
