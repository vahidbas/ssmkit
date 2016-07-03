#pragma once

#include <algorithm>
#include <vector>

namespace ssmkit {
namespace process {

template <typename TProcess>
class BaseProcess {
 public:
  template <typename... TArgs>
  decltype(auto) random_n(const size_t &n, const TArgs &... args) {
    std::vector<decltype(static_cast<TProcess *>(this)->random(args...))>
        output(n);
    std::generate_n(output.begin(), n, [this, &args...]() {
      return static_cast<TProcess *>(this)->random(args...);
    });
    return output;
  }
};

} // namespace process
} // namespace ssmkit
