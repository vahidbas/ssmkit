#pragma once

#include <algorithm>
#include <vector>

namespace ssmpack {
namespace process {

template <typename TProcess>
class BaseProcess {
 public:
  template <typename TRandomVAR, typename... TArgs>
  void random_n(std::vector<TRandomVAR> &output, const size_t &n,
                const TArgs &... args) {
    output.resize(n);
    std::generate_n(output.begin(), n, [this, &args...]() {
      return static_cast<TProcess *>(this)->random(args...);
    });
  }
};
}
}
