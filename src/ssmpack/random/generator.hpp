#pragma once

#include <random>

/* I know nothing about thread-safety of this */

namespace ssmpack {
namespace random {

typedef std::mt19937 CoreGenerator;

class Generator {
  CoreGenerator gen_;
  static Generator *instance_;
  Generator() {}

 public:
  static Generator &get() {
    static Generator instance;
    return instance;
  }

  CoreGenerator & getGenerator() {return gen_;}
  
  template<class TSeed>
  void setSeed(TSeed seed) {gen_.seed(seed);}

  void setRandomSeed()
  {
    std::random_device rd;
    gen_.seed(rd());
  }

};

inline void setRandomSeed()
{
  Generator::get().setRandomSeed();
}

template<class TSeed>
void setSeed(TSeed seed)
{
  Generator::get().setSeed(seed);
}

}
}
