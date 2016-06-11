/**
 * @file generator.hpp
 * @author Vahid Bastani
 *
 * The random number engine.
 */
#ifndef SSMPACK_RANDOM_GENERATOR_HPP
#define SSMPACK_RANDOM_GENERATOR_HPP

#include <random>

namespace ssmpack {
namespace random {

typedef std::mt19937 CoreGenerator;

/** A singleton wrapper of an instance of random generator.
 * This is used in the rendom() methods of distribution classes.
 */
class Generator {
 private:
  //! The core STL random generator
  CoreGenerator gen_;
  /** Default constructor.
   *
   * This is private because there can be only one instance of this class. 
   *
   * @note The object get initialized with random seed on construction. This is
   * essential in multi-thread usage.
   */
  Generator() {
    // random initialization on construction
    setRandomSeed();
  }

 public:
 //! Returns a reference to singleton instance
  static Generator &get() {
    // this is thread local to avoid race condition on gen_
    static thread_local Generator instance;
    return instance;
  }
  //! Returns reference to the core generator
  CoreGenerator & getGenerator() {return gen_;}
  //! Sets the seed for the core generator
  template<class TSeed>
  void setSeed(TSeed seed) {gen_.seed(seed);}
  //! Sets a random seed for core generator
  void setRandomSeed()
  {
    std::random_device rd;
    gen_.seed(rd());
  }

};

//! Convenient function to set random seed for singleton generator object
inline void setRandomSeed()
{
  Generator::get().setRandomSeed();
}

//! Convenient function to set seed for singleton generator object
template<class TSeed>
void setSeed(TSeed seed)
{
  Generator::get().setSeed(seed);
}

} // namespace random
} // namespace ssmpack


#endif //SSMPACK_RANDOM_GENERATOR_HPP
