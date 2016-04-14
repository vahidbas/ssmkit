#pragma once

#include <vector>
#include <algorithm>

namespace PROJECT_NAME{
  namespace distribution{

    template<typename T>
      struct Particle {
        T point;
        double weight;
      };

    // normalize the weights of a set of particles
    template<typename T>
      double normalize_particles(std::vector<Particle<T>> &particles)
      {
        double sum_weights = 0;
        std::for_each(particles.begin(), particles.end(),
        [&sum_weights](Particle<T> &p){sum_weights+=p.weight;}); 
        std::for_each(particles.begin(), particles.end(),
        [sum_weights](Particle<T> &p){p.weight/=sum_weights;});
        return sum_weights;
      }

  }
}
