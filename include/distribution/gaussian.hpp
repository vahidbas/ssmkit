#pragma once

#include <tuple>
#include <algorithm>
#include <vector>
#include <mlpack/core.hpp>

#include "distribution/particle.hpp"

namespace PROJECT_NAME {
namespace distribution {

   class  Gaussian{

       public:
       using PARTICLE_TYPE = Particle<arma::vec>;
       using RV_TYPE = arma::vec;
       using PARAM_TYPE = std::tuple<arma::vec,arma::mat>;

       Gaussian(){}
       Gaussian(const size_t dimention): dist(dimention) {}
       Gaussian(const arma::vec &mean, const arma::mat &covariance): dist(mean,covariance) {}

       arma::vec Random() {return dist.Random();}
       double Likelihood(const arma::vec &observation) {return dist.Probability(observation);}

       // sample one particle
       PARTICLE_TYPE ParticleSample()
       {
            PARTICLE_TYPE p;
            p.point  = Random();
            p.weight = Likelihood(p.point);
            return p;
       }

        // sample N particles
        void ParticleSample_N(std::vector<PARTICLE_TYPE> & pars, size_t N)
        {
            pars.clear();
            pars.resize(N);
            std::generate_n(pars.begin(), N, [this](){return ParticleSample();});           
        }

        Gaussian & Parameterize(const PARAM_TYPE &parameters)
        {
            dist.Mean() = std::get<0>(parameters);
            dist.Covariance(std::get<1>(parameters));
            return (*this);
        }

       private:
       mlpack::distribution::GaussianDistribution dist;

   };

} // PROJECT_NAME
} // distribution
