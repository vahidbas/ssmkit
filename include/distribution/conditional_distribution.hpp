#pragma once
#include <type_traits>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/function_arity.hpp>

#include <vector>
#include <algorithm>

#include "distribution/particle.hpp"

namespace PROJECT_NAME{
namespace distribution{

    template <typename DIST_TYPE, typename PARAM_FUNC>
    class ParametricConditionalDistribution {
        // note: it is not thraed-safe
        public:
        static_assert(boost::function_types::function_arity<PARAM_FUNC>::value == 1,
                      "parameter function should accept only one argument");
         
        using CV_SEQ = typename boost::function_types::parameter_types<PARAM_FUNC>::type;
        using CV_TYPE = typename boost::mpl::at_c<CV_SEQ,0>::type;
        using RV_TYPE = typename boost::function_types::result_type<decltype(&DIST_TYPE::Random)>::type;
        using PARAM_TYPE = typename boost::function_types::result_type<PARAM_FUNC>::type;

        // check if appropiate Parameterize method exists
        static_assert(
        std::is_same<decltype(std::declval<DIST_TYPE>().Parameterize(std::declval<PARAM_TYPE>()))
        ,typename std::add_lvalue_reference<DIST_TYPE>::type>::value, "Parametrize doesn't return void");

        using PARTICLE_TYPE = Particle<RV_TYPE>;
        public:
        RV_TYPE Random(CV_TYPE cv)
        {return dist.Parameterize(param_func(cv)).Random();}

        const DIST_TYPE & getDistribution(CV_TYPE cv)
        {return dist.Parameterize(param_func(cv));}

        double Likelihood(RV_TYPE rv, CV_TYPE cv)
        {return dist.Parameterize(param_func(cv)).Likelihood(rv);}

        // sample one particle from conditioned distribution
        PARTICLE_TYPE ParticleSample(CV_TYPE cv)
        {
            Particle<RV_TYPE> p;
            dist.Parameterize(param_func(cv));
            p.point  = dist.Random();
            p.weight = dist.Likelihood(p.point);
            return p;
        }

        // sample N particle from conditioned distribution
        void ParticleSample_N(std::vector<PARTICLE_TYPE> & pars, size_t N, CV_TYPE cv)
        {
            pars.clear();
            std::generate_n(pars, N, [cv](){return ParticleSample(cv);});           
        }

        template<typename CV_DIST>
        double ApproxiamteMarginalLikelihood_MC(RV_TYPE rv, size_t N, CV_DIST cv_dist)
        {
            std::vector<typename CV_DIST::PARTICLE_TYPE> pars;
            cv_dist.ParticleSample_N(pars,N);
            normalize_particles(pars);
            double lik = 0;
            std::for_each(pars.begin(), pars.end(),
                          [&lik](typename CV_DIST::PARTICLE_TYPE & par){lik+=par.weight*Likelihood(par.point);});
        }

        private:
        DIST_TYPE dist;
        PARAM_FUNC param_func;
    };
} // namespace distribution
} // namespace PROJECT_NAME

using fff = double(*)(int);
struct ddd {char Random(); ddd & Parameterize(double);};
using a = PROJECT_NAME::distribution::ParametricConditionalDistribution<ddd,fff>;
static_assert(std::is_same<a::CV_TYPE, int>::value, "");
static_assert(std::is_same<a::RV_TYPE, char>::value, "");
static_assert(std::is_same<a::PARAM_TYPE, double>::value, "");


