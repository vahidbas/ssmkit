#include <mlpack/core.hpp>

namespace PROJECT_NAME {
namespace filter {
    using mlpack::distribution::GaussianDistribution;  
 //   template<typename... >
    class Kalman {
        public:
        Kalman(
            arma::mat dynamic_m,
            arma::mat observation_m,
            arma::mat control_m,
            GaussianDistribution dynamic_noise,
            GaussianDistribution observation_noise
            ) : dyn_m(dynamic_m), obs_m(observation_m), cnt_m(control_m),
                dyn_n(dynamic_noise),
                obs_n(observation_noise)
            {}

        Kalman(
            arma::mat dynamic_m,
            arma::mat observation_m,
            GaussianDistribution dynamic_noise,
            GaussianDistribution observation_noise
            ) : dyn_m(dynamic_m), obs_m(observation_m),
                dyn_n(dynamic_noise),
                obs_n(observation_noise)
            {
                cnt_m.zeros(dyn_m.n_rows,1);
            }

        void initialize(GaussianDistribution initial_state);
        void predict(arma::vec control);
        void predict() {predict(arma::zeros<arma::vec>(cnt_m.n_cols));}
        void filter(arma::vec observation);
        
        const GaussianDistribution state() const {return filter_s;}
        const GaussianDistribution predicted() const {return predict_s;}

        private:
        arma::mat dyn_m;
        arma::mat obs_m;
        arma::mat cnt_m;
        GaussianDistribution dyn_n;
        GaussianDistribution obs_n;
        GaussianDistribution filter_s;
        GaussianDistribution predict_s;
    };
}
}

void PROJECT_NAME::filter::Kalman::
initialize(GaussianDistribution initial_state)
{
    filter_s = initial_state;
}

void PROJECT_NAME::filter::Kalman::
predict(arma::vec control)
{
    predict_s.Mean() = dyn_m * filter_s.Mean() + cnt_m * control;
    predict_s.Covariance( 
                dyn_m * filter_s.Covariance() * dyn_m.t() + dyn_n.Covariance());
}

void PROJECT_NAME::filter::Kalman::
filter(arma::vec observation)
{
    arma::vec inovation = observation - obs_m * predict_s.Mean();
    arma::mat inovation_cov = 
            obs_m * predict_s.Covariance() * obs_m.t() +
            obs_n.Covariance();
    arma::mat kalman_gain =  predict_s.Covariance() * obs_m.t() * arma::inv_sympd(inovation_cov);

    filter_s.Mean() =  predict_s.Mean() + kalman_gain * inovation;
    filter_s.Covariance( predict_s.Covariance() - kalman_gain * obs_m * predict_s.Covariance());
}
