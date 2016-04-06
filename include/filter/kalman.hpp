#include <mlpack/core.hpp>

namespace PROJECT_NAME {
namespace filter {
 //   template<typename... >
    class Kalman {
        public:
        Kalman(
            arma::mat dynamic_m,
            arma::mat observation_m,
            mlpack::distribution::GaussianDistribution dynamic_noise,
            mlpack::distribution::GaussianDistribution observation_noise
            ) : _dynamic_m(dynamic_m), _observation_m(observation_m),
                _dynamic_noise(dynamic_noise),
                _observation_noise(observation_noise)
            {
                _control_m.zeros(_dynamic_m.n_rows,1);
            }

        void initialize(mlpack::distribution::GaussianDistribution initial_state);
        void predict(arma::vec control);
        void predict() {predict(arma::zeros<arma::vec>(_control_m.n_cols));}
        void filter(arma::vec observation);

        private:
        arma::mat _dynamic_m;
        arma::mat _observation_m;
        arma::mat _control_m;
        mlpack::distribution::GaussianDistribution _dynamic_noise;
        mlpack::distribution::GaussianDistribution _observation_noise;
        mlpack::distribution::GaussianDistribution _filtered_state;
        mlpack::distribution::GaussianDistribution _predicted_state;
    };
}
}

void PROJECT_NAME::filter::Kalman::
initialize(mlpack::distribution::GaussianDistribution initial_state)
{
    _filtered_state = initial_state;
}

void PROJECT_NAME::filter::Kalman::
predict(arma::vec control)
{
    _predicted_state.Mean() = _dynamic_m * _filtered_state.Mean() + _control_m * control;
    _predicted_state.Covariance( 
                _dynamic_m * _filtered_state.Covariance() * _dynamic_m.t() + _dynamic_noise.Covariance());
}

void PROJECT_NAME::filter::Kalman::
filter(arma::vec observation)
{
    arma::vec inovation = observation - _observation_m * _predicted_state.Mean();
    arma::mat inovation_cov = 
            _observation_m * _predicted_state.Covariance() * _observation_m.t() +
            _observation_noise.Covariance();
    arma::mat kalman_gain =  _predicted_state.Covariance() * _observation_m.t() * arma::inv_sympd(inovation_cov);

    _filtered_state.Mean() =  _predicted_state.Mean() + kalman_gain * inovation;
    _filtered_state.Covariance( 
                (arma::eye<arma::mat>(_dynamic_m.n_rows, _dynamic_m.n_cols) - kalman_gain * _observation_m) * _predicted_state.Covariance());
}
