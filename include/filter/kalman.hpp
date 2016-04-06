#include <mlpack/core.hpp>

namespace PROJECT_NAME {
namespace filter {
 //   template<typename... >
    class Kalman {
        public:
        Kalman(){}

        void initialize();
        void predict();
        void filter(arma::vec observation);

        private:
        arma::mat dynamic_m;
        arma::mat observation_m;
        mlpack::distribution::GaussianDistribution dynamic_noise;
        mlpack::distribution::GaussianDistribution observation_noise;
        mlpack::distribution::GaussianDistribution filtered_state;
        mlpack::distribution::GaussianDistribution predicted_state;
    };
}
}
