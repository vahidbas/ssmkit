#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"

#include <iostream>

using namespace ssmpack;

int main ()
{
  // sample time
  double delta = 0.1;
  // state transition matrix
  arma::mat F{{1, 0, delta,     0},
              {0, 1,     0, delta},
              {0, 0,     1,     0},
              {0, 0,     0,     1}};
  // dynamic noise covariance matrix
  arma::mat Q{{1, 0, 0, 0},
              {0, 1, 0, 0},
              {0, 0, 1, 0},
              {0, 0, 0, 1}};
  // measurement matrix
  arma::mat H{{1, 0, 0, 0},
              {0, 1, 0, 0}};
  // measurement noise covariance matrix
  arma::mat R{{1, 0},
              {0, 1}};
  // initial state pdf mean vector
  arma::vec x0{0, 0, 0, 0};
  // initial state pdf covariance matrix
  arma::mat P0{{1, 0, 0, 0},
               {0, 1, 0, 0},
               {0, 0, 1, 0},
               {0, 0, 0, 1}};

}
