Constant velocity dynamic model {#tutorial_constant_velocity}
====================

Introduction
--------------------
In this tutorial we show how `ssmpack` can be used for simulating and tracking a
constant velocity inertial model. The code for this tutorial can be
found in `/example/tutorial.cpp`. 

The State Space Model
---------------------
Let
\f$\mathbf{x}_k = [x_1(k), x_2(k), \dot{x}_1(k), \dot{x}_2(k)]^T \f$
be the state of an object moving in a 2D space at time \f$k\f$, where 
\f$x_1(k)\f$ and \f$\dot{x}_1(k)\f$
are position and velocity in the first dimension and
\f$x_1(k)\f$ and \f$\dot{x}_1(k)\f$
are position and velocity in the second dimension.
The dynamic of this state vector can be formulated as
\f[\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \boldsymbol\omega_k \f]
Where \f$\mathbf{F}\f$ is refered to as transition or dynamic matrix. It is defined
as
\f[ \mathbf{F} = 
\begin{bmatrix}
1 & 0 & \delta & 0      \\
0 & 1 & 0      & \delta \\
0 & 0 & 1      & 0      \\
0 & 0 & 0      & 1
\end{bmatrix}
\f]
for sampling rate of \f$\delta\f$ second. \f$\boldsymbol\omega\f$ is a zero-mean
white Gaussian random process with covariance matrix \f$\mathbf{Q}\f$.

> The \f$\boldsymbol\omega\f$ is meant to model our uncertainty about the
> deterministic dynamic model \f$\mathbf{F}\f$.

In real application often access to the object's state is only available through a
noisy measurement process:
\f{equation}{
\mathbf{z}_k = \mathbf{H}\mathbf{x}_k+\boldsymbol\nu
\f}

wher \f$\mathbf{z}_k\f$ is measurement vector at time \f$k\f$. \f$\boldsymbol\nu\f$ is a zero-mean white Gaussian random process with
covariance matrix \f$\mathbf{R}\f$. \f$\mathbf{H}\f$ is called measurement 
matrix which for our purpose is defined as:
\f[ \mathbf{H} = 
\begin{bmatrix}
1 & 0 & 0 & 0      \\
0 & 1 & 0 & 0
\end{bmatrix}
\f]
which simply mean that the measurment vector is of the positin
of the object plus a Gaussian noise.


> Unlike \f$\boldsymbol\omega\f$, the purpose of \f$\boldsymbol\nu\f$ is to model
> imperfection related to the measurement device.

The dynamic and measurement equations can easily be interepreted as conditional
probability distribution functions (CPDF):
\f[
p(\mathbf{x}_k|\mathbf{x}_{k-1}) = \mathcal{N}(\mathbf{F}\mathbf{x}_{k-1}, \mathbf{Q})
\f]
\f[
p(\mathbf{z}_k|\mathbf{x}_{k}) = \mathcal{N}(\mathbf{H}\mathbf{x}_{k}, \mathbf{R})
\f]
Let also assume that the initial state probability distribution function is a
Gaussian:
\f[
p(\mathbf{x}_0) = \mathcal{N}(\hat{\mathbf{x}}_0, \mathbf{P}_0)
\f]

These probability density functions fully characterize a stochastic process that
can be represented by a two-layer hierarchical dynamic Bayesian network (DBN)
of this form

![two-layer hierarchical DBN](kalman.png "two-layer")

The process construction
------------------------
`ssmpack` coding style is almost the same formal mathematical construction of
dynamic Bayesian networks. We start by defining the parametric PDFs then
building CPDFs based on them. These CPDFs are used to define simple stochastic
processes which are later combined to make more complex stochastic processes.

~~~{.cpp}
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
// initial stae pdf covariance matrix
arma::mat P0{{1, 0, 0, 0},
             {0, 1, 0, 0},
             {0, 0, 1, 0},
             {0, 0, 0, 1}};
~~~

In the previous section we mathematically constructed our state space model.

The class ssmpack::distribution::Gaussian provides multivariate Gaussian. As
first step We make a 4-dimensional Gaussian and 2-dimensional Gaussian  state and
measurement processes respectively.

~~~

dyn_pdf = ssmpack::distribution::Gaussian(4);  // dynamic PDF
mea_pdf = ssmpack::distribution::Gaussian(2);  // measurement PDF

~~~
`ssmpack` construct CPDFs from parametric parametric PDFs and special kind of
functors called parameter maps.
\f$p(\mathbf{x}_k|\mathbf{x}_{k-1})=\mathcal{N}(g(\mathbf{x}_{k-1}))\f$
where 
\f$g(\mathbf{x}_{k-1})\f$ is the parameter map that receives an state variable
and returns corresponding parameter set for Gaussian distribution
\f$(\mathbf{H}\mathbf{x}_{k}, \mathbf{R})\f$. the class
ssmpack::map::LinearGaussian provides the requried parameter map for our case


by default the Gaussian have zero mean vector and identity covariance matrix. 
