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

The first layer of this process is first-order Markov process where the state at
each time instance is only dependent on the state at previous time instant. The
second layer in an independent or memoryless process where the values of the
process at previous time instances do not have any effect on the value of next
time instances.

`ssmpack` coding style is almost the same formal mathematical construction of
dynamic Bayesian networks. We start by defining the parametric PDFs then
building CPDFs based on them. These CPDFs are used to define simple stochastic
processes which are later combined to make more complex stochastic processes.

Defining constant parameters
-----------------------------
We start by defining the parameters of the system:
~~~{.cpp}
// state dimensions
int state_dim = 4;
// measurement dimensions
int meas_dim = 2;
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
~~~
`arma::mat` and `arma::vec` are matrix and vector data-types from `armadillo` library.


Construction probability distribution functions
------------------------
we define initial state
PDF \f$p(\mathbf{x}_0)\f$ using ssmpack::distribution::Gaussian class:
~~~{.cpp}
// initial state pdf
ssmpack::distribution::Gaussian initial_pdf(x0, P0);
~~~
the class `ssmpack::distribution::Gaussian` implements a multivariate Gaussian
distribution and can be used for sampling and calculating likelihood of random
variables. Note that we passed mean vector and covariance matrix we defined
above to its constructor.

We now turn to definition the state transition CPDF and the measurement CPDF.
Before that it is important to understand how CPDFs are treated.
In `ssmpack` CPDFs are defined as a combination of a parametric PDF and a
parameter map. In other words \f$p(x|y) = \mathcal{F}(g(y))\f$ where
\f$\mathcal{F}(\theta)\f$ is parametric distribution (e.g. Gaussian) with
parameter Set \f$\theta\f$ and \f$g(y) = \theta\f$ maps the condition variable \f$y\f$
to a parameter \f$\theta\f$. The parameter set of Gaussian
is a tuple \f$(\mu, \Sigma)\f$ of mean vector and covariance matrix.
Thus, for our purpose the function \f$g\f$ should receive a
state variable and return \f$(\mathbf{F}\mathbf{x}, \mathbf{Q})\f$
for transition CPDF and \f$(\mathbf{H}\mathbf{x}, \mathbf{R})\f$ 
for measurement CPDF. This special kind of parameter map is implemented in class
`ssmpack::map::LinearGaussian(trans, cov)` which constructed using a linear transformation
matrix `trans` and a covariance matrix `cov`. Note that it always return the
same covariance matrix which constructed with. 

The class `ssmpack::distribution::Conditional` provides a generic class
that constructs a conditional distribution form a parametric distribution and a
parameter map. We can make any kind of CPDF with this class.

Now we can construct our CPDFs:
~~~{.cpp}
// state cpdf
auto state_cpdf = ssmpack::distribution::makeConditional(
    ssmpack::distribution::Gaussian(state_dim),
    ssmpack::map::LinearGaussian(F, Q));
// measurement cpdf
auto meas_cpdf = ssmpack::distribution::makeConditional(
    ssmpack::distribution::Gaussian(meas_dim),
    ssmpack::map::LinearGaussian(H, R));
~~~
we used `ssmpack::distribution::makeConditional` function for constructing CPDFs. This
is a convenient interface to class constructor. If we used the constructor
directly, we would have to specify template argument for that.
`ssmpack::distribution::Gaussian(dim)` is another constructor for Gaussian PDF
object with dimensions `dim`. Note that we pass the PDF object to `Conditional`
in order to let it know what type of distribution we need and the parameters of
distribution will come from the parameter map object.

Construction the state space model
------------------------
Now that we have the CPDFs we start constructing the hierarchical process.
First, let build each layer separately:
~~~{.cpp}
// state process
auto state_proc = ssmpack::process::makeMarkov(state_cpdf, initial_pdf);
// measurement process
auto meas_proc = ssmpack::process::makeMemoryless(meas_cpdf);
~~~

The function `ssmpack::process::makeMarkov()` receives an initial PDF and a CPDFs
and construct a Markovian random process of class `ssmpack::process::Markov`
from them.
The function `ssmpack::process::makeMemoryless()` takes a CPDFs
and returns a memoryless or independent random process of class `ssmpack::process::Memoryless`.

Finally, we can use the class `ssmpack::process::Hierarchical` to construct SSM
process.
~~~{.cpp}
// the hirerachical state space model
auto ssm_proc = ssmpack::process::makeHierarchical(state_proc, meas_proc);
~~~
The function `ssmpack::process::makeHierarchical()` takes an arbitrary number of processes
and construct a process of class `ssmpack::process::Hierarchical`. Hierarchical processes
are built by stacking other processes. The `Hierarchical` class connects the random variable of a process to the condition variable of the process in the lower level.

Data simulation
----------------


State estimation
-----------------
