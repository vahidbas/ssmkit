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
\f[
\mathbf{z}_k = \mathbf{H}\mathbf{x}_k+\boldsymbol\nu
\f]

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
