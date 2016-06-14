Tutorial {#tutorial}
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


