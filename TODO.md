This TODO list reflects some desirable features and algorithms to be implemented in SSMKIT.

### Probability Distributions ###
- [x] Gaussian
- [x] Categorical
- [ ] Dirichlet: useful as conjugate prior of `Categorical`
- [ ] Beta
- [ ] Wishart: useful as conjugate prior of `Gaussian`

### Filtering Algorithms ###
- [x] Kalman filter
- [ ] Kalman smoother: perhaps RTS
- [ ] Extended KF
- [ ] Unscented KF
- [x] Particle filter
- [ ] Auxiliary Particle filter
- [ ] Particle smoother
- [ ] HMM: forward-backward Algorithms
- [ ] Rao-Blackwellized Particle filter
- [ ] Markov jump particle filter

### Learning (Identification) Algorithms ###
- [ ] Strovik's filter
- [ ] Extended Parameter filter
- [ ] Subspace System Identification
- [ ] Gaussian Process SSM
- [ ] SMC^2
- [ ] Liu-West filter
- [ ] Baum–Welch algorithm

### Performance Evaluation ###
Classes for evaluating performances of filtering and learning
algorithms
- [ ] MSE (filtering)
- [ ] observation likelihood (filtering)
