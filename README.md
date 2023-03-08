# Switching GPLFM for discontinuous nonlinear system identification

The switching Gaussian process latent force model (GPLFM) is a physics-enhanced machine learning tool developed for performing nonlinear system identification in dynamical systems characterised by discontinuous nonlinearities and/or the presence of different motion regimes. Given a set of noisy experimental observations of the system response and a known input function, the switching GPLFM computes:
- the posterior distribution of the latent states of the system
- the posterior distribution of the latent nonlinear force 
- the sequence of the motion regimes (including the occurrence of discontinuities in the nonlinear force)

In addition, further steps can be operated within the switching GPLFM framework to:
- characterise the functional form of the identified nonlinear force
- estimate the uncertain parameters of the system

Further information about the switching GPLFM implementation and its application can be found in the corresponding journal paper [[1](https://github.com/l-marino/switching-GPLFM/edit/main/README.md#references)].
## Content and instructions

This repository contains two examples of applications of the switching GPLFM:
1. A simulated randomly-excited dry friction oscillator 
2. A harmonically base-excited single-storey frame with a brass-to-steel contact

Each of the corresponding folders includes the following files:
- The script `switching_GPLFM_main.m` is the main file. It includes the simulation/uploading of the measurements, the parameter estimation and the nonlinear force characterisation procedures. It is also where the main settings of the switching GPLFM (number of models, Gaussian mixture components, etc) can be selected by the user.
- The function `Switching_GPLFM_VBMC.m` is where settings regarding the inference of GP hyperparameters can be adjusted, including priors, as well as lower and upper bounds for the posterior distribution. The posterior of the hyperparameters is inferred by using the Variational Bayes Monte Carlo method developed by Luigi Acerbi  [[2](https://github.com/l-marino/switching-GPLFM/edit/main/README.md#references), [3](https://github.com/l-marino/switching-GPLFM/edit/main/README.md#references)], whose codes are included in the folder **vbmc-master** and also available [here](https://github.com/acerbilab/vbmc).
- The function `Switching_GPLFM_ADF_EC.m` is the core of the method. The posterior distribution of the latent states and nonlinear force are computed via assumed density filtering (ADF) and an expectation-correction (EC) smoothing algorithm. This approach is strongly based on the Switching Linear Dynamical Systems (SLDS) theory from David Barber [[4](https://github.com/l-marino/switching-GPLFM/edit/main/README.md#references)], whose codes are included in the folder **slds** and also available [here](http://web4.cs.ucl.ac.uk/staff/D.Barber/software/slds.zip). The state-space representation of the GP latent force is obtained, for different kernel functions, by using Solin and Särkkä's approach [[5](https://github.com/l-marino/switching-GPLFM/edit/main/README.md#references)]. Their codes are included in the folder **kernels** and are also available [here](https://users.aalto.fi/~asolin/documents/pdf/Solin-Sarkka-2014-AISTATS-code.zip).
- The function `Friction_SDOF.m` can be used for simulating the response of a single degree-of-freedom mass-spring-damper system with a friction contact to a given input forcing. The forcing function and a rate-dependent friction law can be defined by the user, along with initial conditions, simulation time, sampling frequency and additional white noise.


## References

1. Marino, L., Cicirello, A. (2023) A switching Gaussian process latent force model for the identification of mechanical systems with a discontinuous nonlinearity. *arXiv preprint arXiv:2303.03858.
2. Acerbi, L. (2018). Variational Bayesian Monte Carlo. In: *Advances in Neural Information Processing Systems 31*: 8222-8232.
3. Acerbi, L. (2020). Variational Bayesian Monte Carlo with Noisy Likelihoods. In: *Advances in Neural Information Processing Systems 33*: 8211-8222.
4. Barber, D. (2006) Expectation Correction for smoothed inference in switching linear dynamical systems. *Journal of Machine Learning Research 7*: 2515-2540.
5. Solin, A., Särkkä, S. (2014) Explicit link between periodic covariance functions and state space models. In: *Proceedings of
the 17th International Conference on Artificial Intelligence and Statistics 33*, 904–912.
Please cite the above references if you use the switching GPLFM in your work. You can also demonstrate your appreciation by *starring* the [switching-GPLFM](https://github.com/l-marino/switching-GPLFM) repository on GitHub.
