
# Discovery test
# Author: A. Ramirez-Morales 
# Based on G. Cowan paper [Eur. Phys. J. C (2011) 71: 1554]

import numpy as np
import matplotlib.pyplot as plt


def f_q0_given_0(q0):
    # add documentation here ...
    q0 = np.asarray(q0)
    result = np.zeros_like(q0, dtype=np.float64)
    # first part, delta part for the Dirac delta  q0 == 0 we set f = 0.5 weight at 0
    delta_flag = (q0 == 0)
    result[delta_flag] = 0.5
    # second part, continue part
    cont_flag = (q0 > 0)
    q0_pos = q0[cont_flag]
    result[cont_flag] = 0.5 * (1 / np.sqrt(2 * np.pi * q0_pos)) * np.exp(-0.5 * q0_pos)
    return result
  

def q0_estimator(n, m, s, mu, tau, mu_hat, b_hat, b_hat_mu):
    # agregar documentacion [english]
    like_cond = ((mu * s + b_hat_mu) ** n * np.exp(-(mu * s + b_hat_mu))) * \
                ((tau * b_hat_mu) ** m * np.exp(-(tau * b_hat_mu)))
    
    like_max = ((mu_hat * s + b_hat) ** n * np.exp(-(mu_hat * s + b_hat))) * \
               ((tau * b_hat) ** m * np.exp(-(tau * b_hat)))
    
    if like_cond == 0 or like_max == 0:
        return 0.0
    
    return -2.0 * np.log(like_cond / like_max)

# parameters
b = 50
s = 1
n_events = 1000000
tau = 1
mu = 0  # for q0

# Monte Carlo simulation under H0 (mu = 0)
# that is, we are assuming that in the data there is no signal f(q0|0)
np.random.seed(42)
n_obs = np.random.poisson(lam=mu*s + b, size=n_events)
m_obs = np.random.poisson(lam=tau * b,  size=n_events)
q0_mc = np.zeros(n_events)

for i in range(n_events):
    n = n_obs[i]
    m = m_obs[i]

    mu_hat = max(0.0, (n - m / tau) / s)
    b_hat = m / tau

    first_term = (n + m - (1 + tau) * mu * s) / (2 * (1 + tau))
    second_term = ((n + m - (1 + tau) * mu * s) ** 2 + 4 * (1 + tau) * m * mu * s) / (4 * (1 + tau) ** 2)
    b_hat_mu = first_term + np.sqrt(second_term)

    if mu_hat > 0:
        q0_mc[i] = q0_estimator(n, m, s, mu, tau, mu_hat, b_hat, b_hat_mu)


# plot the MC simulated distribution
bins = np.linspace(0, 40, 30)
hist, edges = np.histogram(q0_mc, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])
plt.figure(figsize=(8, 5))
plt.step(centers, hist, where='mid', label='MC (Poisson toys)', color='green')

# asymptotic theory distribution  [Cowan]
q0_vals = np.linspace(0.001, 40, 1000)
pdf_theory = f_q0_given_0(q0_vals)
plt.plot(q0_vals, pdf_theory, 'b-', label='Theoretical $f(q_0|0)$')

# plots to compare both 
plt.yscale('log')
plt.ylim((1e-8,10))
plt.xlabel(r'$q_0$')
plt.ylabel('Density (log scale)')
plt.title('Comparison of MC vs Theoretical Distribution of $q_0$ (with complementary data)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
