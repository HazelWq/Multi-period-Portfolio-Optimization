import pandas as pd
import numpy as np
import cvxpy as cp

def optimization7(pi_pre, pi_t_1, Sigma_t, n, H, gamma_trade):
    pi_t = cp.Variable((n,H))
    I = np.eye(n)
    ones = np.ones((n,1))
    constraints = [pi_t >= 0]
    obj = 0
    g_t = np.zeros((n,H))
    for h in range(H):
        Sigma_tau = Sigma_t[h,:,:]
        pi_tau = pi_t[:,h].reshape((-1,1))
        pi_tau_1 = pi_t_1[:,h].reshape((-1,1))
        delta_tau = np.trace(Sigma_tau)/40/n
        total = pi_tau_1.T@Sigma_tau@pi_tau_1

        g_tau = np.multiply(pi_tau_1, Sigma_tau@pi_tau_1)/total
        g_t[:,h] = g_tau.reshape(n)

        #A_tau = (np.diag(Sigma_tau@pi_tau_1)+Sigma_tau@np.vstack([pi_tau_1.T]*n))/total\
        #        -2*np.multiply(pi_tau_1, Sigma_tau@pi_tau_1)@pi_tau_1.T@Sigma_tau/total**2
        A_tau = (np.diag(Sigma_tau @ pi_tau_1) + n * pi_tau_1@ones.T@Sigma_tau
                - 2 * n * g_tau @ pi_tau_1.T @ Sigma_tau) / total

        Q_tau = 2*A_tau.T@A_tau+delta_tau*I
        q_tau = 2*A_tau.T@g_tau-Q_tau@pi_tau_1
        if h == 0:
            pi_p = pi_pre
        else:
            pi_p = pi_t[:,h-1].reshape((-1,1))
        obj += 1/2*cp.quad_form(pi_tau,Q_tau) + pi_tau.T@q_tau + gamma_trade*cp.norm(pi_tau-pi_p,1)

        constraints.append(cp.sum(pi_tau)==1)
        constraints.append(pi_tau < 0.5)

    prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
    prob.solve(solver='OSQP')

    return pi_t.value, g_t


def Algorithm1(pi_pre, Sigma, gamma_trade, H, tol=0.001):
    k = 0 # iteration
    gamma_k = 0.8 # gamma_0 in [0,1] adjustment speed

    n = pi_pre.shape[0]
    tol_obj_1 = 1
    pi = np.array([1 / n] * n * H).reshape(n, H)
    pi_k = pi.copy()

    while True:
        # set max iteration
        if k > 1000:
            print('Not converged')
            print(tol_obj, abs(tol_obj_1))
            break
        # Solve Problem 7 and get optimal solution
        pi_opt, g = optimization7(pi_pre, pi_k, Sigma, n, H, gamma_trade)

        # calculate tolerance of objective
        tol_obj = g - pi
        tol_obj = tol_obj ** 2
        tol_obj = sum(sum(tol_obj))
        if abs(tol_obj_1-tol_obj) <= tol:
            print(f'converged in {k} iterations.')
            break
        tol_obj_1 = tol_obj

        #update pi_k
        pi_k = pi_opt
        # update gamma_k+1
        gamma_k = 1 - 0.0000001*gamma_k

        k += 1

    return pi_k


"""n = 10
H = 5
Sigma = np.zeros((H,n,n))
np.random.seed(42)
for h in range(H):
    sigma = np.random.randn(10,10)
    Sigma[h,:,:] = np.cov(sigma)
    # Sigma[h, :, :] = np.diag([2,0.5,0.9,0.1,8, 0.5,0.6,0.7,0.2,1])
pi_p = np.array([2/n,0]*int(n/2)).reshape(n,1)
print(pi_p)
a = Algorithm1(pi_p, Sigma, H=H, gamma_trade=0.001)
print(a)"""