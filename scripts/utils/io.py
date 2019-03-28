
import numpy as np


# Defining the Forward Step of the Algorithm
def forwardStep(outcomes, a, b, priors):

    # Setting up Alpha Vector with Zeros
    alpha = np.zeros((outcomes.shape[0], a.shape[0]))
    alpha_norm = np.zeros((outcomes.shape[0], a.shape[0]))

    # Initializing Alpha vectors at starting point
    alpha[0] = priors * b[:,outcomes[0]]

    alpha_norm[0] = alpha[0] / np.sum(alpha[0])

    # Induction Stept
    for t in range(outcomes.shape[0]-1):
        for i in range(a.shape[0]):
            alpha[t+1,i] = b[i,outcomes[t+1]] * alpha[t].dot(a[:,i])
        alpha_norm[t+1] = alpha[t+1] /np.sum(alpha[t+1])

    return alpha

def backwardStep(outcomes,a,b):
    # Initialize Betas
    beta = np.ones((outcomes.shape[0], a.shape[0]))

    # Backward Induction
    for t in range(outcomes.shape[0] -2,-1,-1):
        for i in range(a.shape[0]):
            beta[t,i] = (beta[t+1] * b[:,outcomes[t+1]]) @ a[i,:]

    return beta

def baumWelch(outcomes,a,b,priors, max_iters =100):

    T = len(outcomes)

    for r in range(max_iters):

        alpha = forwardStep(outcomes,a,b,priors)
        beta = backwardStep(outcomes,a,b)

        xi = np.zeros((a.shape[0],a.shape[0],T-1))

        for t in range(T-1):
            denominator = (alpha)[t,:] @ a * b[:, outcomes[t + 1]] @ beta[t + 1, :]
            for i in range(a.shape[0]):
                numerator = alpha[t, i] * a[i, :] * b[:, outcomes[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator


        gamma = np.sum(xi, axis = 1)

        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))


        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
             b[:, l] = np.sum(gamma[:, outcomes == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a , b
