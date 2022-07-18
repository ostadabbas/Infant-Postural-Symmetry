import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def binary_em(y):
    R = 10
    N = 700 * 4
    n_iters = 10
    thres = 0.5
    # assuming beta parameters
    # a1, a2 of alpha for each rater
    a1 = np.zeros((1, R))
    a1[:, :] = 5
    a2 = np.zeros((1, R))
    a2[:, :] = 1
    # b1, b2 of beta for each rater
    b1 = np.zeros((1, R))
    b1[:, :] = 5
    b2 = np.zeros((1, R))
    b2[:, :] = 1
    # p1, p2 of prevalence
    p1 = 5
    p2 = 1

    # initialization
    # initialize mu for each instance
    mu = np.mean(y, axis=1)
    mu_old = mu.copy()

    alpha = np.zeros((1, R))
    beta = np.zeros((1, R))
    alpha_tmp = np.zeros((1, R))
    beta_tmp = np.zeros((1, R))
    a = np.ones((N, 1))
    b = np.ones((N, 1))
    p = 0
    p_tmp = 0
    tol = 0.0001
    n_stopped = n_iters
    alpha_errors = []
    beta_errors = []
    for iter in range(n_iters):
        print(iter)
        # E-step
        for j in range(R):
            alpha[:, j] = (a1[:, j] - 1 + np.sum(np.multiply(mu, y[:, j]))) / (a1[:, j] + a2[:, j] - 2 + np.sum(mu))
            beta[:, j] = (b1[:, j] - 1 + np.sum(np.multiply((1 - mu), (1 - y[:, j])))) / (
                        b1[:, j] + b2[:, j] - 2 + np.sum((1 - mu)))
        p = (p1 - 1 + np.sum(mu)) / (p1 + p2 - 2 + N)

        # M-step
        for i in range(N):
            for j in range(R):
                a[i, :] *= pow(alpha[:, j], y[i, j]) * pow((1 - alpha[:, j]), (1 - y[i, j]))
                b[i, :] *= pow(beta[:, j], (1 - y[i, j])) * pow((1 - beta[:, j]), y[i, j])

            mu[i] = (a[i, :] * p) / (a[i, :] * p + b[i, :] * (1 - p))


        alpha_MSE = np.square(np.subtract(alpha, alpha_tmp)).mean()
        beta_MSE = np.square(np.subtract(beta, beta_tmp)).mean()

        alpha_errors.append(alpha_MSE)
        beta_errors.append(beta_MSE)

        if alpha_MSE < tol and beta_MSE < tol:
            n_stopped = iter + 1
            print(f'Iteration {iter}:')
            print(f'Prevalence p = {p}')
            print('Sensitivity Alpha:')
            print(alpha)
            print('Specificity Beta:')
            print(beta)
            print(f'When threshold = {thres}, the estimated annotation is:')

            anno = np.where(mu > thres, 1, 0)
            print(anno)

            return mu

        p_tmp = p.copy()
        alpha_tmp = alpha.copy()
        beta_tmp = beta.copy()


# load annotations
data = pd.read_excel ('./data/survey_data/asymmetryanalysisAll.xlsx')
df = pd.DataFrame(data, columns=['Rater1AngelRange', 'Rater2AngelRange', 'Rater3AngelRange','Rater4AngelRange' ,'Rater5AngelRange',
                                 'Rater6AngelRange', 'Rater7AngelRange', 'Rater8AngelRange','Rater9AngelRange' ,'Rater10AngelRange'])
y = df.to_numpy()

# convert three ordinal data to two binary data
# when c = 1
c = 1
array1 = y > c
y_1 = array1.astype(int)

# when c = 2
c = 2
array2 = y > c
y_2 = array2.astype(int)

# Pr(y > 1)
mu_1 = binary_em(y_1)

# Pr(y > 2)
mu_2 = binary_em(y_2)

# Pr(y = 2)
mu2 = mu_1 - mu_2

# Pr(y = 3)
mu3 = mu_2

# Pr(y = 1)
mu1 = 1 - mu2 - mu3


mu = pd.DataFrame(
    {1: mu1.tolist(),
     2: mu2.tolist(),
     3: mu3.tolist()
    })
df = mu.round(2)
label = df.idxmax(axis = 1)

df.insert(3, "class", label, True)

df.insert(4, "mu_1", mu_1.round(2), True)
df.insert(5, "mu_2", mu_2.round(2), True)
df.to_csv("./outputs/angle_estimated_annotations.csv")

