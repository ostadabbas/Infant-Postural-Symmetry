import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# load annotations
data = pd.read_excel ('./data/survey_data/asymmetryanalysisAll.xlsx')
df = pd.DataFrame(data, columns=['Rater1Symmetry', 'Rater2Symmetry', 'Rater3Symmetry','Rater4Symmetry' ,'Rater5Symmetry',
                                 'Rater6Symmetry', 'Rater7Symmetry', 'Rater8Symmetry','Rater9Symmetry' ,'Rater10Symmetry'])
y = df.to_numpy()
print(y.shape)

R = 10
N = 700*4
n_iters = 10
thres = 0.5
# assuming beta parameters
# a1, a2 of alpha for each rater
a1 = np.zeros((1, R))
a1[:,:] = 2
a2 = np.zeros((1, R))
a2[:,:] = 2
# b1, b2 of beta for each rater
b1 = np.zeros((1, R))
b1[:,:] = 2
b2 = np.zeros((1, R))
b2[:,:] = 2
# p1, p2 of prevalence
p1 = 2
p2 = 2

# initialization
# initialize mu for each instance
mu = np.mean(y, axis = 1)
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
        beta[:, j] = (b1[:, j] - 1 + np.sum(np.multiply((1 - mu), (1 - y[:, j])))) / (b1[:, j] + b2[:, j] - 2 + np.sum((1 - mu)))
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

        # save estimated annotation
        est_anno = np.transpose(np.array([mu, anno]))
        np.savetxt("./outputs/symmetry_estimated_annotations.csv", est_anno, delimiter=',', header="mean,estimation", comments="")

        break

    p_tmp = p.copy()
    alpha_tmp = alpha.copy()
    beta_tmp = beta.copy()

