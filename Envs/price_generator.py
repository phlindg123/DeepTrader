import numpy as np

def make_stocks(length=1000, num_stocks=2):
    alpha = 0.9
    k = 2
    cov = np.random.normal(0, 5, [num_stocks, num_stocks])
    cov = cov.dot(cov.T)
    A = np.random.multivariate_normal(np.zeros(num_stocks), cov, size=[length])
    B = np.random.multivariate_normal(np.zeros(num_stocks), cov, size=[length])
    bs = [np.zeros(num_stocks)]
    ps = [np.zeros(num_stocks)]
    for a, b in zip(A,B):
        bv = alpha*bs[-1] + b
        bs.append(bv)
        pv = ps[-1] + bs[-2] + k*a
        ps.append(pv)
    ps = np.array(ps)
    R = ps.max(0) - ps.min(0)
    prices = np.exp(ps/(R))*np.random.uniform(10, 250, num_stocks)
    return prices