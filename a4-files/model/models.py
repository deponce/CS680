import numpy as np
def psudo_div(a, b):
    zeros = np.zeros_like(a)
    infmask = (b == float("inf"))
    zeromask = (b == 0)
    #n_zeros = np.sum(zeromask)
    #zeros[:,zeromask] = np.ones((a.shape[0], n_zeros))/a.shape[0]
    mask = np.bitwise_not(np.bitwise_or(infmask, zeromask))
    zeros[:, mask] = np.divide(a[:, mask], b[mask], out=np.zeros_like(a[:, mask]), where=b[mask] != 0)
    zeros[(a == float("inf"))] = 1
    return zeros
def psudo_div_r(a,b):
    return a/b if b!=0 else 1

def normlize_col(arr):
    if arr.shape[0] ==1:
        return arr
    max = -np.linalg.norm(arr, float("inf"), axis=0)
    return arr/max

class GMM:
    def __init__(self, k, max_iter, tol=1e-5):
        self.k = k
        _un_norm_Pi = np.random.random(self.k)
        self.Pi = _un_norm_Pi/np.sum(_un_norm_Pi)
        self.max_iter = max_iter
        self.r = np.zeros(self.k)
        self.l = np.array([])
        self.tol = tol
        self.S = None
    def set_k(self,k):
        self.k = k
        self.r = np.ones(self.k)
        _un_norm_Pi = np.random.random(self.k)
        self.Pi = _un_norm_Pi/np.sum(_un_norm_Pi)
    def fit(self, x):
        n_data, dim_data = x.shape
        self.S = np.ones((self.k, dim_data))
        self.Mu = np.max(x)*np.random.random((self.k, dim_data))  # means(centroids of clusters)

        R_ik = np.zeros((self.k, n_data))    # responsibility of each data point
        log_R_ik = np.zeros((self.k, n_data))
        for iter in range(self.max_iter):
            for k in range(self.k):
                """
                div_sqr = (x-self.Mu[k])**2
                index = np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0)
                index = index-np.min(index)
                R_ik[k] = self.Pi[k] * np.prod(
                                               np.divide(1, self.S[k], out=np.ones_like(div_sqr), where=self.S[k] != 0)**0.5
                                               * np.exp(-0.5*index),
                                               axis=1)
                """
                div_sqr = (x-self.Mu[k])**2
                index = np.sum(np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0), 1)
                index = index - np.min(index)
                prex_divd = np.sum(np.log(self.S[k][self.S[k] != 0]))
                log_R_ik[k] = np.log(self.Pi[k])-0.5*prex_divd-0.5*index
            if dim_data>=1:
                log_R_ik = normlize_col(log_R_ik)
            R_ik = np.exp(log_R_ik)
            R_i = np.sum(R_ik, 0)
            R_ik = psudo_div(R_ik, R_i)
            self.l = np.append(self.l, -np.sum(np.log(R_i[R_i!=0])))
            if iter > 1 and np.abs(self.l[iter]-self.l[iter-1])<self.tol*np.abs(self.l[iter]):
                break
            for k in range(self.k):
                self.r[k] = np.sum(R_ik[k])
                self.Pi[k] = self.r[k] / n_data
                un_norm_mu = np.matmul(R_ik[k], x)
                self.Mu[k] = np.divide(un_norm_mu, self.r[k], out=np.zeros_like(un_norm_mu), where=self.r[k] != 0)
                un_norm_s = np.sum(np.expand_dims(R_ik[k], axis=0).T * (x - self.Mu[k]) ** 2, 0)
                #self.S[k] = np.sum(np.expand_dims(R_ik[k], axis=0).T * (x - self.Mu[k])**2, 0)/self.r[k]
                self.S[k] = np.divide(un_norm_s, self.r[k], out=np.zeros_like(un_norm_s), where=self.r[k] != 0)
        return self.l
    def predict(self, x):
        n_data, dim_data = x.shape
        div = np.tile(x, self.k).reshape(n_data,self.k,dim_data)-self.Mu
        prob = np.exp(np.sum(-psudo_div(div**2, self.S), 2))
        prob = psudo_div(prob.T, np.sum(prob, 1))
        return prob
""" 
index = np.sum(psudo_div((x-self.Mu[k])**2,self.S[k]), 1)
                R_ik[k] = self.Pi[k]*psudo_div_r(1, np.cumprod(self.S[k])[-1]**0.5) * \
                          np.exp(-0.5*index)
                """