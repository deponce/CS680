import numpy as np
def psudo_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
def psudo_div_r(a,b):
    return a/b if b!=0 else 1
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
        for iter in range(self.max_iter):
            for k in range(self.k):

                div_sqr = (x-self.Mu[k])**2
                index = np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0)
                index = index-np.min(index)
                tmp = np.ones_like(self.S[k])
                tmp[self.S[k] != 0] = 1 / self.S[k][self.S[k] != 0]
                R_ik[k] = self.Pi[k] * np.prod(
                                               tmp**0.5
                                               * np.exp(-0.5*index),
                                               axis=1)


                """
                div_sqr = (x-self.Mu[k])**2
                index = np.sum(np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0), 1)
                index = index-np.min(index)
                R_ik[k] = self.Pi[k]/(np.prod(self.S[k][self.S[k] != 0])**0.5) * np.exp(-0.5*index)
                """
            R_i = np.sum(R_ik, 0)
            R_ik = psudo_div(R_ik, R_i)
            self.l = np.append(self.l, -np.sum(np.log(R_i[R_i!=0])))
            if iter > 1 and np.abs(self.l[iter]-self.l[iter-1])<self.tol*np.abs(self.l[iter]):
                break
            for k in range(self.k):
                self.r[k] = np.sum(R_ik[k])
                self.Pi[k] = self.r[k] / n_data
                self.Mu[k] = np.matmul(R_ik[k], x) / self.r[k]
                self.S[k] = np.sum(np.expand_dims(R_ik[k], axis=0).T * (x - self.Mu[k])**2, 0)\
                            /self.r[k]
        return self.l

    def predict(self, x):
        n_data, dim_data = x.shape
        div = np.tile(x, self.k).reshape(n_data,self.k,dim_data)-self.Mu
        prob = np.exp(np.sum(-psudo_div(div**2, self.S), 2))
        prob = psudo_div(prob.T, np.sum(prob, 1))
        return prob
