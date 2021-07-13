import numpy as np

class GMM:
    def __init__(self, k, max_iter, tol=1e-5):
        self.k = k
        self.S = None
        self.Mu = None
        _un_norm_Pi = np.random.random(self.k)
        self.Pi = _un_norm_Pi/np.sum(_un_norm_Pi)
        self.max_iter = max_iter
        self.r = np.zeros(self.k)
        self.l = np.array([])
        self.R_ik = None
        self.tol = tol
    def set_k(self,k):
        self.k = k
        self.r = np.ones(self.k)
        _un_norm_Pi = np.random.random(self.k)
        self.Pi = _un_norm_Pi/np.sum(_un_norm_Pi)

    def __psudo_div(self,a,b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def fit(self, x):
        n_data, dim_data = x.shape
        self.S = np.ones((self.k, dim_data))
        self.Mu = np.random.random((self.k, dim_data))  # means(centroids of clusters)
        self.R_ik = np.zeros((self.k, n_data))    # responsibility of each data point
        for iter in range(self.max_iter):
            for k in range(self.k):
                #if (0 in self.S[k]):
                 #   print("er")
                self.R_ik[k] = self.Pi[k]*(np.linalg.norm(self.S[k])**(-0.5)) * np.exp(-0.5*np.sum(self.__psudo_div((x-self.Mu[k])**2, self.S[k]), 1))
            R_i = np.sum(self.R_ik, 0)
            #if (np.NaN in R_i or 0 in R_i):
               # print("er")
            #self.R_ik.transpose()
            """
            np.tile(self.Pi*np.linalg.norm(self.S, axis=1)**(-0.5), n_data).reshape(self.k,n_data)*np.exp(-0.5*np.sum(np.tile(x, self.k).reshape((self.k, n_data, dim_data))-np.tile(self.Mu, n_data).reshape((self.k, dim_data, n_data)).transpose(0,2,1),2))
            """
            self.R_ik = self.__psudo_div(self.R_ik, R_i)
            self.l = np.append(self.l, -np.sum(np.log(R_i[R_i!=0])))
            if iter > 1 and np.abs(self.l[iter]-self.l[iter-1])<self.tol*np.abs(self.l[iter]):
                break

            for k in range(self.k):
                self.r[k] = np.sum(self.R_ik[k])
                self.Pi[k] = self.r[k] / n_data
                self.Mu[k] = self.__psudo_div(np.matmul(self.R_ik[k], x), self.r[k])
                #self.Mu[k] = np.matmul(self.R_ik[k], x)/self.r[k]
            # self.Mu = np.matmul(self.R_ik, x)/np.tile(self.r, dim_data).reshape((self.k, dim_data))
            #for k in range(self.k):
                #self.S[k] = np.sum(np.tile(self.R_ik[k],dim_data).reshape((n_data, dim_data))*(x-self.Mu[k])**2)/np.sum(self.R_ik[k])
                #self.S[k] = np.matmul(self.R_ik[k], x ** 2) / self.r[k] -(self.Mu[k]**2)
                self.S[k] = self.__psudo_div(np.matmul(self.R_ik[k], x ** 2), self.r[k]) - (self.Mu[k]**2)
                #while 0 in self.S[k]:
                 #   self.S[k] += 1e-15
        return self.l

    #def predict(self, x):

