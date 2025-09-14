import tensorly as tl
import numpy as np
from tensorly.tenalg import svd_interface

# Direct implementation of https://github.com/tensorly/Proceedings_IEEE_companion_notebooks/blob/master/MDA.ipynb encapsulated
class MultilinearDiscriminantAnalysis:
    
    def __init__(self, total_iters=5, rank=16):
        tl.set_backend('numpy')
        self.rank = rank
        self.total_iters = total_iters
        self.factors = None
        self.global_mean = None
        self.class_means = None
        self.people_idx = None
        self.num_classes = None
        self.num_modes = None
        self.num_each_class = None

    # directly from original work    
    def compute_mode_k_scatters(self, mode, factors, X_train, y_train):
        B_scat = 0
        W_scat = 0

        for c in range(self.num_classes):
            M = self.class_means[c] - self.global_mean
            proj_but_k = tl.unfold(tl.tenalg.multi_mode_dot(M, factors, transpose=True, skip=mode), mode)
            B_scat += self.num_each_class[c] * tl.dot(proj_but_k, proj_but_k.T)

            # intra-class
            for j in range(self.num_each_class[c]):
                M = X_train[self.people_idx[c][j]] - self.class_means[c]

                proj_but_k = tl.unfold(tl.tenalg.multi_mode_dot(M, factors, transpose=True, skip=mode), mode)
                W_scat += tl.dot(proj_but_k, proj_but_k.T)
                
        return B_scat, W_scat

    """
    Changed partial_svd to svd_interface as Tensorly no longer supports partial_svd
    i changed it to build class_means instead of concatenating a list within a loop
    apart from that is exact logic from original work
    """
    def fit(self, X, y):
        # Convert to tensors
        X_train = tl.tensor(X)
        y_train = tl.tensor(y)
        
        self.global_mean = tl.mean(X_train, axis=0)
        
        # Store the indices of training data that belong to each identity
        unique_labels = np.unique(y_train)
        self.people_idx = [[] for _ in range(len(unique_labels))]
        
        for i, identity in enumerate(y_train):
            # store the person's id label
            self.people_idx[identity] += [i]
        
        self.class_means = [tl.mean(X_train[person_idx], axis=0) 
                   for person_idx in self.people_idx]
        
        self.num_classes = len(self.class_means)
        self.num_modes = len(X_train.shape) - 1
        self.num_each_class = [len(p) for p in self.people_idx]
        
        self.factors = [tl.ones((dim, self.rank)) for i, dim in enumerate(list(X_train.shape)[1:])]
        
        # iteration logic from notebook
        for t in range(1, self.total_iters + 1):
            for k in range(self.num_modes):
                B_scat, W_scat = self.compute_mode_k_scatters(k, self.factors, X_train, y_train)
                
                # Use svd_interface instead of deprecated partial_svd
                U = svd_interface(tl.dot(tl.tensor(np.linalg.inv(W_scat)), B_scat), n_eigenvecs=self.rank)[0]
                self.factors[k] = U
        return self
    
    def transform(self, X):
        X_tensor = tl.tensor(X)
        
        # from original except uses all modes rather than 2D tensors
        modes_to_use = list(range(1, self.num_modes + 1))
        Z = tl.tenalg.multi_mode_dot(X_tensor, self.factors, modes=modes_to_use, transpose=True)
        
        # flattened numpy array
        n_samples = Z.shape[0]
        return Z.reshape(n_samples, -1)

    # shortform
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)