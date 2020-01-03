# ZCA and MeanOnlyBNLayer implementations copied from
#   https://github.com/TimSalimans/weight_norm/blob/master/nn.py
#
# Modifications made to MeanOnlyBNLayer:
# - Added configurable momentum.
# - Added 'modify_incoming' flag for weight matrix sharing (not used in this project).
# - Sums and means use float32 datatype.

import numpy as np
import theano as th
import theano.tensor as T
from scipy import linalg
import lasagne

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = th.shared(np.dot(tmp, U.T).astype(th.config.floatX))
        self.inv_ZCA_mat = th.shared(np.dot(tmp2, U.T).astype(th.config.floatX))
        self.mean = th.shared(m.astype(th.config.floatX))

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return T.dot(x.flatten(2) - self.mean.dimshuffle('x',0), self.ZCA_mat).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
            
    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0],np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x',0)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
