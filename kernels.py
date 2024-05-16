import concurrent
import copy
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, List
import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sklearn.metrics

Tensor = torch.Tensor

def kernel_factory(name: str, param: dict):
    assert name in ["rbf", "linear", "normpoly", "poly", "laplacian"]
    kernel = None
    if name == "rbf":
        kernel = GaussianKernelTorch(**param)
    elif name == "normpoly":
        kernel = NormPolyKernelTorch(**param)
    elif name == "poly":
        kernel = PolyKernelTorch(**param)
    elif name == "linear":
        kernel = LinearKernel()
    elif name == "laplacian":
        kernel = LaplacianKernelTorch(**param)
    return kernel

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.mm(X.t(), Y)

class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(GaussianKernelTorch, self).__init__()
        if type(sigma2) == float:
            self.sigma2 = Parameter(torch.tensor(float(sigma2)), requires_grad=False)
            self.register_parameter("sigma2", self.sigma2)
        else:
            self.sigma2 = sigma2

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes a matrix of the norm of the difference.
            """
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            return res

        D = my_cdist(X,Y)

        return torch.exp(- torch.pow(D, 2) / (self.sigma2))

class LaplacianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(LaplacianKernelTorch, self).__init__()
        if type(sigma2) == float:
            self.sigma2 = Parameter(torch.tensor(float(sigma2)), requires_grad=False)
            self.register_parameter("sigma2", self.sigma2)
        else:
            self.sigma2 = sigma2

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        X, Y = X.t(), Y.t()
        # K = sklearn.metrics.pairwise.laplacian_kernel(X, Y, gamma=(1/self.sigma2).item())
        res = torch.cdist(X, Y, p=1.0)

        return torch.exp(- res / (self.sigma2))



def laplacian_kernel(X, Y=None, gamma=None):
    """Compute the laplacian kernel between X and Y.
    The laplacian kernel is defined as::
        K(x, y) = exp(-gamma ||x-y||_1)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <laplacian_kernel>`.
    .. versionadded:: 0.17
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.
    Y : ndarray of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.
    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        The kernel matrix.
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = -gamma * manhattan_distances(X, Y)
    np.exp(K, K)  # exponentiate K in-place
    return K



class PolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=1.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.pow(torch.matmul(X.t(), Y), self.d)

class NormPolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=1.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        X = X.t()
        Y = Y.t()
        D1 = torch.diag(1. / torch.sqrt(torch.pow(torch.sum(torch.pow(X, 2), dim=1) + self.c ** 2, self.d)))
        D2 = torch.diag(1. / torch.sqrt(torch.pow(torch.sum(torch.pow(Y, 2), dim=1) + self.c ** 2, self.d)))
        return torch.matmul(torch.matmul(D1, torch.pow((torch.mm(X, Y.t()) + self.c**2), self.d)), D2)