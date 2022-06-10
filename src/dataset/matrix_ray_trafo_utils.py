import torch
from torch import nn
import scipy
import numpy as np

class MatrixRayTrafo:
    """
    Ray transform given by a (sparse) matrix.
    """
    def __init__(self, matrix, im_shape, proj_shape, order='C'):
        """
        Parameters
        ----------
        matrix : :class:`scipy.sparse.spmatrix` or array
            Matrix defining the mapping from images to projections.
            Must support matrix-vector multiplication via ``matrix.dot()``.
            Image and projection dimensions must be flattened in the specified
            `order`.
        im_shape : 2-tuple of int
            Image shape.
        proj_shape : 2-tuple of int
            Projection shape.
        order : {``'C'``, ``'F'``}, optional
            Order for reshaping images and projections from matrix shape to
            vector shape and vice versa.
            The default is ``'C'``.
        """
        self.matrix = matrix
        self.im_shape = im_shape
        self.proj_shape = proj_shape
        self.order = order

    def apply(self, x):
        """
        Apply the forward projection by (sparse) matrix multiplication.
        """
        x_flat = np.reshape(np.asarray(x), -1, order=self.order)
        y_flat = self.matrix.dot(x_flat)
        y = np.reshape(y_flat, self.proj_shape, order=self.order)
        return y

    def apply_adjoint(self, y):
        """
        Apply the adjoint by (sparse) matrix multiplication.
        """
        y_flat = np.reshape(np.asarray(y), -1, order=self.order)
        x_flat = self.matrix.T.dot(y_flat)
        x = np.reshape(x_flat, self.im_shape, order=self.order)
        return x


class MatrixModule(nn.Module):
    """
    Module applying (sparse) matrix-vector multiplication.
    """
    def __init__(self, matrix, out_shape, sparse=False):
        """
        Parameters
        ----------
        matrix : :class:`torch.Tensor`
            Tensor with two dimensions defining a linear mapping.
            Must be sparse if `sparse=True`.
        out_shape : sequence of int
            Output shape, excluding batch and channel dimensions.
        sparse : bool, optional
            Whether to use sparse matrix multiplication.
            Default: `True`.
        """
        super().__init__()
        self.register_buffer('matrix', matrix, persistent=False)
        self.out_shape = out_shape
        self.sparse = sparse

    def forward(self, inp):
        """
        Apply the forward projection by (sparse) matrix multiplication.

        Parameters
        ----------
        inp : :class:`torch.Tensor`
            Tensor of shape ``B x C x ...``.
        """
        inp_flat = inp.view(inp.shape[0] * inp.shape[1], -1)
        inp_flat = inp_flat.transpose(1, 0)
        if self.sparse:
            out_flat = torch.sparse.mm(self.matrix, inp_flat)
        else:
            out_flat = torch.matmul(self.matrix, inp_flat)
        out_flat = out_flat.transpose(1, 0)
        out = out_flat.view(inp.shape[0], inp.shape[1], *self.out_shape)
        return out

class tSVDMatrixModule(nn.Module):

    def __init__(self, tsvd_matrix, out_shape, adjoint=False):
        
        super().__init__()
        U, S, Vh = tsvd_matrix
        if adjoint:
            tsvd_matrix = (Vh, S, U)
            U, S, Vh = tsvd_matrix

        self.register_buffer('U', U, persistent=False)
        self.register_buffer('S', S, persistent=False)
        self.register_buffer('Vh', Vh, persistent=False)

        self.out_shape = out_shape

    def forward(self, inp):

        inp_flat = inp.view(inp.shape[0] * inp.shape[1], -1)
        inp_flat = inp_flat.transpose(1, 0)
        out_flat = self.U @ ( self.S[:, None] * (self.Vh.T @ inp_flat) )
        out_flat = out_flat.transpose(1, 0)
        out = out_flat.view(inp.shape[0], inp.shape[1], *self.out_shape)
        return out

def get_matrix_ray_trafo_module(matrix, im_shape, proj_shape, adjoint=False,
                                sparse=True):
    """
    Return a :class:`SparseMatrixModule` applying the ray transform given
    by a :class:`scipy.sparse.spmatrix`.

    Parameters
    ----------
    matrix : :class:`scipy.sparse.spmatrix` or array
        Matrix defining the mapping from images to projections.
        Image and projection dimensions must be flattened in ``'C'`` order.
        Must be a :class:`scipy.sparse.spmatrix` if `sparse=True`.
    im_shape : 2-sequence of int
        Image shape.
    proj_shape : 2-sequence of int
        Projection shape.
    adjoint : bool, optional
        Whether to return the adjoint instead of the forward ray transform.
        Default: `False`.
    sparse : bool, optional
        Whether to use sparse matrix multiplication.
        Default: `True`.

    Returns
    -------
    module : :class:`SparseMatrixModule`
        Module applying the forward projection.
    """
    if adjoint:
        matrix = matrix.T
    matrix = matrix.astype('float32')
    if sparse:
        matrix = matrix.tocoo()
        indices = torch.stack([torch.from_numpy(matrix.row),
                               torch.from_numpy(matrix.col)])
        values = torch.from_numpy(matrix.data)
        matrix_tensor = torch.sparse_coo_tensor(indices, values, matrix.shape)
        matrix_tensor = matrix_tensor.coalesce()
    else:
        if scipy.sparse.isspmatrix(matrix):
            matrix = matrix.todense()
        matrix_tensor = torch.from_numpy(matrix)
    out_shape = im_shape if adjoint else proj_shape
    module = MatrixModule(matrix_tensor, out_shape, sparse=sparse)
    return module
