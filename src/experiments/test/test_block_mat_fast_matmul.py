import torch
import numpy as np
from opt_einsum import contract



##### SLOW METHOD: #####
# A is a 3*6 random matrix]
N = 1 # dim_output
P = 6 # dim_params 
A = torch.zeros((N,P)).normal_(0, 1) # jac

# B is a 6*6 block-diagonal matrix with blocks of 2*2 (a.k.a. P^K, P^K where P^K is dim of params in conv kernel) Sigma_theta
B = torch.zeros((P,P))
B[0:2,0:2] = torch.tensor([[1,2],[3,4]])
B[2:4,2:4] = torch.tensor([[5,6],[7,8]])
B[4:6,4:6] = torch.tensor([[9,10],[11,12]])

# multiply A and B to obtain C
C = A @ B

print(C)
print('****************************************************')

##### FAST METHOD: #####
# divide A into many smaller matrices by slicing vertically;
# each smaller matrix has the same number of coumns as the blocks in B,
# and the same number of rows as the original A
R = 3
A_reshaped = A.view(N,R,2).permute(1,0,2) 

# store B as a series of small matrices, where each small matrix is a block
# from the original B
B_reshaped = torch.zeros((R,2,2))
B_reshaped[0] = torch.tensor([[1,2],[3,4]])
B_reshaped[1] = torch.tensor([[5,6],[7,8]])
B_reshaped[2] = torch.tensor([[9,10],[11,12]])

# matrix multiplication; compare with the original C
C_reshaped = A_reshaped @ B_reshaped
C_reshaped = contract('nxk,nkc->ncx', A_reshaped, B_reshaped)
C_reshaped = C_reshaped.reshape([P, N]).t()

print(C_reshaped)
print('****************************************************')

# this is the method used in BlocksGPpriors
def matrix_prior_cov_mul(x, idx=None):
    N = x.shape[0]
    x = x.view(-1, B_reshaped.shape[0], B_reshaped.shape[-1])
    x = x.permute(1, 0, 2)
    out = x @ B_reshaped
    out = out.permute(0, 2, 1).reshape([B_reshaped.shape[0]
            * B_reshaped.shape[-1], N]).t()
    return out

out = matrix_prior_cov_mul(A)
print(out)
print('****************************************************')

# print(A) # Jac 
# print(A.shape)
# print('****************************************************')

# print(B) # cov 
# print(B.shape)
# print('****************************************************')

