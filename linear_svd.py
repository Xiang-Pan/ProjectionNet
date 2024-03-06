import torch
import numpy as np
import logging
logger = logging.getLogger(f"./logs/{__name__}.log")

# Normalize V so we don't need to divide by norm. 
def normalize(V):   
	d	  = V.shape[0]
	norms   = torch.norm(V, 2, dim=1)
	V[:,:]  = V / norms.view(d, 1)
	return norms

# fasthpp function as provided
# New algorithm with O(d/t + log2(t)) operations. 
def fasthpp(V, X, stop_recursion=3): 
    """
        V: matrix that represent weights of householder matrices (d, d)
        X: rectangular matrix (d, bs) to compute H(V) @ X
        stop_recursion: integer that controls how many merge iterations before recursion stops. 
                        if None recursion continues until base case. 
    """
    d = V.shape[0]

    with torch.cuda.device_of(V):
        Y_ = V.clone().T
        W_ = -2*Y_.clone()

    # Only works for powers of two. 
    assert (d & (d-1)) == 0 and d != 0, "d should be power of two. You can just pad the matrix. " 

    # Step 1: compute (Y, W)s by merging! 
    k = 1
    for i, c in enumerate(range(int(np.log2(d)))):  
        k_2 = k 
        k  *= 2

        m1_ = Y_.view(d//k_2, k_2, d)[0::2] @ torch.transpose(W_.view(d//k_2, k_2, d)[1::2], 1, 2)
        m2_ = torch.transpose(W_.view(d//k_2, k_2, d)[0::2], 1, 2) @ m1_

        W_ = W_.view(d//k_2, k_2, d)
        W_[1::2] += torch.transpose(m2_, 1, 2)
        W_ = W_.view(d, d)

        if stop_recursion is not None and c == stop_recursion: break

    # Step 2: 
    if stop_recursion is None:   return X + W_.T @ (Y_ @ X) 
    else: 
        # For each (W,Y) pair multiply with 
        for i in range(d // k-1, -1, -1 ):
            X = X + W_[i*k: (i+1)*k].T @ (Y_[i*k: (i+1)*k]  @ X )
        return X 

# Orthogonal class using fasthpp
class Orthogonal(torch.nn.Module):
    def __init__(self, d, device="cuda"):
        super(Orthogonal, self).__init__()
        self.V = torch.zeros((d, d)).normal_(0, 1)
        normalize(self.V.T)  # Assuming normalize function is defined
        self.V = self.V.to(device)

    def forward(self, X):
        # fasthpp X: (d, bs)
        return fasthpp(self.V, X, stop_recursion=4)

# LinearSVD class
class LinearSVD(torch.nn.Module): 
    def __init__(self, d): 
        super(LinearSVD, self).__init__()
        self.d = d
        self.U = Orthogonal(d)
        self.D = torch.empty(d, 1).uniform_(0.99, 1.01)
        self.V = Orthogonal(d)

    def forward(self, X):
        X = self.U(X)
        X = self.D * X 
        X = self.V(X)
        return X 

if __name__ == "__main__":
    # Example usage
    d = 512
    bs = 32
    neuralSVD = LinearSVD(d=d, m=64)
    X = torch.zeros(d, bs).normal_()
    result = neuralSVD(X)
    print(result.shape)