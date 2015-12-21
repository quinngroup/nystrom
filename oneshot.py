import numpy as np
import random 
import utils 

N = 313
D = 11
M = 150
SCALE = 1000.0

X = np.matrix(np.random.normal(size = [D,N])*SCALE)  # create a dataset of size N and D dimensions with values in [0,SCALE)
sample = random.sample(range(N),M)  # choose M out of N for sample
W = X[:,sample] #extract sample matrix W
kernel_func = lambda x1,x2: utils.rbf_kernel(x1,x2,gamma = 0.05)   # define kernel function
#kernel_func = utils.dot_kernel

#sample by sample kernel matrix (MxM kernel matrix)
Kw = utils.construct_kernel_matrix(W,W, kernel = kernel_func) #W.T*W for dot product

#whole  by sample kernel matrix (NxM truncated kernel matrix)
C  = utils.construct_kernel_matrix(X,W, kernel = kernel_func) #X.T*W




U_kw,S_kw,Vt_kw = np.linalg.svd(Kw,full_matrices = False)  #decompose Kw
S_kw_pinv = np.diag(utils.vector_pinv(S_kw)) #compute the moore-penrose pseudoinverse of S_kw

#compute the building block of Knys, the G matrix = C * V_w * S_w^-1
#where Knys = G*G.T = C*pinv(Kw)*C.T
#note that V_w = V_kw = Vt_kw.T
#and S_w = sqrt(S_kw)
#so S_w^-1 = sqrt(pinv(S_kw)) = sqrt(S_kw_pinv) computed by the previous code line
G=C*Vt_kw.T*np.sqrt(S_kw_pinv)	
								
U_g,S_g,Vt_g = np.linalg.svd(G,full_matrices = False)
Vnys = G*Vt_g.T*np.diag(utils.vector_pinv(S_g))   #Vnys = U_g = G*V_g*S^-1


#four ways to construct the kernel matrix now
Knys1 = G*G.T                  
Knys2 = C*Vt_kw.T*S_kw_pinv*U_kw.T*C.T		# Knys = C*pinv(Kw)*C.T ,,, pinv(Kw) = V_kw* S_kw^-1 * U_kw.T
Knys3 = Vnys*np.diag(S_g*S_g)*Vnys.T		# Knys = Vnys * Sg^2 * Vnys.T	
Knys4 = U_g*np.diag(S_g*S_g)*U_g.T


#whole kernel matrix that we want to approximate (NxN), computed for comparison here
Kx = utils.construct_kernel_matrix(X,X, kernel = kernel_func) #X.T*X
print(np.linalg.norm(Knys1 - Kx))
print(np.linalg.norm(Knys2 - Kx))
print(np.linalg.norm(Knys3 - Kx))
print(np.linalg.norm(Knys4 - Kx))


