from oneshot import *


def double_nystrom(X,s,m,l,k = -1, kernel_func = utils.dot_kernel):
	D,N = X.shape #dataset X is of size N and D dimensions
	sample = random.sample(range(N),s)  # choose m out of N for sample
	S = X[:,sample] #extract sample matrix S
	
	#whole  by sample kernel matrix (Nxs truncated kernel matrix)
	C0  = utils.construct_kernel_matrix(X,S, kernel = kernel_func) #X.T*S
	
	#apply one-shot nystrom with m samples on S to obtain the decomposition of Ks and retain only l eigenvectors
	V_s_l, _ = one_shot_nystrom(S,m,l,kernel_func)
	
	C = C0*V_s_l  #project C0 onto the retained eigenvectors of Vs,l
	
	#sample by sample kernel matrix (sxs kernel matrix)
	Ks = utils.construct_kernel_matrix(S,S, kernel = kernel_func) #S.T*S for dot product
	Kw = V_s_l.T*Ks*V_s_l #construct Kw by trimming Ks  
	
	#rest is oneshot nystrom on Kw and C, retaining possibly k 
	

	U_kw,S_kw,Vt_kw = utils.svd(Kw,full_matrices = False)  #decompose Kw
	S_kw_pinv = np.diag(utils.vector_pinv(S_kw)) #compute the moore-penrose pseudoinverse of S_kw

	#compute the building block of Knys, the G matrix = C * V_w * S_w^-1
	#where Knys = G*G.T = C*pinv(Kw)*C.T
	#note that V_w = V_kw = Vt_kw.T
	#and S_w = sqrt(S_kw)
	#so S_w^-1 = sqrt(pinv(S_kw)) = sqrt(S_kw_pinv) computed by the previous code line
	G=C*Vt_kw.T*np.sqrt(S_kw_pinv)	
									
	U_g,S_g,Vt_g = utils.svd(G,full_matrices = False)
	Vnys = G*Vt_g.T*np.diag(utils.vector_pinv(S_g))   #Vnys = U_g = G*V_g*S^-1
	if k > 0 :
		S_nys_k = S_g[:k]
		return Vnys[:,:k],S_nys_k*S_nys_k
	return Vnys,S_g*S_g #,Knys1,Knys2,Knys3,Knys4
	
if __name__ == '__main__':
	N = 313
	D = 40
	s = 70
	m = 50
	l = 30
	k = 30
	SCALE = 1000.0

	X = np.matrix(np.random.normal(size = [D,N])*SCALE)  # create a dataset of size N and D dimensions with values in [0,SCALE)

	kernel_func = lambda x1,x2: utils.rbf_kernel(x1,x2,gamma = 0.05)   # define kernel function
	#kernel_func = utils.dot_kernel


	#whole kernel matrix that we want to approximate (NxN), computed for comparison here
	Kx = utils.construct_kernel_matrix(X,X, kernel = kernel_func) #X.T*X

	Vnys, Snys = double_nystrom(X,s,m,l,-1,kernel_func)
	Knys = Vnys*np.diag(Snys)*Vnys.T
	print(np.linalg.norm(Knys - Kx))
	Vnys, Snys = one_shot_nystrom(X,s,k,kernel_func)
	Knys = Vnys*np.diag(Snys)*Vnys.T
	print(np.linalg.norm(Knys - Kx))
	#print(np.linalg.norm(Knys1 - Kx))
	#print(np.linalg.norm(Knys2 - Kx))
	#print(np.linalg.norm(Knys3 - Kx))
	#print(np.linalg.norm(Knys4 - Kx))