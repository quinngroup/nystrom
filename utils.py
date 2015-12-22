import math
import numpy as np

def vector_pinv(vector, tol = -1):
	if tol == -1:
		tol = np.finfo(float).eps*np.size(vector)*np.max(vector)
	return np.array([1/x if x > tol else x for x in np.nditer(vector)])

def dot_kernel(x1,x2):
	return np.dot(x1,x2.T)[0,0]
def rbf_kernel(x1,x2,gamma = 0.05):
	diff = x1-x2
	return math.exp(-gamma*dot_kernel(diff,diff))
def svd(x,**kw):
	print("SVD: ", x.shape)
	return np.linalg.svd(x,**kw)
def construct_kernel_matrix(X1,X2,kernel=dot_kernel):
	return np.matrix(np.array([[kernel(xi,xj) for xj in X2.T] for xi in X1.T]))

	