#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author: A. Pezzotta -- pezzota [AT] crick.ac.uk

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.sparse.linalg import splu, lsqr, lsmr
from scipy.sparse import coo_matrix

def linsolver_SVD (A, b):
	u, s, vh = np.linalg.svd(A, full_matrices=False)
	c = np.dot(u.T, b)
	w = np.linalg.solve(np.diag(s), c)
	z = np.dot(vh.T, w)
	return z

def directSolve (pt, end, method='lsqr'):

	N = len(pt)
	M = pt.T - np.eye(N)
	mat = np.delete(M, end, axis=1)   # delete columns corresponding to target nodes
	vec = - np.sum(M[:,end], axis=1)  # vector implementing "boundary" conditions

	# print(M,"\n")
	# print(mat,"\n")
	# print(vec,"\n")

	Z = np.ones(N)
	if method == 'svd':
		Zrec = np.squeeze(linsolver_SVD(mat, vec))
	elif method == 'lsqr':
		mat = coo_matrix(mat)
		Zrec = lsqr(mat, vec)[0]
	else:
		raise ValueError("directSolve: unknown method")

	Z[np.delete(np.arange(N), end)] = Zrec
	return Z
