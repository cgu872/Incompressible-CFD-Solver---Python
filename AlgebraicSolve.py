# -*- coding:utf-8 -*-
# @data 2025/4/10

import numpy as np
import sys
import scipy.sparse as sp
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import gmres
import Writelog

def solvers(Coeffs,elementN,elementNeighbours,solvescheme,field,iComponent):
    #1 -------- define sparse matrix of A from A @ x = b
    diagonal_matrix=sp.diags(Coeffs.ac[:, 0], 0, format='csr')
    add_matrix = sp.lil_matrix((elementN, elementN))
    for icell in range(0, elementN):
        cols = elementNeighbours[icell]
        values = Coeffs.anb[icell]
        for col, val in zip(cols, values):
            add_matrix[icell, col] = val
    add_matrix = add_matrix.tocsr()
    sparse_matrix = diagonal_matrix + add_matrix
    local_residual=Coeffs.bc # A[x'] = B - A[x*], residual form

    #2 ======= solve the algebraic equation
    Solver='BiCGstab'
    rTol=solvescheme['relTol']
    aTol=solvescheme['tolerance']
    maxIter=solvescheme['maxIter']
    rTol = 0.001
    maxIter=None
    # define DILU preconditioner
    # ml = pyamg.smoothed_aggregation_solver(A, presmoother='gauss_seidel', postsmoother='gauss_seidel')
    # M = ml.aspreconditioner()
    if Solver == 'CG':
        #Conjugate Gradient method
        dphi, info = cg(sparse_matrix, local_residual, tol=rTol, atol=aTol, maxiter=maxIter)
    elif Solver=='BiCG':
        #BiConjugate Gradient method
        # ---usage of bicg---#
        # specify atol explicitly for version(>1.7.3) compatibility
        # M is Preconditioner for A, approximate the inverse of A
        # maxiter is Maximum number of iterations
        # norm(b - A @ x) <= max(tol*norm(b), atol), where Frobenius norm
        # -------------------#
        dphi, info = bicg(sparse_matrix, local_residual, x0=None, tol=rTol, atol=aTol, maxiter=maxIter, M=None)
    elif Solver=='BiCGstab':
        #BiConjugate Gradient Stabilized
        dphi, info = bicgstab(sparse_matrix, local_residual, tol=rTol, atol=aTol, maxiter=maxIter)
    elif Solver == 'CG':
        #Conjugate Gradient method
        dphi, info = cg(sparse_matrix, local_residual, tol=rTol, atol=aTol, maxiter=maxIter)
    elif Solver == 'GMRes':
        #Generalized Minimal RESidual method
        dphi, info = gmres(sparse_matrix, local_residual, tol=rTol, maxiter=maxIter)
    else:
        sys.exit('Undefined solvers')

    #3 =======calculate the residuals
    maxRes,rmsRes,initialRes,finalRes=residual(dphi,sparse_matrix,local_residual)
    Writelog.ResInfo(maxRes,rmsRes,initialRes,finalRes,field.name,iComponent)

    #4.1 =======update the internal field
    field.phi[0:elementN,iComponent]=field.phi[0:elementN,iComponent]+dphi


def residual(dphi,sparse_matrix,local_residual):
    #cal Max, Root-Mean Square, Mean of Residuals
    maxRes=np.max(local_residual)
    rmsRes=np.sqrt(np.mean(np.square(local_residual)))
    initialRes=np.mean(np.abs(local_residual))
    finalResArray=local_residual - sparse_matrix.dot(dphi[:, np.newaxis])
    finalRes=np.mean(np.abs(finalResArray))
    return maxRes,rmsRes,initialRes,finalRes

