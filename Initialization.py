# -*- coding:utf-8 -*-
# @data 2025/3/27
import numpy as np

class Coeff(object):
    def __init__(self,elementN,elementNeighbours):
        self.ac = np.zeros((elementN, 1))
        self.ac_old = np.zeros((elementN, 1))
        self.bc = np.zeros((elementN, 1))
        self.anb = [[0] * len(sublist) for sublist in elementNeighbours]
        # dc,rc used in ILU Solver
        self.dc = np.zeros((elementN, 1))
        self.rc = np.zeros((elementN, 1))
        # Correction
        self.dphi = np.zeros((elementN, 1))

class Flux(object):
    def __init__(self,faceN, elementN):
        # Face fluxes
        self.FluxCf = np.zeros((faceN, 1))
        self.FluxFf = np.zeros((faceN, 1))
        self.FluxVf = np.zeros((faceN, 1))
        self.FluxTf = np.zeros((faceN, 1))
        # Volume fluxes
        self.FluxC = np.zeros((elementN, 1))
        self.FluxV = np.zeros((elementN, 1))
        self.FluxT = np.zeros((elementN, 1))
        self.FluxC_old = np.zeros((elementN, 1))