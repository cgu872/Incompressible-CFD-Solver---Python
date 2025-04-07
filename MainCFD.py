# -*- coding:utf-8 -*-
# @data 2025/3/19

import time
import sys,re,os
import numpy as np
from itertools import islice
from numba import jit
import pandas as pd
import csv,json,math
import copy
import ReadMesh, ReadSolver, ReadTimefile, ReadProperties
import Interpolate, Initialization, Writelog
import DealExplicitField,Discretization


if __name__ == '__main__':
    startTime = time.time()
    workdir="C:\\Users\\ASUS\\Desktop\\airfoil"
    # workdir = "C:\\Users\\cgu872\\Desktop\\airfoil"
    # workdir = "C:\\Users\\ASUS\\Desktop\\mesh7"

    #================1. Read Setting================#
    #------------------1.1 Read Mesh--------------------#
    [pointdata, facedata, ownerdata, neighbourdata, boundarydata,
    pointN, faceN, InnerfaceN, BoundaryfaceN, elementN, elementBN, BoundaryTypeN,
    elementNeighbours, LinkFaces, elementFaces,
    faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes,
    faceCF, faceCf, faceFf, faceWeights,wallDist,
    cellOwnFace,cellNeiFace]=ReadMesh.MeshDeal(workdir)
    endTime = time.time()
    costTime = (endTime - startTime) / 60.0
    print("Reading mesh cost %.3f minutes" % costTime)
    Writelog.MeshInfo(pointN, faceN, InnerfaceN, elementN,costTime,workdir)
    # ------------------1.2 Read controlDict, fvSchemes and fvSolution--------------------#
    controlDict=ReadSolver.ReadcontrolDict(workdir)
    fvSchemes=ReadSolver.ReadfvSchemes(workdir)
    fvSolution=ReadSolver.ReadfvSolution(workdir)
    # ------------------1.3 Read Time file and initialize fields--------------------#
    Ufield,pfield,timeDirectory=ReadTimefile.ReadStartTime(workdir,controlDict,InnerfaceN,
                                                           elementN,elementBN,ownerdata,boundarydata,faceSf)
    CurrentTime = float(timeDirectory)
    EndTime=float(controlDict['endTime'])
    # ------------------1.4 Read Transport and Turbulence Properties--------------------#
    mufield,rhofield,transportProperties=ReadProperties.Transport(workdir,elementN,elementBN)
    turbulenceProperties=ReadProperties.Turbulence(workdir)
    Writelog.SolveInfo(fvSchemes,timeDirectory,transportProperties,turbulenceProperties)

    # ================2. Initialization================#
    # ------------------2.1 Calculate the mass flux in faces (mdot_f)--------------------#
    U_f,mdot_f=DealExplicitField.MassFlux(Ufield,rhofield,transportProperties,elementN,InnerfaceN,
                                          faceN,faceSf,faceWeights,ownerdata,neighbourdata)
    # ------------------2.2 Gradients for cell and boundary face--------------------#
    Ufield=DealExplicitField.Gradient(fvSchemes['gradSchemes'],Ufield,elementN,InnerfaceN,ownerdata,neighbourdata,boundarydata,
                                      faceSf,faceWeights,elementVolumes,cellOwnFace,cellNeiFace,faceCf, faceFf)
    pfield=DealExplicitField.Gradient(fvSchemes['gradSchemes'],pfield,elementN,InnerfaceN,ownerdata,neighbourdata,boundarydata,
                                      faceSf,faceWeights,elementVolumes,cellOwnFace,cellNeiFace,faceCf, faceFf)
    ppfield=copy.deepcopy(pfield) #Setup Pressure Correction field
    # 还需建立DU1，DU2，DU3，DUT1，DUT2，DUT3，pp场，不用先放着
    # ------------------2.3 Initialize coefficients matrix--------------------
    Coeffs = Initialization.Coeff(elementN,elementNeighbours)
    Fluxes = Initialization.Flux(faceN, elementN)

    # ================3. Iteration================#
    IterationsN = 0
    # user-defined false time step先放着
    if fvSchemes['ddtSchemes'] == 'steadyState':
        while (CurrentTime <= EndTime):
            IterTime1 = time.time()
            IterationsN += 1
            CurrentTime=CurrentTime+float(controlDict['deltaT'])
            Writelog.IterInfo(IterationsN)
            # Updates U, p, mdot_f
            Ufield.prevIter=copy.deepcopy(Ufield.phi)
            pfield.prevIter = copy.deepcopy(pfield.phi)
            mdot_f_prevIter = copy.deepcopy(mdot_f)
            # ---------------Momentum Ux,Uy,Uz--------------- #
            for iComponent in range(0,3):
                Coeffs = Initialization.Coeff(elementN, elementNeighbours)
                Discretization.Momentum(iComponent,fvSchemes,Ufield,pfield,mdot_f,mufield,rhofield,Coeffs,\
                            ownerdata,neighbourdata,cellOwnFace,cellNeiFace,faceSf,faceCF,faceWeights,\
                            faceCentroids,elementCentroids,elementVolumes,InnerfaceN,faceN,elementN,\
                            elementNeighbours,LinkFaces,elementFaces,wallDist)

            # ---------------Continuity pressure correction p'--------------- #
            Coeffs = Initialization.Coeff(elementN, elementNeighbours)

            IterTime2 = time.time()
            costTime = (IterTime2 - IterTime1) / 60.0
            print("One Iteration cost %.3f minutes" % costTime)
            Writelog.IterTime(costTime)
    else: #Transient
        print("Transient computation not available")


    # jsonOutput="C:\\Users\\cgu872\\Desktop\\123.json"
    # with open(jsonOutput, "w") as f:
    #     json.dump(boundarydata, f)
    # numbers = [float(num) for num in numbers]
    # os.remove(tifOutput2)
    # del demdata