# -*- coding:utf-8 -*-
# @data 2025/4/11

import numpy as np
from ReadTimefile import unitVector

def UField(Ufield,elementN,InnerfaceN,faceSf,ownerdata,boundarydata):
    # 4.2 =======update the boundary field
    NumberBPatches = len(boundarydata)
    for iBPatch in range(0, NumberBPatches):
        theBCInfo = boundarydata[iBPatch]
        PatchDefine = Ufield.initialfield['boundaryField'][theBCInfo['name']]
        NumberBFaces = int(theBCInfo['nFaces'])
        iFaceStart = int(theBCInfo['startFace'])
        iFaceEnd = iFaceStart + NumberBFaces
        iElementStart = elementN + iFaceStart - InnerfaceN
        iElementEnd = iElementStart + NumberBFaces
        if PatchDefine['type'] == 'zeroGradient':
            own = ownerdata[iFaceStart:iFaceEnd]
            Ufield.phi[iElementStart:iElementEnd] = Ufield.phi[own]
        elif PatchDefine['type'] == 'symmetry' or PatchDefine['type'] == 'empty':
            own = ownerdata[iFaceStart:iFaceEnd]
            unitSf = unitVector(faceSf, iFaceStart, iFaceEnd)
            U_normal = np.sum(Ufield.phi[own] * unitSf, axis=1)[:,np.newaxis] * unitSf  # np.einsum('ij,ij->i', self.phi[own],unitSf)
            Ufield.phi[iElementStart:iElementEnd] = Ufield.phi[own] - U_normal
        # fixedValue,noSlip can't change