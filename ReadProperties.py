# -*- coding:utf-8 -*-
# @data 2025/3/25
# @file ReadProperties.py

import sys,re,os
import numpy as np
import copy

class setConstantfield(object):
    def __init__(self,filename,value,phi,type):
        self.name = filename
        self.type = type
        self.phi = phi*value
        self.prevIter = copy.deepcopy(phi)*value
        self.prevTimeStep = copy.deepcopy(phi)*value

def Transport(workdir,elementN,elementBN):
    filedir=os.path.join(workdir,"constant\\transportProperties")
    propertyValue = {"mu": 1.8e-5,
                     "rho": 1.225,
                     "nu": 1.5e-5}
    phi = np.ones((elementN + elementBN, 1))
    type = 'volScalarField'
    mufield=setConstantfield('mu',propertyValue['mu'],phi,type)
    rhofield=setConstantfield('rho',propertyValue['rho'],phi,type)
    return mufield,rhofield,propertyValue


def Turbulence(workdir,):
    filedir = os.path.join(workdir, "constant\\turbulenceProperties")
    turbulenceProperties={"simulationType": "RAS",
                        "RAS": {"RASModel": "kEpsilon",
                                "turbulence": "on",
                                "printCoeffs": "off"}}
    return turbulenceProperties
