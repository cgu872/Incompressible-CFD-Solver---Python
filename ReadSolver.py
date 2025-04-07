# -*- coding:utf-8 -*-
# @data 2025/3/19

import sys,re,os
import numpy as np
from itertools import islice


def ReadcontrolDict(workdir):
    os.chdir(workdir)
    filedir=os.path.join(workdir,"system\\controlDict")
    if not os.path.exists(filedir):
        sys.exit("There is no controlDict.")
    else:
        controlDict={}
        pattern = r'^(application|startFrom|startTime|stopAt|endTime|deltaT|writeControl|writeInterval|purgeWrite)'
        with open(filedir, "r", encoding="utf-8") as file:
            lines =  [line.strip().split("//")[0] for line in file if re.match(pattern, line, re.IGNORECASE)]
        contents = [line.strip(';').split() for line in lines]
        for content in contents:
            controlDict[content[0]] = content[1]
        # with open(filedir, "r", encoding="utf-8") as file:
        #     content = file.read()
        # pattern = re.compile(r'\b(application|startFrom|startTime|stopAt|endTime|deltaT|writeControl|writeInterval|purgeWrite)\b\s+(\S+);', re.DOTALL)
        # matches = pattern.findall(content)
    return controlDict

def ReadfvSchemes(workdir):
    os.chdir(workdir)
    filedir=os.path.join(workdir,"system\\fvSchemes")
    if not os.path.exists(filedir):
        sys.exit("There is no fvSchemes.")
    else:
        fvSchemes = {}
        fvSchemes['ddtSchemes']='steadyState' #Euler
        fvSchemes['gradSchemes'] = 'Gauss linear'
        fvSchemes['divSchemes'] = 'Gauss linear' #second order upwind
        fvSchemes['laplacianSchemes'] = 'Gauss linear corrected'
        fvSchemes['interpolationSchemes'] = 'linear'
        fvSchemes['snGradSchemes'] = 'corrected'
    return fvSchemes

def ReadfvSolution(workdir):
    os.chdir(workdir)
    filedir = os.path.join(workdir, "system\\fvSolution")
    if not os.path.exists(filedir):
        sys.exit("There is no fvSolution.")
    else:
        fvSolution = {}
        p, U={},{}
        p['solver']='GAMG'
        p['preconditioner']='DILU'
        p['tolerance'] = 1e-09
        p['relTol'] = 0.01
        p['nPreSweeps'] = 1
        p['nPostSweeps'] = 3
        p['maxIter'] = 20
        p['nFinestSweeps'] = 2
        U['solver'] = 'smoothSolver'
        U['smoother'] = 'DILU'
        U['tolerance'] = 1e-09
        U['relTol'] = 0.01
        U['maxIter'] = 20
        fvSolution['solvers']={}
        fvSolution['solvers']['p']=p
        fvSolution['solvers']['U']=U
        fvSolution['SIMPLE'] ={}
        fvSolution['SIMPLE']['nCorrectors'] = 1
        fvSolution['SIMPLE']['pRefCell'] = 1
        fvSolution['SIMPLE']['pRefValue'] = 0
        residualControl={}
        residualControl['p']=0.0001
        residualControl['U']=0.0001
        fvSolution['SIMPLE']['residualControl'] = residualControl
        fvSolution['relaxationFactors'] = {}
        fields,equations={},{}
        fields['p']=0.1
        equations['U']=0.9
        fvSolution['relaxationFactors']['equations']=equations
        fvSolution['relaxationFactors']['fields '] =fields
    return fvSolution

