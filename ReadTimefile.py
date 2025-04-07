# -*- coding:utf-8 -*-
# @data 2025/3/19

import sys,re,os
import numpy as np
import copy
# from pyparsing import Word, alphas, alphanums, Group, Dict, Suppress, oneOf

def extract_content(s):
    str=re.findall(r'\((.*?)\)', s)
    num=[list(map(float, str[0].split()))]
    return np.array(num)

def unitVector(faceSf,iFaceStart,iFaceEnd):
    magSf = np.linalg.norm(faceSf[iFaceStart:iFaceEnd], axis=1, ord=2)
    unitSf = faceSf[iFaceStart:iFaceEnd] / magSf[:, np.newaxis]
    return unitSf

class setfield(object):
    def __init__(self,filedir,filename,initialfield,phi,type):
        self.name=filename
        self.initialfield = initialfield
        self.type = type
        self.filedir = os.path.join(filedir,filename)
        self.phi = phi
        self.prevIter = copy.deepcopy(phi)
        self.prevTimeStep = copy.deepcopy(phi)
    def internalField(self,elementN):
        value=self.initialfield['internalField']
        if value[0] == 'uniform':
            self.phi[0:elementN] = value[1]
        else:
            print('To be supplemented')
        ''' use for real p,U file
        pattern = r'^(internalField)'
        with open(self.filedir, "r", encoding="utf-8") as file:
            lines = [line.strip().split("//")[0] for line in file if re.match(pattern, line, re.IGNORECASE)]
        if lines[0].strip(';').split()[1]=='uniform':
            if len(lines[0].strip(';').split())>3:
                self.value = extract_content(lines[0].strip(';'))
            else:
                self.value = float(lines[0].strip(';').split()[2])
            self.phi[0:elementN] = self.value
        else:
            print('To be supplemented')
        '''
    def boundaryField(self,ownerdata,boundarydata,faceSf,InnerfaceN,elementN):
        NumberBPatches=len(boundarydata)
        for iBPatch in range(0,NumberBPatches):
            theBCInfo = boundarydata[iBPatch]
            PatchDefine=self.initialfield['boundaryField'][theBCInfo['name']]
            NumberBFaces = int(theBCInfo['nFaces'])
            iFaceStart = int(theBCInfo['startFace'])
            iFaceEnd = iFaceStart + NumberBFaces
            iElementStart = elementN + iFaceStart - InnerfaceN
            iElementEnd = iElementStart + NumberBFaces
            if PatchDefine['type']=='fixedValue':
                valueset = PatchDefine['value']
                if valueset[0]=='uniform':
                    self.phi[iElementStart:iElementEnd] = valueset[1]
                else:
                    print('To be supplemented')
            elif (PatchDefine['type']=='zeroGradient' or PatchDefine['type'] == 'empty'):
                self.phi[iElementStart:iElementEnd] = self.phi[ownerdata[iFaceStart:iFaceEnd]]
            elif (PatchDefine['type'] == 'symmetry'):
                own = ownerdata[iFaceStart:iFaceEnd]
                if self.name=='U':
                    unitSf=unitVector(faceSf,iFaceStart,iFaceEnd)
                    U_normal=np.sum(self.phi[own] * unitSf, axis=1)[:, np.newaxis]*unitSf #np.einsum('ij,ij->i', self.phi[own],unitSf)
                    self.phi[iElementStart:iElementEnd] =self.phi[own]-U_normal
                else:
                    self.phi[iElementStart:iElementEnd] = self.phi[own]
            elif (PatchDefine['type'] == 'noSlip'): #only for U
                self.phi[iElementStart:iElementEnd]=np.zeros_like(self.phi[iElementStart:iElementEnd])
            else:
                print('Undefined Boundary Condition: ',theBCInfo['name'])
            # if PatchDefine.get('value') is not None:

#==================================================================================#
def ReadStartTime(workdir,controlDict,InnerfaceN,elementN,elementBN,
                  ownerdata,boundarydata,faceSf):
    # 1.-------------------------------------------------------#
    #找初始场文件
    # -------------------------------------------------------#
    startFrom=controlDict['startFrom']
    if startFrom=='startTime':
        timeDirectory=str(controlDict['startTime'])
    elif (startFrom=='firstTime'):
        timeDirectory = '0'
    elif (startFrom == 'latestTime'):
        items = os.listdir(workdir)
        folders = [int(item) for item in items if os.path.isdir(os.path.join(workdir, item)) and item.isdigit()]
        timeDirectory =str(np.max(folders))
    else:
        sys.exit("Invalid startTime in controlDict.")

    # 2.-------------------------------------------------------#
    #读0/p,0/U并写成字典
    # -------------------------------------------------------#
    filedir = os.path.join(workdir, timeDirectory)
    PhysicalField=os.listdir(filedir)
    #自行设置初始场文件，之后再修改成读取OpenFOAM文件
    initialU={"internalField": ["uniform",np.array([20,0,0])],
    "boundaryField":
        {"inlet": {"type": "fixedValue","value": ["uniform",np.array([20,0,0])]},
        "outlet": {"type": "zeroGradient"},
        "walls": {"type": "noSlip"},#noSlip
        "front": {"type": "empty"},
        "back": {"type": "empty"}}
              }
    initialp = {"internalField": ["uniform", 0],
            "boundaryField":
                {"inlet": {"type": "zeroGradient"},
                 "outlet": {"type": "fixedValue", "value": ["uniform", 0]},
                 "walls": {"type": "zeroGradient"},
                 "front": {"type": "empty"},
                 "back": {"type": "empty"}}
            }
    '''
    key = Word(alphas, alphanums + "_")
    value = Word(alphanums + "._-")
    property_def = key  + value + Suppress(";")
    block = Group(key + Suppress("{") + Dict(property_def[...]) + Suppress("}"))
    foam_dict = Dict(Group("boundaryField" + Suppress("{") + Dict(block[...] | Suppress("}"))))
    parsed_data = foam_dict.parseString(foam_text).asDict()
    '''
    # for filename in PhysicalField:
    #     fielddir = os.path.join(filedir, filename)
    #     with open(fielddir, "r", encoding="utf-8") as file:

    # 3.-------------------------------------------------------#
    #初始化场phi，设置内部场和边界场数值
    # -------------------------------------------------------#
    for filename in PhysicalField:
        if filename=='U':
            phi = np.zeros((elementN + elementBN, 3))
            type='volVectorField'
            Ufield = setfield(filedir,filename,initialU,phi,type)
            Ufield.internalField(elementN)
            Ufield.boundaryField(ownerdata,boundarydata,faceSf,InnerfaceN,elementN)
        elif filename == 'p':
            phi = np.zeros((elementN + elementBN, 1))
            type = 'volScalarField'
            pfield = setfield(filedir,filename,initialp,phi,type)
            pfield.internalField(elementN)
            pfield.boundaryField(ownerdata,boundarydata,faceSf,InnerfaceN,elementN)
        else:
            print('To be supplemented')
    return Ufield,pfield,timeDirectory

