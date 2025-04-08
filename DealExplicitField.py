# -*- coding:utf-8 -*-
# @data 2025/3/28
import sys
import numpy as np
import Interpolate

def MassFlux(Ufield,rhofield,transportProperties,elementN,InnerfaceN,faceN,faceSf,faceWeights,ownerdata,neighbourdata):
    own=ownerdata[0:InnerfaceN]
    nei=neighbourdata
    fweights=np.array(faceWeights)[0:InnerfaceN,np.newaxis]
    U=Ufield.phi
    U_innerf=Interpolate.ElementsToFaces('linear', U, own, nei, fweights)
    U_f=np.vstack((U_innerf,U[elementN:]))
    rho_f=transportProperties['rho']
    mdot_f=np.sum(U_f*faceSf, axis=1)*rho_f
    return U_f,mdot_f[:,np.newaxis]

def Gradient(schemes,field,elementN,InnerfaceN,ownerdata,neighbourdata,boundarydata,faceSf,faceWeights,
             elementVolumes,cellOwnFace,cellNeiFace,faceCf, faceFf):
    if schemes=='Gauss linear':
        own = ownerdata[0:InnerfaceN]
        nei = neighbourdata
        fweights = np.array(faceWeights)[0:InnerfaceN, np.newaxis]
        phi = field.phi
        phi_innerf=Interpolate.ElementsToFaces('linear', phi, own, nei, fweights)
        phi_f = np.vstack((phi_innerf, phi[elementN:]))
        componentN=np.size(phi, axis=1)
        phiGrad=[]
        for icomponent in range(0,componentN):
            # First Green-Gauss
            phiflux = phi_f[:,icomponent, np.newaxis]*faceSf
            phiGradcomponent=[]
            for icell in range(0,elementN):
                phiGradC=np.sum(phiflux[cellOwnFace[icell]],axis=0)-np.sum(phiflux[cellNeiFace[icell]],axis=0)
                phiGradC=phiGradC/elementVolumes[icell]
                phiGradcomponent.append(phiGradC)
            phiGradCell=np.array(phiGradcomponent)
            # Correct for phi_f in the faces
            phi_innerf2=(phi_innerf[:,icomponent, np.newaxis]+fweights*np.sum(phiGradCell[nei] * faceFf, axis=1)[:, np.newaxis]
                         +(1-fweights)*np.sum(phiGradCell[own]*faceCf[0:InnerfaceN], axis=1)[:, np.newaxis])
            phi_f2 = np.vstack((phi_innerf2, phi[elementN:,icomponent,np.newaxis]))
            # Second Green-Gauss
            phiflux = phi_f2 * faceSf
            phiGradcomponent = []
            for icell in range(0, elementN):
                phiGradC = (np.sum(phiflux[cellOwnFace[icell]], axis=0)
                            - np.sum(phiflux[cellNeiFace[icell]], axis=0))
                phiGradC = phiGradC / elementVolumes[icell]
                phiGradcomponent.append(phiGradC)
            phiGradCell = np.array(phiGradcomponent)
            # Boundary(faces) gradients
            for iBPatch in range(0, len(boundarydata)):
                Btype=field.initialfield['boundaryField'][boundarydata[iBPatch]['name']]['type']
                startFace=int(boundarydata[iBPatch]['startFace'])
                nFaces=int(boundarydata[iBPatch]['nFaces'])
                own_b = ownerdata[startFace:startFace+nFaces]
                CB=np.array(faceCf[startFace:startFace+nFaces])
                distance_CB=np.linalg.norm(CB,axis=1)[:,np.newaxis]
                e_CB=CB / np.linalg.norm(CB,axis=1)[:,np.newaxis]
                if Btype=='symmetry' or Btype=='empty' or Btype=='zeroGradient':
                    phiGradBound=np.zeros((nFaces,np.size(phiGradCell,axis=1)))
                else:
                    phiGradBound=(phiGradCell[own_b]-np.sum(phiGradCell[own_b]*e_CB, axis=0)*e_CB
                                  +(phi_f[startFace:startFace+nFaces]-phi_f2[own_b])/distance_CB*e_CB)
                phiGradCell=np.vstack((phiGradCell,phiGradBound))
            phiGrad.append(phiGradCell)
    else:
        sys.exit("Undefined Gradient Scheme.")
    field.gradphi=phiGrad
    return field