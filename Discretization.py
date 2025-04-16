# -*- coding:utf-8 -*-
# @data 2025/4/1

import sys
import numpy as np
import Initialization,Interpolate

def Momentum(iComponent,fvSchemes,fvSolution,Ufield,pfield,mdot_f,mufield,rhofield,Coeffs,
             ownerdata, neighbourdata,cellOwnFace,cellNeiFace,faceSf,faceCF,faceWeights,
             faceCentroids,elementCentroids,elementVolumes,InnerfaceN,faceN,elementN,
             elementNeighbours,LinkFaces,elementFaces,faceCf,boundarydata):
    MomentumTerms = ['Convection', 'Diffusion', 'PressureGradient']
    UnderRelaxation='FalseTransient'
    phi=Ufield.phi[:,iComponent,np.newaxis]
    own = ownerdata
    nei = neighbourdata
    for terms in MomentumTerms:
        if terms=='Convection':
            Fluxes = Initialization.Flux(faceN, elementN)
            #--------implicit one order Upwind internal faces phi_f
            Fluxes.FluxCf[0:InnerfaceN] = np.maximum(mdot_f[0:InnerfaceN],0)
            Fluxes.FluxFf[0:InnerfaceN] = -np.maximum(-mdot_f[0:InnerfaceN], 0)
            #--------implicit one order Upwind boundary faces phi_f
            NumberBPatches = len(boundarydata)
            for iBPatch in range(0, NumberBPatches):
                theBCInfo = boundarydata[iBPatch]
                PatchDefine = Ufield.initialfield['boundaryField'][theBCInfo['name']]
                NumberBFaces = int(theBCInfo['nFaces'])
                iFaceStart = int(theBCInfo['startFace'])
                iFaceEnd = iFaceStart + NumberBFaces
                if PatchDefine['type'] == 'zeroGradient':
                    Fluxes.FluxCf[iFaceStart:iFaceEnd] = mdot_f[iFaceStart:iFaceEnd]
            #--------explicit one order Upwind internal faces Tf for its owner (defined as bc-Tf)
            Fluxes.FluxTf[0:InnerfaceN] = Fluxes.FluxCf[0:InnerfaceN]*phi[own[0:InnerfaceN]] + Fluxes.FluxFf[own[0:InnerfaceN]]*phi[nei]
            #--------explicit boundary faces Tf for its owner
            Fluxes.FluxTf[InnerfaceN:faceN] = mdot_f[InnerfaceN:faceN] * phi[own[InnerfaceN:faceN]]
            #--------explicit high oder internal faces Tf
            if fvSchemes['divSchemes'] == 'Gauss upwind':
                continue
            elif fvSchemes['divSchemes']=='Gauss linear': #only internal faces,means no deferred correction for boundary
                Upindex = np.where(mdot_f[0:InnerfaceN, 0] >= 0, own[0:InnerfaceN], nei)
                '''
                Upindex = np.zeros(InnerfaceN)
                for iface in range(0,InnerfaceN):
                    if mdot_f[iface] >= 0:
                        Upindex[iface]=own[iface]
                    else:
                        Upindex[iface]=nei[iface]
                '''
                gradphi=Ufield.gradphi[iComponent][0:elementN]
                phiGradC = gradphi[Upindex]
                phiGrad_f = Interpolate.GradientElementsToFaces('Gauss linear corrected',gradphi,phi[0:elementN],\
                            own[0:InnerfaceN],nei,faceCF[0:InnerfaceN],faceWeights[0:InnerfaceN])
                d_Cf=np.array(faceCentroids[0:InnerfaceN])-np.array(elementCentroids)[Upindex]
                dc_HO= mdot_f[0:InnerfaceN]*np.einsum('ij,ij->i',(2*phiGradC-phiGrad_f),d_Cf)[:,np.newaxis] #Deferred Correction for high order scheme
                Fluxes.FluxTf[0:InnerfaceN]=Fluxes.FluxTf[0:InnerfaceN]+dc_HO
            elif fvSchemes['divSchemes']=='QUICK':
                Upindex = np.where(mdot_f[0:InnerfaceN, 0] >= 0, own[0:InnerfaceN], nei)
                gradphi = Ufield.gradphi[iComponent][0:elementN]
                phiGradC = gradphi[Upindex]
                phiGrad_f = Interpolate.GradientElementsToFaces('Gauss linear corrected', gradphi,phi[0:elementN], \
                                                                own[0:InnerfaceN], nei, faceCF[0:InnerfaceN],faceWeights[0:InnerfaceN])
                d_Cf = np.array(faceCentroids[0:InnerfaceN]) - np.array(elementCentroids)[Upindex]
                dc_HO = mdot_f[0:InnerfaceN] * (0.5*np.einsum('ij,ij->i', (phiGradC + phiGrad_f), d_Cf)[:,np.newaxis]) # Deferred Correction
                Fluxes.FluxTf[0:InnerfaceN] = Fluxes.FluxTf[0:InnerfaceN] + dc_HO
            else:
                sys.exit("Undefined Convection Scheme: "+fvSchemes['divSchemes'])
            #--------assemble matrix based on cells, bounary FluxCf and FluxFf should be zero except zeroGradient
            for icell in range(0,elementN):
                Coeffs.ac[icell]=Coeffs.ac[icell]+np.sum(Fluxes.FluxCf[cellOwnFace[icell]], axis=0)-np.sum(Fluxes.FluxFf[cellNeiFace[icell]],axis=0)
                AroundFace = LinkFaces[icell]
                for idx, inei in enumerate(AroundFace): #anb与elementNeighbours对应
                    if inei in cellOwnFace[icell]:
                        Coeffs.anb[icell][idx] = Coeffs.anb[icell][idx]+Fluxes.FluxFf[inei] # boundary 不计算anb
                    else:
                        Coeffs.anb[icell][idx] = Coeffs.anb[icell][idx]-Fluxes.FluxCf[inei]
                Coeffs.bc[icell]=Coeffs.bc[icell]+np.sum(Fluxes.FluxTf[cellNeiFace[icell]],axis=0)-np.sum(Fluxes.FluxTf[cellOwnFace[icell]], axis=0)
            # Divergence Correction for stability purposes
            for icell in range(0, elementN):
                effDiv=np.sum(mdot_f[cellOwnFace[icell]],axis=0)-np.sum(mdot_f[cellNeiFace[icell]],axis=0)
                Fluxes.FluxC[icell]= np.maximum(effDiv,0)-effDiv
                Fluxes.FluxT[icell]= -effDiv*phi[icell]
                Coeffs.ac[icell]=Coeffs.ac[icell]+Fluxes.FluxC[icell]
                Coeffs.bc[icell]=Coeffs.bc[icell]-Fluxes.FluxT[icell]
        elif terms=='Diffusion':
            Fluxes = Initialization.Flux(faceN, elementN)
            mu=mufield.phi[0]
            #--------internal faces 系数ac,anb都不会变，迭代过程中可省略
            Sf=np.array(faceSf[0:InnerfaceN])
            magSf=np.linalg.norm(Sf, axis=1, ord=2)[:,np.newaxis]
            CF= np.array(faceCF[0:InnerfaceN])
            d_CF=np.linalg.norm(CF, axis=1, ord=2)[:,np.newaxis]
            e_CF = CF / d_CF
            gradphi = Ufield.gradphi[iComponent][0:elementN]
            phiGrad_f = Interpolate.GradientElementsToFaces('linear', gradphi,phi[0:elementN], \
                        own[0:InnerfaceN], nei, faceCF[0:InnerfaceN],faceWeights[0:InnerfaceN])
            #decomposition of SF
            NonOrthogonalScheme='Over-Relaxed Approach'
            if NonOrthogonalScheme=='Over-Relaxed Approach':
                # Over-Relaxed Approach
                Ef = (np.einsum('ij,ij->i',Sf,Sf)/np.einsum('ij,ij->i',e_CF,Sf))[:,np.newaxis]*e_CF
            elif NonOrthogonalScheme=='Orthogonal Correction Approach':
                # Orthogonal Correction Approach
                Ef = magSf*e_CF
            elif NonOrthogonalScheme=='Minimum Correction Approach':
                # Minimum Correction Approach
                Ef = np.einsum('ij,ij->i',e_CF,Sf)[:,np.newaxis]*e_CF
            else:
                sys.exit("Undefined Non-Orthogonal Diffusion Scheme" )
            # Tf = Sf - Ef
            geoDiff_f = np.linalg.norm(Ef, axis=1, ord=2)[:,np.newaxis]/d_CF
            Fluxes.FluxCf[0:InnerfaceN] = mu * geoDiff_f
            Fluxes.FluxFf[0:InnerfaceN] = -mu * geoDiff_f
            Fluxes.FluxTf = mu*np.einsum('ij,ij->i',np.vstack((phiGrad_f,Ufield.gradphi[iComponent][elementN:])),np.array(faceSf))[:,np.newaxis]
            #--------boundary faces
            NumberBPatches = len(boundarydata)
            for iBPatch in range(0, NumberBPatches):
                theBCInfo = boundarydata[iBPatch]
                PatchDefine = Ufield.initialfield['boundaryField'][theBCInfo['name']]
                NumberBFaces = int(theBCInfo['nFaces'])
                iFaceStart = int(theBCInfo['startFace'])
                iFaceEnd = iFaceStart + NumberBFaces
                iElementStart = elementN + iFaceStart - InnerfaceN
                iElementEnd = iElementStart + NumberBFaces
                if PatchDefine['type'] == 'fixedValue' or PatchDefine['type'] == 'noSlip':
                    magSf_b = np.linalg.norm(np.array(faceSf[iFaceStart:iFaceEnd]), axis=1, ord=2)[:,np.newaxis]
                    magCf_b =np.linalg.norm(np.array(faceCf[iFaceStart:iFaceEnd]), axis=1, ord=2)[:, np.newaxis]
                    magCf_b = magCf_b*np.array(faceSf[iFaceStart:iFaceEnd])/magSf_b # for Non-orthogonal Grids
                    valueset = mu*magSf_b/magCf_b
                    Fluxes.FluxCf[iFaceStart:iFaceEnd] = valueset
                    Fluxes.FluxTf[iFaceStart:iFaceEnd] = Fluxes.FluxTf[iFaceStart:iFaceEnd]+valueset*phi[iElementStart:iElementEnd]
                #zeroGradient,symmetry,empty have no effects
            for icell in range(0, elementN):
                Coeffs.ac[icell] = Coeffs.ac[icell] + np.sum(Fluxes.FluxCf[elementFaces[icell]], axis=0)
                #ac are all +, means np.sum(mu * geoDiff_f[LinkFaces[icell]])
                AroundFace = LinkFaces[icell]
                for idx, inei in enumerate(AroundFace):
                    Coeffs.anb[icell][idx] = Coeffs.anb[icell][idx]+Fluxes.FluxFf[inei]
                Coeffs.bc[icell] = Coeffs.bc[icell] + np.sum(Fluxes.FluxTf[cellOwnFace[icell]], axis=0)-np.sum(Fluxes.FluxTf[cellNeiFace[icell]], axis=0)
        elif terms == 'PressureGradient':
            Fluxes = Initialization.Flux(faceN, elementN)
            p_grad=pfield.gradphi[0]
            volume = np.array(elementVolumes)[:,np.newaxis]
            Fluxes.FluxT = volume*p_grad[0:elementN,iComponent,np.newaxis] #∂p/∂x分别对应于ux处理,显式计算
            Coeffs.bc=Coeffs.bc-Fluxes.FluxT
        elif terms == 'Transient':
            Fluxes = Initialization.Flux(faceN, elementN)
            sys.exit("Undefined Transient Scheme")
        elif terms == 'Source': #压力梯度以外的体积力源项 body force term
            Fluxes = Initialization.Flux(faceN, elementN)
            bodyforce=9.8 #暂定，未知
            volume = np.array(elementVolumes)[:, np.newaxis]
            Fluxes.FluxT = volume * bodyforce
            Coeffs.bc = Coeffs.bc - Fluxes.FluxT#正负号不确定
        else:
            sys.exit('Undefined discretization terms')

    # Implicit Under-Relaxation Methods
    if UnderRelaxation == 'Patankar':
        Coeffs.ac = Coeffs.ac/fvSolution['relaxationFactors']['equations'][Ufield.name]
    elif UnderRelaxation == 'FalseTransient':
        Fluxes = Initialization.Flux(faceN,elementN)  # a modification of the Euler first order implicit transient method
        volume = np.array(elementVolumes)[:, np.newaxis]
        phi_old = Ufield.prevTimeStep[0:elementN, iComponent, np.newaxis]
        rho = rhofield.phi[0:elementN]
        rho_old = rhofield.prevTimeStep[0:elementN]
        falseDeltaT = 1e5
        Fluxes.FluxC = volume * rho / falseDeltaT
        Fluxes.FluxC_old = - volume * rho_old / falseDeltaT
        Fluxes.FluxT = Fluxes.FluxC * phi[0:elementN] + Fluxes.FluxC_old * phi_old
        Coeffs.ac = Coeffs.ac + Fluxes.FluxC
        Coeffs.bc = Coeffs.bc - Fluxes.FluxT  # 减去local_FluxC*phi还是因为构成残差形式
    else:
        print('No Under-Relaxation')

def Continuity():
    1+1
