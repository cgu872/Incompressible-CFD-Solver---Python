# -*- coding:utf-8 -*-
# @data 2025/3/27
import sys
import numpy as np

# usage
# Interpolate.ElementsToFaces('linear', U, own, nei, fweights)
# Interpolate.ElementsToFaces('linearUpwind', U, own, nei, fweights,mdot_f[0:InnerfaceN])
def ElementsToFaces(InterpolationScheme,phi,own,nei,fweights,mdot_innerf=None):
    if InterpolationScheme=='linear':
        phi_f = fweights * phi[nei] + (1 - fweights) * phi[own]
    elif InterpolationScheme=='linearUpwind':
        ownindex=np.array(own)[np.where(mdot_innerf>=0)]
        neiindex = np.array(nei)[np.where(mdot_innerf < 0)]
        phi_f = np.zeros((len(mdot_innerf),phi.shape[1]))
        phi_f[np.where(mdot_innerf >= 0)] = phi[ownindex]
        phi_f[np.where(mdot_innerf < 0)] = phi[neiindex]
    elif InterpolationScheme=='vanLeerV':
        sys.exit("Undefined Interpolation Scheme.")
    else:
        sys.exit("Undefined Interpolation Scheme.")
    return phi_f
# 只导入内部面相关的数据，所以一些数据需要进行截取

# usage
# Interpolate.GradientElementsToFaces('Gauss linear',gradphi,phi,own,nei,faceCF,fweights)
# Interpolate.GradientElementsToFaces('Gauss upwind',gradphi,phi,own,nei,faceCF,fweights,mdot_f[0:InnerfaceN])
def GradientElementsToFaces(InterpolationScheme,gradphi,phi,own,nei,faceCF,fweights,mdot_innerf=None):
    fweights=np.array(fweights)[:,np.newaxis]
    if InterpolationScheme=='linear' or InterpolationScheme=='Gauss linear':
        gradphi_f=fweights*gradphi[nei]+(1-fweights)*gradphi[own]
    elif InterpolationScheme=='Gauss linear corrected':
        gradphi_f=fweights*gradphi[nei]+(1-fweights)*gradphi[own]
        CF=np.linalg.norm(faceCF, axis=1, ord=2)[:,np.newaxis]
        e_CF = faceCF/CF
        Correction=((phi[nei]-phi[own])/CF-np.einsum('ij,ij->i',gradphi_f,e_CF)[:,np.newaxis])*e_CF
        gradphi_f=gradphi_f+Correction
    elif InterpolationScheme == 'Gauss upwind':
        pos = np.zeros_like(mdot_innerf)
        pos[mdot_innerf>=0]=1
        gradphi_f=pos*gradphi[own]+(1-pos)*gradphi[nei]
    else:
        sys.exit("Undefined Interpolation Scheme.")
    return gradphi_f