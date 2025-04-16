# -*- coding:utf-8 -*-
# @data 2025/3/17
# @file ReadMesh.py with @jit(nopython=True)

import sys,re,os
import numpy as np
from itertools import islice
from numba import jit, typed

def MeshDeal(workdir):
    os.chdir(workdir)

    #1.-------------------------------------------------------#
    # read [pointdata,facedata,ownerdata,neighbourdata,boundarydata,
    #       pointN,faceN,InnerfaceN,BoundaryfaceN,
    #       elementN,elementBN,BoundaryTypeN]
    # -------------------------------------------------------#
    meshdir=os.path.join(workdir,"constant\\polyMesh")
    if not os.path.exists(meshdir):
        sys.exit("There is no OpenFOAM mesh.")
    else:
        pointfile=os.path.join(meshdir,"points")
        pointdata=[]
        with open(pointfile, 'r', encoding='utf-8') as file:
            lines = [next(file).strip() for _ in range(50)]  # 只读取前50行
        pointNN = [line for line in lines if re.fullmatch(r'^\s*\d+\s*$', line)]
        if not (len(pointNN)==1):
            sys.exit("Fail to read points file .")
        pointN = int(pointNN[0])
        with open(pointfile, "r", encoding="utf-8") as file:
            content = [line.strip() for line in file.readlines()] #按行列表，无换行符\n
        pointdata = [np.fromstring(line.replace("(", "").replace(")", ""), sep=" ") for line in content if re.fullmatch(r'^\(.*\)$',line)]
        if not pointN==len(pointdata):
            sys.exit("Fail to read points file .")

        facefile=os.path.join(meshdir,"faces")
        facedata=[]
        with open(facefile, "r", encoding="utf-8") as file:
            lines = [next(file).strip() for _ in range(50)]
        faceNN = [line for line in lines if re.fullmatch(r'^\s*\d+\s*$', line)]
        if not (len(faceNN)==1):
            sys.exit("Fail to read faces file .")
        faceN = int(faceNN[0])    #number of the total faces
        with open(facefile, "r", encoding="utf-8") as file:
            content = [line.strip() for line in file.readlines()]
        facedata = [np.fromstring(re.search(r'\((.*?)\)', line).group(1), sep=" ", dtype=int) for line in content if re.fullmatch(r'\d\(.*\)$', line)]
        if not faceN==len(facedata):
            sys.exit("Fail to read faces file .")

        ownerfile = os.path.join(meshdir, "owner")
        with open(ownerfile, "r", encoding="utf-8") as file:
            lines = [next(file).strip() for _ in range(50)]
        Numindex = next((i for i, line in enumerate(lines) if line == "("), -1)-1
        faceNN = int(lines[Numindex])   #Verify again
        if not faceN==faceNN:
            sys.exit("Fail to read owner file .")
        with open(ownerfile, "r", encoding="utf-8") as file:
            lines = file.readlines()
        ownerdata=list(map(int, lines[Numindex+2:Numindex+2+faceN]))

        neighbourfile = os.path.join(meshdir, "neighbour")
        with open(neighbourfile, "r", encoding="utf-8") as file:
            lines = [next(file).strip() for _ in range(50)]
        Numindex = next((i for i, line in enumerate(lines) if line == "("), -1)-1
        InnerfaceN = int(lines[Numindex])    #number of the inner faces
        BoundaryfaceN=faceN-InnerfaceN       #number of the boundary faces
        with open(neighbourfile, "r", encoding="utf-8") as file:
            lines = file.readlines()
        neighbourdata=list(map(int, lines[Numindex+2:Numindex+2+InnerfaceN]))
        elementN=len(set(ownerdata)) #number of the elements or np.max(ownerdata)+1 because the numpy comes from 0
        elementBN = BoundaryfaceN        #number of the elements including boundary faces

        boundaryfile=os.path.join(meshdir,"boundary")
        boundarydata= []
        with open(boundaryfile, "r", encoding="utf-8") as file:
            lines = [next(file).strip() for _ in range(50)]
        Numindex = next((i for i, line in enumerate(lines) if line == "("), -1) - 1
        BoundaryTypeN = int(lines[Numindex])
        with open(boundaryfile, "r", encoding="utf-8") as file:
            next(islice(file, Numindex-1, Numindex), None)
            content = file.read()
        pattern = re.compile(r'(\w+)\s*\{([^}]*)\}', re.DOTALL)
        matches = pattern.findall(content)
        for match in matches:
            bound_name = match[0]
            bound_content = match[1].strip()
            bound_dict = {}
            for line in bound_content.splitlines():
                line=line.strip(";")
                key=line.strip().split()
                bound_dict[key[0]] = key[1]
            bound_dict['name'] = bound_name
            boundarydata.append(bound_dict)
    del lines, file, match, matches
    elementNeighbours, LinkFaces, elementFaces,cellOwnFace,cellNeiFace = MeshArrange(np.array(ownerdata), np.array(neighbourdata),elementN,InnerfaceN)
    faceCentroids, faceSf, faceAreas = MeshCal1(np.array(pointdata), typed.List(facedata), np.array(ownerdata),typed.List(elementFaces),faceN,elementN)
    elementCentroids, elementVolumes = MeshCal2(typed.List(elementFaces),np.array(faceCentroids),np.array(faceSf),elementN)
    faceCF, faceCf, faceFf, faceWeights = MeshCal3(ownerdata, neighbourdata,elementCentroids, faceCentroids, faceSf, InnerfaceN,BoundaryfaceN)
    return (pointdata, facedata, ownerdata, neighbourdata, boundarydata,
            pointN, faceN, InnerfaceN, BoundaryfaceN, elementN, elementBN, BoundaryTypeN,
            elementNeighbours, LinkFaces, elementFaces,
            faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes,
            faceCF, faceCf, faceFf, faceWeights,
            cellOwnFace, cellNeiFace)

@jit(nopython=True)
def MeshArrange(ownarray, neiarray,elementN,InnerfaceN):
    # 2.-------------------------------------------------------#
    # rewrite data based on cell index
    # elementFaces，elementNeighbours
    # upperAnbCoeffIndex, lowerAnbCoeffIndex
    # -------------------------------------------------------#
    elementFaces,elementNeighbours=[],[]
    LinkFaces=[]
    cellOwnFace,cellNeiFace=[],[]
    for icell in range(0,elementN):
        ownfaces=np.where(ownarray == icell)[0]
        neifaces=np.where(neiarray == icell)[0]
        cellOwnFace.append(ownfaces)
        cellNeiFace.append(neifaces)
        allfaces=np.sort(np.concatenate((ownfaces, neifaces)))
        innerfaces=np.concatenate((ownfaces[ownfaces<InnerfaceN], neifaces))
        sorted_indices = np.argsort(innerfaces)
        innerfaces=innerfaces[sorted_indices]
        neicells=np.concatenate((neiarray[ownfaces[ownfaces<InnerfaceN]],ownarray[neifaces]))
        neicells=neicells[sorted_indices]
        elementFaces.append(allfaces)
        LinkFaces.append(innerfaces)
        elementNeighbours.append(neicells)
    return elementNeighbours, LinkFaces, elementFaces,cellOwnFace,cellNeiFace

@jit(nopython=True)
def MeshCal1(pointarray, facedata,ownerdata,elementFaces,faceN,elementN):
    # 3.-------------------------------------------------------#
    # calculate face Centroids,face normal vector, Areas, Cell Centroids,Volumes
    # faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes
    # faceWeights, faceCF, faceCf, faceFf
    # -------------------------------------------------------#
    faceCentroids,faceSf,faceAreas=[],[],[]
    for iface in range(0, faceN):
        NodeIndex=facedata[iface]
        # rough face centroid
        local_centre = np.sum(pointarray[NodeIndex],0)/len(NodeIndex)
        point = pointarray[NodeIndex]
        line = point - local_centre
        line = np.vstack((line,line[0,:].reshape(1, -1)))
        point = np.vstack((point, point[0, :].reshape(1, -1)))
        local_Sf,local_centroid=np.empty((0,3)),np.empty((0,3))
        for iline in range(0,len(NodeIndex)):
            local_Sf=np.vstack((local_Sf,0.5 * np.cross(line[iline], line[iline+1]).reshape(1, -1)))
            local_centroid=np.vstack((local_centroid,(local_centre+point[iline]+point[iline+1]).reshape(1, -1)/3))
        #surface normal vector
        Sf = np.sum(local_Sf,0)
        area=np.linalg.norm(Sf, ord=2) #Euclidean norm
        centroid = np.zeros((3,))
        for ipart in range(0,len(NodeIndex)):
            centroid=centroid+np.linalg.norm(local_Sf[ipart],ord=2) * local_centroid[ipart]
        centroid = centroid/area
        faceCentroids.append(centroid)
        faceSf.append(Sf)
        faceAreas.append(area)
    return faceCentroids, faceSf, faceAreas

@jit(nopython=True)
def MeshCal2(elementFaces,faceCentroids,faceSf,elementN):
    elementCentroids, elementVolumes = [], []
    for icell in range(0, elementN):
        FaceIndex = elementFaces[icell]
        AroFaceCen=faceCentroids[FaceIndex]
        local_centre = np.sum(AroFaceCen,0)/len(FaceIndex)
        Cf=AroFaceCen-local_centre
        local_Sf = faceSf[FaceIndex]
        localVolume = np.abs(np.sum(local_Sf*Cf,1))/3
        totalVolume = np.sum(localVolume)
        localCentroid=0.75 * AroFaceCen + 0.25 * local_centre
        realCentroids=np.sum(localCentroid*localVolume.reshape(-1,1),0)/totalVolume
        elementCentroids.append(realCentroids)
        elementVolumes.append(totalVolume)
    return elementCentroids, elementVolumes

def MeshCal3(ownerdata, neighbourdata,elementCentroids, faceCentroids, faceSf, InnerfaceN,BoundaryfaceN):
    faceCF, faceCf, faceFf, faceWeights = [],[],[],[]
    for iface in range(0, InnerfaceN):
        n=faceSf[iface]/np.linalg.norm(faceSf[iface], ord=2)
        own=ownerdata[iface]
        nei=neighbourdata[iface]
        faceCF.append(elementCentroids[nei]-elementCentroids[own])
        faceCf.append(faceCentroids[iface] - elementCentroids[own])
        faceFf.append(faceCentroids[iface] - elementCentroids[nei])
        faceWeights.append(np.dot(faceCf[iface],n)/(np.dot(faceCf[iface],n) - np.dot(faceFf[iface],n)))
    for iface in range(InnerfaceN, InnerfaceN+BoundaryfaceN):
        own = ownerdata[iface]
        faceCF.append(faceCentroids[iface]-elementCentroids[own]) #no F in the boundary
        faceCf.append(faceCentroids[iface] - elementCentroids[own])
        faceWeights.append(1.0)
    return faceCF, faceCf, faceFf, faceWeights

