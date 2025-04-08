# -*- coding:utf-8 -*-
# @data 2025/3/17
# @file MeshDeal.py
import copy
import sys,re,os
import numpy as np
from itertools import islice
# from numba import jit

# @jit(nopython=True)
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

    # 2.-------------------------------------------------------#
    # rewrite data based on cell index
    # elementFaces，elementNeighbours
    # upperAnbCoeffIndex, lowerAnbCoeffIndex
    # -------------------------------------------------------#
    ownarray=np.array(ownerdata)
    neiarray=np.array(neighbourdata)
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
    '''
    elementNeighbours=[[] for _ in range(elementN)]
    elementFaces=[[] for _ in range(elementN)]
    for iface in range(0,InnerfaceN):
        own=ownerdata[iface]
        nei=neighbourdata[iface]
        elementNeighbours[own].append(nei) #the face index owned by cell
        elementNeighbours[nei].append(own) #the cell index near cell
        elementFaces[own].append(iface)    #比如根据cell编号找周围的面构成[facedata[i] for i in elementFaces[0]]
        elementFaces[nei].append(iface)
    LinkFaces=copy.deepcopy(elementFaces)
    for iface in range(InnerfaceN, faceN):
        own = ownerdata[iface]
        elementFaces[own].append(iface)   #boundary faces in cell
    cellOwnFace = [np.where(np.array(ownerdata)==icell)[0] for icell in range(0,elementN)]
    cellNeiFace = [np.where(np.array(neighbourdata)==icell)[0] for icell in range(0,elementN)]
    '''
    '''
    upperAnbCoeffIndex=[[] for _ in range(InnerfaceN)]#记录了每个内部面在其owner的elementNeighbours中的编号
    lowerAnbCoeffIndex=[[] for _ in range(InnerfaceN)]#在其neighbour的elementNeighbours中的编号
    index=np.where(np.array(elementNeighbours[own])==nei)
    anb[own][index]= + FluxFf(iFace)
    index=np.where(np.array(elementNeighbours[nei])==own)
    anb[nei][index]= - FluxFf(iFace)
    # cell 包含的point
    elementNodes = [[] for _ in range(elementN)]
    for icell in range(0,elementN):
        [facedata[i] for i in elementFaces[icell]]
    '''

    # 3.-------------------------------------------------------#
    # calculate face Centroids,face normal vector, Areas, Cell Centroids,Volumes
    # faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes
    # faceWeights, faceCF, faceCf, faceFf
    # -------------------------------------------------------#
    faceCentroids,faceSf,faceAreas=[],[],[]
    for iface in range(0, faceN):
        centroid = np.zeros(3)
        Sf = np.zeros(3)#surface normal vector
        area = 0
        NodeIndex=facedata[iface]
        local_centre = np.zeros(3) #rough face centroid
        for iNode in NodeIndex:
            local_centre = local_centre + pointdata[iNode]
        local_centre=local_centre/len(NodeIndex)
        line=[pointdata[iTriangle]-local_centre for iTriangle in NodeIndex]
        line.append(line[0])
        point=[pointdata[iTriangle] for iTriangle in NodeIndex]
        point.append(point[0])
        local_Sf = [0.5 * np.cross(line[iline], line[iline+1]) for iline in range(0,len(NodeIndex))]
        local_centroid=[(local_centre+point[iline]+point[iline+1])/3 for iline in range(0,len(NodeIndex))]
        Sf = np.sum(local_Sf,0)
        area=np.linalg.norm(Sf, ord=2) #Euclidean norm
        centroid=[np.linalg.norm(local_Sf[iTriangle], ord=2)*local_centroid[iTriangle] for iTriangle in range(0,len(NodeIndex))]
        centroid=np.sum(centroid,0)/area
        faceCentroids.append(centroid)
        faceSf.append(Sf)
        faceAreas.append(area)
    elementCentroids, elementVolumes = [], []
    for icell in range(0, elementN):
        FaceIndex = elementFaces[icell]
        local_centre=np.average([faceCentroids[i] for i in FaceIndex], 0) #rough cell centroid
        Cf=[faceCentroids[i] - local_centre for i in FaceIndex]
        local_Sf=[faceSf[i] if icell == ownerdata[i] else -faceSf[i] for i in FaceIndex]
        localVolume = [np.dot(local_Sf[i],Cf[i])/3 for i in range(0, len(FaceIndex))]
        totalVolume = np.sum(localVolume,0)
        localCentroid = [0.75 * faceCentroids[i] + 0.25 * local_centre for i in FaceIndex]
        realCentroids=np.sum([localCentroid[i]*localVolume[i] for i in range(0, len(FaceIndex))],0)/totalVolume
        elementVolumes.append(totalVolume)
        elementCentroids.append(realCentroids)
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
        n = faceSf[iface] / np.linalg.norm(faceSf[iface], ord=2)
        own = ownerdata[iface]
        faceCF.append(faceCentroids[iface]-elementCentroids[own]) #no F in the boundary
        faceCf.append(faceCentroids[iface] - elementCentroids[own])
        faceWeights.append(1.0)
    return (pointdata, facedata, ownerdata, neighbourdata, boundarydata,
            pointN,faceN,InnerfaceN,BoundaryfaceN,elementN,elementBN,BoundaryTypeN,
            elementNeighbours, LinkFaces, elementFaces,
            faceCentroids,faceSf,faceAreas,elementCentroids, elementVolumes,
            faceCF, faceCf, faceFf, faceWeights,
            cellOwnFace,cellNeiFace)