# **Incompressible-CFD-Solver---Python**
This project aims at showing how to solve the incompressible, isothermal, Newtonian Navier-Stokes equations with constant viscosity and no energy equation, while explaining the related theories and the structure of a CFD solver.

## 0. Governing equations

The physical starting point is the Navier-Stokes equations themselves. The general scalar transport equation is introduced only later for discretization; it is not the original form of the Navier-Stokes equations.

### 0.1 Mass conservation (continuity equation)

The differential and flux form of the mass conservation for any control volume $V$ is written as follows:

$$
\frac{\partial \rho}{\partial t}
+
\nabla \cdot (\rho \mathbf{u})
=0.
\tag{0.1}
$$

Here, the first term represents the local rate of density change inside the control volume, and the second term represents the net mass outflow through its boundary. $\rho$ is density and $\mathbf{u}$ is the velocity vector.

### 0.2 Momentum conservation: Newtonian fluid

The non-conservative form of the momentum equation:

$$
\rho\left(
\frac{\partial \mathbf{u}}{\partial t}
+
(\mathbf{u}\cdot\nabla)\mathbf{u}
\right)
=
\nabla\cdot\boldsymbol{\sigma}
+
\mathbf{b}.
$$

The surface force is obtained from the surface traction through Cauchy's stress theorem and the divergence theorem. Here, $\boldsymbol{\sigma}$ is the total stress tensor and $\mathbf{b}$ is the body force per unit volume. The material derivative on the left-hand side represents the acceleration of a fluid particle.

The conservative form of the momentum equation:

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
=
\nabla\cdot\boldsymbol{\sigma}
+
\mathbf{b}.
$$

where $\rho \mathbf{u u}$ is the dyadic product, and its divergence is a vector. The conservative form is commonly used in CFD since it ensures exact conservation of momentum within the numerical discretization.

The key step is to decompose the total stress tensor into pressure stress and viscous stress:

$$
\boldsymbol{\sigma}
=
-p\mathbf{I}
+
\boldsymbol{\tau}.
$$

The term $-p\mathbf{I}$ is the isotropic normal stress that would exist in a fluid at rest. The tensor $\boldsymbol{\tau}$ is the extra stress caused by fluid deformation. Taking the divergence gives

$$
\begin{aligned}
\nabla\cdot\boldsymbol{\sigma}
&=
\nabla\cdot(-p\mathbf{I}+\boldsymbol{\tau})\\
&=
-\nabla p+\nabla\cdot\boldsymbol{\tau}.
\end{aligned}
$$

The pressure part follows from the index form

$$
\left[\nabla\cdot(-p\mathbf{I})\right]_i
=
\frac{\partial}{\partial x_j}(-p\delta_{ij})
=
-\frac{\partial p}{\partial x_i}.
$$

Therefore, the momentum equation becomes

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
=
-\nabla p
+
\nabla\cdot\boldsymbol{\tau}
+
\mathbf{b}.
$$

For an isotropic Newtonian fluid, the viscous stress is linearly related to the rate of deformation. The deformation-rate tensor is
$$
\mathbf{D}
=
\frac{1}{2}
\left[
\nabla\mathbf{u}
+
(\nabla\mathbf{u})^{\mathrm{T}}
\right].
$$

Using the convention adopted in this document, the Newtonian constitutive relation is

$$
\boldsymbol{\tau}
=
2\mu\mathbf{D}
+
\lambda(\nabla\cdot\mathbf{u})\mathbf{I}
=
\mu
\left[
\nabla\mathbf{u}
+
(\nabla\mathbf{u})^{\mathrm{T}}
\right]
+
\lambda(\nabla\cdot\mathbf{u})\mathbf{I},
$$

where $\mu$ is the dynamic viscosity and $\lambda$ is the second coefficient of viscosity. Substituting this constitutive relation into the momentum equation gives

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
=
-\nabla p
+
\nabla\cdot
\left\{
\mu
\left[
\nabla\mathbf{u}
+
(\nabla\mathbf{u})^{\mathrm{T}}
\right]
\right\}
+
\nabla\cdot
\left[
\lambda(\nabla\cdot\mathbf{u})\mathbf{I}
\right]
+
\mathbf{b}.
$$

After the fluid and its operating conditions have been specified, the variation of $\mu$ and $\lambda$ over the computational domain and the considered time interval can be evaluated. If this variation is negligible for the problem of interest, a constant-property model is adopted and $\mu$ and $\lambda$ are treated as constants. Under this model, the viscous-stress divergence becomes

$$
\begin{aligned}
\nabla\cdot\boldsymbol{\tau}
&=
\mu\nabla\cdot
\left[
\nabla\mathbf{u}
+
(\nabla\mathbf{u})^{\mathrm{T}}
\right]
+
\lambda\nabla(\nabla\cdot\mathbf{u})\\
&=
\mu\nabla^2\mathbf{u}
+
\mu\nabla(\nabla\cdot\mathbf{u})
+
\lambda\nabla(\nabla\cdot\mathbf{u})\\
&=
\mu\nabla^2\mathbf{u}
+
(\mu+\lambda)\nabla(\nabla\cdot\mathbf{u}).
\end{aligned}
$$

The identities used above are

$$
\nabla\cdot(\nabla\mathbf{u})
=
\nabla^2\mathbf{u},
$$

and

$$
\nabla\cdot
\left[(\nabla\mathbf{u})^{\mathrm{T}}\right]
=
\nabla(\nabla\cdot\mathbf{u}).
$$

The constant-property Newtonian momentum equation is therefore

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
=
-\nabla p
+
\mu\nabla^2\mathbf{u}
+
(\lambda+\mu)\nabla(\nabla\cdot\mathbf{u})
+
\mathbf{b}.
\tag{0.2}
$$

This is still a vector equation and represents the three momentum equations at the same time. The stress tensor is used as an intermediate tool: first split the total stress into pressure and viscous parts, then express the viscous part through velocity gradients. The term $\mu\nabla^2\mathbf{u}$ is the shear-viscous term, while $(\lambda+\mu)\nabla(\nabla\cdot\mathbf{u})$ is the compressibility-viscous term.


### 0.3 Incompressible form

The substantial derivative of density can be written as

$$
\frac{D\rho}{Dt}
=
\frac{\partial \rho}{\partial t}
+
\mathbf{u}\cdot\nabla\rho.
$$

Using this notation, the mass conservation equation can also be written as

$$
\frac{D\rho}{Dt}
+
\rho\nabla\cdot\mathbf{u}
=0.
$$

For an incompressible fluid with constant density,

$$
\frac{D\rho}{Dt}=0.
$$

The mass conservation equation therefore reduces to

$$
\nabla\cdot\mathbf{u}=0.
\tag{0.3}
$$

The simplification path is summarized in the following diagram:
<p align="center">
  <img
    src="docs/figures/mass_conservation_incompressible.svg"
    alt="Derivation from mass conservation to incompressible continuity"
    width="400"
  >
</p>
<p align="center">
  <em>Figure 0.1. Derivation from mass conservation to incompressible continuity.</em>
</p>

Because $\nabla\cdot\mathbf{u}=0$, the compressibility-viscous term vanishes. The incompressible Navier-Stokes momentum equation is consequently

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
=
-\nabla p
+
\mu\nabla^2\mathbf{u}
+
\mathbf{b}.
\tag{0.4}
$$

The simplification path is summarized in the following diagram:
<p align="center">
  <img
    src="docs/figures/stress_decomposition_momentum.svg"
    alt="Derivation from stress decomposition to incompressible momentum"
    width="650"
  >
</p>
<p align="center">
  <em>Figure 0.2. Derivation from momentum conservation to incompressible momentum.</em>
</p>

Equations $(0.3)$ and $(0.4)$ are coupled through the velocity field $\mathbf{u}$ and pressure field $p$. Before Reynolds averaging, they form the starting incompressible Navier-Stokes system:

$$
\boxed{
\begin{aligned}
\nabla\cdot\mathbf{u} &= 0,\\
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot\{\rho \mathbf{u u}\}
&=
-\nabla p
+
\mu\nabla^2\mathbf{u}
+
\mathbf{b}.
\end{aligned}
}
\tag{0.5}
$$

For a steady incompressible flow without Reynolds averaging, the time-derivative term is removed:

$$
\nabla \cdot\{\rho \mathbf{u u}\}
=
-\nabla p
+
\mu\nabla^2\mathbf{u}
+
\mathbf{b}.
$$

The finite-volume discretization in this project starts from these mass and momentum equations. Only after this point are the equations written component by component and cast into a form suitable for discretization.

### 0.4 General transport equation

Navier-Stokes equations contain one continuity equation and three momentum component equations. Although their physical meanings are different, their mathematical structures are similar: each equation contains a transient term, a convective term, a diffusive term, and a source term. Therefore, before introducing the finite-volume discretization, it is useful to write them in one general scalar transport form:

$$
\boxed{
\underbrace{
\frac{\partial(\rho\phi)}{\partial t}
}_{\text{transient term}}
+
\underbrace{
\nabla\cdot(\rho\mathbf{u}\phi)
}_{\text{convective term}}
=
\underbrace{
\nabla\cdot(\Gamma^\phi\nabla\phi)
}_{\text{diffusive term}}
+
\underbrace{
Q^\phi
}_{\text{source term}}
}
\tag{0.6}
$$
Here, $\phi$ is the transported variable, $\Gamma^\phi$ is the diffusion coefficient, and $Q^\phi$ is the source term. Once the discretization procedure is derived for equation $(0.6)$, the same coefficient-assembly structure can be reused for different equations by changing only $\phi$, $\Gamma^\phi$, and $Q^\phi$.

<div align="center">

| Equation | $\phi$ | $\Gamma^\phi$ | $Q^\phi$ |
| :---: | :---: | :---: | :---: |
| Continuity equation | $1$ | $0$ | $0$ |
| $x$ momentum equation | $u_x$ | $\mu$ | $-\dfrac{\partial p}{\partial x}+b_x$ |
| $y$ momentum equation | $u_y$ | $\mu$ | $-\dfrac{\partial p}{\partial y}+b_y$ |
| $z$ momentum equation | $u_z$ | $\mu$ | $-\dfrac{\partial p}{\partial z}+b_z$ |

</div>

For the momentum equations, choosing $\phi=u_x$, $u_y$, or $u_z$ gives the three scalar component equations of the vector momentum equation.

## 1. Finite Volume Mesh

Before the governing equations can be discretized, the physical domain must be divided into a finite number of control volumes. This geometric discretization produces a mesh, on which the conservation equations are integrated and solved.

In the finite-volume method, each partial differential equation is integrated over a control volume. After discretization, the continuous differential equations are converted into algebraic equations. The value of a variable in one cell is linked to the values in its neighbouring cells through the shared faces. Therefore, a mesh is not only a set of points in space; it must also describe how points, faces, cells, and boundary patches are connected.

### 1.1 Finite-volume mesh data structure

A finite-volume mesh contains two types of information:

- **Geometric information**: coordinates, face centers, face areas, face normal vectors, cell centers, cell volumes, and interpolation distances.
- **Topological information**: point-to-face, face-to-cell, cell-to-neighbour, and boundary-patch connectivity.

The geometric information is used to compute surface fluxes, gradients, diffusion terms, interpolation weights, and non-orthogonal correction vectors. The topological information is used to find the owner cell, neighbour cell, boundary face, and neighbouring cells involved in each algebraic equation.

### 1.2 OpenFOAM raw mesh format

OpenFOAM stores the polyhedral mesh in the following directory:

```text
constant/polyMesh/
├── points
├── faces
├── owner
├── neighbour
└── boundary
```

These five files provide the topology and basic geometry of the mesh.

<div align="center">

| File | Function |
| :---: | :---: |
| `points` | Stores the three-dimensional coordinates of all mesh vertices |
| `faces` | Stores which point labels form each face |
| `owner` | Stores the owner cell of each face |
| `neighbour` | Stores the cell on the other side of each internal face |
| `boundary` | Stores boundary patch names, types, and face ranges |

</div>

OpenFOAM uses zero-based indexing. For example, if a face is written as

```text
4(0 1 5 4)
```

the first number `4` means that the face has four vertices, and the labels `0 1 5 4` refer to indices of the corresponding points in the `points` list. The order of these point labels is important because it determines the direction of the face area vector. The face area vector is oriented outward from the owner cell. For an internal face, this means the face normal direction points from the owner cell towards the neighbour cell.

In the `faces` file, faces are usually ordered with all internal faces first, followed by the boundary faces grouped by patch.

An internal face connects two cells. For an internal face `faceI`,

```text
owner[faceI]     = owner cell label
neighbour[faceI] = neighbour cell label
```

A boundary face connects only one real computational cell. Therefore, boundary faces have an owner cell but do not have a valid neighbour cell. The corresponding patch information is stored in the `boundary` file.

Standard OpenFOAM meshes usually do not require an independent `cells` file. Cell connectivity can be recovered from the face addressing provided by `faces`, `owner`, and `neighbour`.

<p align="center">
  <img
    src="docs/figures/OpenFOAM_mesh.svg"
    alt="OpenFOAM polyMesh topology"
    width="760"
  >
</p>

<p align="center">
  <em>Figure 1.1. Topological information stored in OpenFOAM polyMesh files.</em>
</p>

### 1.3 Reading OpenFOAM mesh files

This project does not define a new mesh file format. Instead, it directly uses the OpenFOAM `polyMesh` data structure and converts the raw OpenFOAM mesh files into the internal data needed by the finite-volume solver.

In this project, the OpenFOAM mesh is read by the `MeshDeal(workdir)` function in `ReadMesh.py`. The first step is to read the basic mesh data and mesh counts:

<div align="center">

| Variable | Meaning |
| :---: | :---: |
| `pointdata` | Coordinates of all mesh points |
| `facedata` | Point labels that define each face |
| `ownerdata` | Owner cell label of each face |
| `neighbourdata` | Neighbour cell label of each internal face |
| `boundarydata` | Boundary patch information |
| `pointN` | Number of points |
| `faceN` | Number of faces |
| `InnerfaceN` | Number of internal faces |
| `BoundaryfaceN` | Number of boundary faces |
| `cellN` | Number of cells |
| `boundaryFieldN` | Number of boundary-face entries |
| `BoundaryTypeN` | Number of boundary patches |

</div>

For each OpenFOAM list file, the reader first obtains the number of entries and then reads the following data block between the parentheses. The point coordinates and label lists are converted into NumPy arrays, while the variable-length face definitions and boundary patches are parsed into structured mesh connectivity data. Once the raw mesh data have been read, the code computes several derived connectivity lists and geometric quantities.

### 1.4 Cell-face addressing

The function

```python
MeshArrange(ownerdata, neighbourdata, cellN, InnerfaceN)
```

converts the OpenFOAM face-based addressing

```text
owner[faceID]
neighbour[faceID]
```

into cell-based addressing used by the solver:

<div align="center">

| Variable | Meaning |
| :---: | :---: |
| `cellFaces` | All faces surrounding each cell |
| `cellOwnFace` | Faces for which the cell is the owner cell |
| `cellNeiFace` | Faces for which the cell is the neighbour cell |
| `LinkFaces` | Internal faces connected to each cell |
| `cellNeighbours` | Neighbouring cell labels of each cell |

</div>

All returned lists are indexed by the cell label. Since Python and OpenFOAM use zero-based indexing, the first entry corresponds to cell `0`, the second entry corresponds to cell `1`, and so on. For example,

```text
cellFaces[0]    = [0, 1, 2, 21548, 21626, 21627]
cellOwnFace[0]  = [0, 1, 2, 21548, 21626, 21627]
cellNeiFace[0]  = []
LinkFaces[0]    = [0, 1, 2]
cellNeighbours[0] = [1, 39, 78]
```

This means that cell `0` is surrounded by faces with face IDs `0`, `1`, `2`, `21548`, `21626`, and `21627`. These numbers are face labels, not point labels. The point labels on each face are still stored in the original face-to-point connectivity list `facedata`. For example, `facedata[faceID]` gives the point labels that define one face.

The same face labels appear in `cellOwnFace[0]`, so cell `0` is the owner cell of these faces. `cellNeiFace[0]` is empty, which means cell `0` does not appear as the neighbour cell of any face. Since `neighbour` addressing only exists for internal faces, entries in `cellNeiFace` are internal faces. In this example, `LinkFaces[0]` shows that faces `0`, `1`, and `2` are the internal faces connected to cell `0`, and `cellNeighbours[0]` shows that the corresponding neighbouring cells are cells `1`, `39`, and `78`.

`LinkFaces[cellID]` stores all internal faces connected to cell `cellID`, and `cellNeighbours[cellID]` stores the neighbouring cell labels corresponding to these internal faces. The two lists are ordered by internal face label, so

```text
LinkFaces[cellID][k]
cellNeighbours[cellID][k]
```

describe the same cell-to-cell connection.

The core idea of `MeshArrange` is simple. First, the code loops over all faces once and appends each face label to initially empty cell-indexed lists:

```python
for iface, own in enumerate(ownarray):
    own_face_lists[own].append(iface)

    if iface < InnerfaceN:
        nei = neiarray[iface]

        link_face_lists[own].append(iface)
        neighbour_cell_lists[own].append(nei)

        nei_face_lists[nei].append(iface)
        link_face_lists[nei].append(iface)
        neighbour_cell_lists[nei].append(own)
```

For every face, `own_face_lists[own]` records that `own` is the owner cell of `iface`. If the face is an internal face, it also has a neighbour cell. The same face is then registered on both sides: for the owner cell, the connected neighbour is `nei`; for the neighbour cell, the connected neighbour is `own`.

Second, the temporary lists are organized into the final data form. In this step, `cellOwnFace` and `cellNeiFace` are combined and sorted to form `cellFaces`, while each cell-indexed entry is converted into an array inside a Python list.

This information is needed later when assembling finite-volume equations. For example, the algebraic equation of one cell must know which faces belong to the cell and which neighbouring cells are connected through internal faces.

### 1.5 Geometric quantities

The raw OpenFOAM mesh files provide point coordinates and topological connectivity. The solver still needs derived geometric quantities, including face centers, face area vectors, face areas, cell centers, cell volumes, and face-to-cell interpolation distances. These quantities are computed in three steps:

<div align="center">

| Function | Geometric quantities |
| :---: | :--- |
| `MeshCal1` | Face centers, face area vectors, and face areas |
| `MeshCal2` | Cell centers and cell volumes |
| `MeshCal3` | Face-to-cell vectors and interpolation weights |

</div>

#### Face geometric quantities

The first step is the face-based geometric calculation implemented by

```python
MeshCal1(pointarray, face_starts, face_nodes, faceN)
```

The returned face quantities are:

<div align="center">

| Variable | Meaning |
| :---: | :---: |
| `faceCentroids` | Face center coordinates |
| `faceSf` | Face area vector |
| `faceAreas` | Face area magnitude |

</div>

The vertices of face `faceID` are obtained from the raw face-to-point connectivity:

```python
facedata[faceID]
```

For example, if `facedata[10] = [3, 7, 8, 4]`, then face `10` is defined by points `3`, `7`, `8`, and `4`. The point coordinates themselves are stored in `pointarray`, so `pointarray[3]`, `pointarray[7]`, `pointarray[8]`, and `pointarray[4]` are the vertex coordinates of this face.

For each face, the code first obtains all point labels on the face and reads their coordinates from `pointarray`. A temporary face center $\mathbf{x}_G$ is calculated by averaging the vertex coordinates:

$$
\mathbf{x}_{G} =
\frac{1}{k}
\sum_{i=1}^{k}\mathbf{x}_i .
$$

Here, $k$ is the number of points on the face, and $\mathbf{x}_i$ is the coordinate of the $i$ th face point. This temporary center is used only to split the polygonal face into triangular parts. Each triangular part is formed by $\mathbf{x}_G$ and two adjacent face vertices, as shown in Figure 1.2.

For each triangular part, the local area vector is computed from the cross product of two edge vectors:

$$
\mathbf{S}_i =
\frac{1}{2}
\left(\mathbf{x}_i-\mathbf{x}_G\right)
\times
\left(\mathbf{x}_{i+1}-\mathbf{x}_G\right).
$$

This corresponds to the following component-wise implementation in the code:

```python
line_i = pointarray[node_i] - local_center
line_j = pointarray[node_j] - local_center
local_Sf[iline, 0] = 0.5 * (line_i[1] * line_j[2] - line_i[2] * line_j[1])
local_Sf[iline, 1] = 0.5 * (line_i[2] * line_j[0] - line_i[0] * line_j[2])
local_Sf[iline, 2] = 0.5 * (line_i[0] * line_j[1] - line_i[1] * line_j[0])
```

The total face area vector is the vector sum of all local triangular area vectors:

$$
\mathbf{S}_f =
\sum_{i=1}^{k}\mathbf{S}_i .
$$

This is why the code first performs

```python
Sf = np.sum(local_Sf, 0)
```

This vector is stored as `faceSf`. It contains both the face area magnitude and the face normal direction. The code relies on the OpenFOAM face-point ordering, so the direction follows the OpenFOAM convention: for an internal face, `faceSf` points from the owner cell toward the neighbour cell. Therefore, `faceSf` is an area vector, not a unit normal vector.

The scalar face area is the Euclidean norm of this vector:

$$
S_f = |\mathbf{S}_f|.
$$

This scalar is stored as `faceAreas`:

```python
area = sqrt(Sf[0]**2 + Sf[1]**2 + Sf[2]**2)
```

For a planar face with consistently ordered vertices, computing the scalar area as $|\sum_i \mathbf{S}_i|$ is equivalent to summing the magnitudes of the triangular areas, because all local area vectors have the same direction.

The face center is calculated as an area-weighted average of the local triangular centers:

$$
\mathbf{x}_{Fc} =
\frac{\displaystyle\sum_{i=1}^{k} S_i\mathbf{x}_{Si}}
{\displaystyle\sum_{i=1}^{k} S_i}.
$$

Here, $S_i=|\mathbf{S}_i|$ is the scalar area of the $i$ th triangular part, and $\mathbf{x}_{Si}$ is the center of that triangular part. In the code this center is stored in `local_centroid[iline]`, and the final face center is stored in `faceCentroids`. Since the code assumes a planar face, the denominator is the scalar face area `area`, which is equal to $\sum_i S_i$ for consistently oriented triangular parts.

If a unit face normal vector is needed later, it can be obtained from

```text
n_f = faceSf / faceAreas
```

The current code does not return `n_f` separately because it can be derived from `faceSf` and `faceAreas`.

<p align="center">
  <img
    src="docs/figures/Surface Area and Centroid of Faces.svg"
    alt="Polygonal face decomposition into triangular parts"
    width="500"
  >
</p>

<p align="center">
  <em>Figure 1.2. Decomposition of a polygonal face into triangular parts for face area vector and face center calculation.</em>
</p>

#### Cell geometric quantities

The function

```python
MeshCal2(cell_starts, cell_face_indices, faceCentroids, faceSf, cellN)
```

computes cell-based geometric quantities.

The returned cell quantities are:

<div align="center">

| Variable | Meaning |
| :---: | :---: |
| `cellCentroids` | Cell center coordinates |
| `cellVolumes` | Cell volumes |

</div>

For each cell, the code first collects all faces surrounding the cell and estimates a temporary cell center from the average of the face centers:

$$
\mathbf{x}_{G} =
\frac{1}{m}
\sum_{i=1}^{m}\mathbf{x}_{Fci} .
$$

Here, $m$ is the number of faces surrounding the cell, and $\mathbf{x}_{Fci}$ is the center of face $f_i$. In the code, this temporary center is stored as `local_center`:

```python
local_center = np.zeros(3)
for i in range(start, end):
    local_center += faceCentroids[cell_face_indices[i]]
local_center /= face_count
```

The cell is then decomposed into pyramids. For each surrounding face, the pyramid base is the face and the pyramid apex is the temporary cell center $\mathbf{x}_{G}$, as shown in Figure 1.3.

The vector from the temporary cell center to the face center is

$$
\mathbf{d}_{i} = \mathbf{x}_{Fci}-\mathbf{x}_{G} .
$$

This corresponds to the code:

```python
Cf = faceCentroids[iface] - local_center
```

The volume contribution associated with face $f$ is computed from the face area vector and this center-to-face vector:

$$
V_i =
\frac{1}{3}
\left|
\mathbf{S}_{fi}\cdot\mathbf{d}_{i}
\right|.
$$

This is implemented as

```python
localVolume = abs(np.sum(faceSf[iface] * Cf)) / 3
```

The total cell volume is the sum of all pyramid volume contributions:

$$
V_C =
\sum_{i=1}^{m} V_i .
$$

This scalar is stored as `cellVolumes`.

The center of each local pyramid is located on the line between the face center and the temporary cell center. For a pyramid, the center is three quarters of the way from the apex to the base center:

$$
\mathbf{x}_{Vi}
=
\frac{3}{4}\mathbf{x}_{Fci}
+
\frac{1}{4}\mathbf{x}_{G} .
$$

This corresponds to the code:

```python
localCentroid = 0.75 * faceCentroids[iface] + 0.25 * local_center
```

The final cell center is computed as the volume-weighted average of all local pyramid centers:

$$
\mathbf{x}_C =
\frac{\displaystyle\sum_{i=1}^{m} V_i\mathbf{x}_{Vi}}
{\displaystyle\sum_{i=1}^{m} V_i}.
$$

The result is stored in `cellCentroids`.

<p align="center">
  <img
    src="docs/figures/Cell Volume and Center of Cells.svg"
    alt="Polyhedral cell decomposition into pyramid volumes"
    width="560"
  >
</p>

<p align="center">
  <em>Figure 1.3. Decomposition of a polyhedral cell into local pyramid volumes for cell volume and cell center calculation.</em>
</p>


#### Face-to-cell geometric quantities

The function

```python
MeshCal3(ownerdata, neighbourdata, cellCentroids, faceCentroids, faceSf,InnerfaceN, BoundaryfaceN)
```

computes geometric vectors between face centers and cell centers. These quantities are needed for face interpolation, gradient reconstruction, boundary treatment, and non-orthogonal correction terms.

For an internal face, there is an owner cell `O` and a neighbour cell `N`. The center of the owner cell is denoted by $\mathbf{x}_O$, the center of the neighbour cell is denoted by $\mathbf{x}_N$, and the face center is denoted by $\mathbf{x}_f$. The code computes:

<div align="center">

| Variable | Meaning |
| :---: | :---: |
| `faceCF` | Vector from owner cell center to neighbour cell center |
| `faceCf` | Vector from owner cell center to face center |
| `faceFf` | Vector from neighbour cell center to face center |
| `faceWeights` | Interpolation weight at the face |

</div>

The corresponding vectors are

$$ 
\mathbf{d}_{ON}
=
\mathbf{x}_N-\mathbf{x}_O,
\qquad
\mathbf{d}_{Of}
=
\mathbf{x}_f-\mathbf{x}_O,
\qquad
\mathbf{d}_{Nf}
=
\mathbf{x}_f-\mathbf{x}_N .
$$

In the code, these are stored as

```python
faceCF[iface] = cellCentroids[nei] - cellCentroids[own]
faceCf[iface] = faceCentroids[iface] - cellCentroids[own]
faceFf[iface] = faceCentroids[iface] - cellCentroids[nei]
```

The unit face normal used in this step is obtained from the face area vector:

$$
\mathbf{n}_f
=
\frac{\mathbf{S}_f}{S_f},
\qquad
S_f=|\mathbf{S}_f|.
$$

The interpolation weight is calculated from the projected distances along $\mathbf{n}_f$, as shown in Figure 1.4. Since the OpenFOAM face area vector points from the owner cell to the neighbour cell for an internal face, $\mathbf{d}_{Of}\cdot\mathbf{n}_f$ gives the projected distance from the owner cell center to the face, while $-\mathbf{d}_{Nf}\cdot\mathbf{n}_f$ gives the projected distance from the face to the neighbour cell center.

<p align="center">
  <img
    src="docs/figures/Weight.svg"
    alt="Face interpolation weight based on projected owner-face and neighbour-face distances"
    width="230"
  >
</p>

<p align="center">
  <em>Figure 1.4. Face interpolation weight based on projected distances along the face normal direction.</em>
</p>

For the linear interpolation used by the central differencing scheme, the interpolation weight is

$$
g_f
=
\frac{\mathbf{d}_{Of}\cdot\mathbf{n}_f}
{\mathbf{d}_{Of}\cdot\mathbf{n}_f-\mathbf{d}_{Nf}\cdot\mathbf{n}_f}.
$$

This corresponds to the implementation:

```python
denom = np.dot(faceCf[iface], n) - np.dot(faceFf[iface], n)
faceWeights[iface] = np.dot(faceCf[iface], n) / denom
```

Here, $g_f$ is the interpolation weight stored in `faceWeights`, and it is the neighbour-cell weight. The face value is then written as a weighted average of the owner-cell and neighbour-cell values:

$$
\phi_f = g_f \phi_N + (1 - g_f)\phi_O .
$$

For boundary faces, there is no valid neighbour cell. Therefore, the code only computes vectors from the owner cell center to the boundary face center:

$$
\mathbf{d}_{Of}
=
\mathbf{x}_f-\mathbf{x}_O .
$$

`faceCF` is also assigned this owner-to-face vector because there is no neighbour-cell center:

```python
faceCF[iface] = faceCentroids[iface] - cellCentroids[own]
faceCf[iface] = faceCentroids[iface] - cellCentroids[own]
```

The boundary-face interpolation weight keeps its initialized value:

```python
faceWeights = 1.0
```

### 1.6 Mesh quantities returned by `MeshDeal`

The complete mesh-reading function returns both raw OpenFOAM data and derived mesh quantities:

<div align="center">

| Variable | Type of information |
| :---: | :---: |
| `pointdata` | Raw point coordinates |
| `facedata` | Raw face-to-point connectivity |
| `ownerdata` | Raw face-to-owner-cell addressing |
| `neighbourdata` | Raw internal-face neighbour addressing |
| `boundarydata` | Raw boundary patch information |
| `pointN` | Number of points |
| `faceN` | Number of faces |
| `InnerfaceN` | Number of internal faces |
| `BoundaryfaceN` | Number of boundary faces |
| `cellN` | Number of cells |
| `boundaryFieldN` | Number of boundary-face entries |
| `BoundaryTypeN` | Number of boundary patches |
| `cellNeighbours` | Neighbouring cells for each cell |
| `LinkFaces` | Internal faces linked to each cell |
| `cellFaces` | All faces surrounding each cell |
| `faceCentroids` | Face centers |
| `faceSf` | Face area vectors |
| `faceAreas` | Face area magnitudes |
| `cellCentroids` | Cell centers |
| `cellVolumes` | Cell volumes |
| `faceCF` | Owner-to-neighbour cell-center vectors |
| `faceCf` | Owner-cell-center to face-center vectors |
| `faceFf` | Neighbour-cell-center to face-center vectors |
| `faceWeights` | Face interpolation weights |
| `cellOwnFace` | Owner faces grouped by cell |
| `cellNeiFace` | Neighbour faces grouped by cell |

</div>

These precomputed quantities form the geometric and topological foundation of the finite-volume solver. Later discretization routines use them to assemble convection, diffusion, gradient, boundary, and pressure-correction terms.

### 1.7 Summary

The OpenFOAM `polyMesh` format stores the mesh in a face-based way. The raw files provide point coordinates, face definitions, owner cells, neighbour cells, and boundary patch information. This project reads those files, reconstructs cell-face connectivity, and precomputes geometric quantities needed by the finite-volume discretization.

In short:

- `points` and `faces` define the geometry of the mesh surfaces.
- `owner` and `neighbour` define internal cell connectivity.
- `boundary` defines boundary patches and boundary-face ranges.
- `ReadMesh.py` converts these raw files into solver-ready mesh data.

## 2. Discretization of convective term
The convection term in the momentum equation is non-linear, because it is velocity multiplies its own gradient $(\mathbf{u} \cdot \nabla)\mathbf{u}$ , which can't satisfy the definition of a linear operator. **Picard (fixed-point) iteration** scheme is typically used to linearize the convective term as $$(\mathbf{u}^{n} \cdot \nabla)\mathbf{u}^{n} \approx (\mathbf{u}^{n-1} \cdot \nabla)\mathbf{u}^{n}$$ where $\mathbf{u}^{n-1}$ is the velocity solution at the previous iteration. In the conservative form of the incompressible Navier–Stokes equations, the convective term can be written as $\nabla \cdot(\rho \mathbf{u} \phi)$. That is, the velocity is taken from the previous step, while flux $\phi$ is discretized implicitly, which leads to a lag in the flux information.

In order to explain how to discrete the convective term in the FVM, only convective term and source term are considered in the momentum equation:
$$
\nabla \cdot(\rho \mathbf{u} \phi)=Q^\phi
$$
After volume integrals over cell $C$ and **Gauss (divergence) theorem**,
$$
\oint_{\partial V_C} {(\rho \mathbf{u} \phi) \cdot \mathrm{d} \mathbf{S}} = \int_{V_C} Q^\phi \, dV
$$
where bold letters indicate vectors, $(\cdot)$ is the dot product operator, $\mathbf{S}$ represents the surface normal vector, and $\oint_{\partial V_C}$ is the surface integral over the volume. The Gauss theorem is applied to transform the volume integrals of the convection terms into surface integrals. Using a Gaussian quadrature and only one integration point located at the center of the face, namely **midpoint integration rule**, yielding second order accuracy, the semi-discretized equation as follows:
$$
\sum_{f} {(\rho \mathbf{u}_f \phi_f \mathbf{S}_f)} = {Q^{\phi}_C V_C} \\
{\sum_{f}\left(\dot{m}_f \phi_f \right) = Q_C^\phi V_C} \tag{1.1}
$$
where $f$ represents the faces around cell $C$, $\dot{m}_f=(\rho \mathbf{u})_f \cdot \mathbf{S}_f$ is the face mass flux, and the source term is evaluated explicitly.

Initially, the convective term is discretized using a symmetrical linear profile (central difference scheme), but the linear symmetric profile gives equal weights to the two nodes sharing the face with no directional preference. The **upwind scheme** basically mimics the basic physics of advection in that the cell face value is made dependent on the upwind nodal value, leading to physically plausible predictions, although the first order upwind profile is highly diffusive. The first order upwind scheme of convective term can be written as:
$$
\dot{m}_f \phi_f=\left \| \dot{m}_f,0 \right \|\phi_C -\left \| -\dot{m}_f,0 \right \| \phi_F
$$
where subscripts $C$ and $F$ denote the two cells connected by face $f$. Cell $C$ is the owner cell, and cell $F$ is the neighbour cell. The face area vector is defined outward from the owner cell, so for an internal face it points from cell $C$ to cell $F$. Thus, $\dot{m}_f$ is calculated using the face area vector associated with cell $C$.

Both the upwind and central difference schemes have severe limitations, the former because of its poor accuracy due to **numerical diffusion**, and the latter because of its instability also known as **numerical dispersion error**. To improve the accuracy and stability, a higher order upwind approximation of the convection term is needed. In the unstructured grid, the high order schemes in terms of the gradients are listed as follows:
$$
\text{Second Order Upwind scheme}: \phi_f= \phi_f^{Upwind}+(2 \nabla \phi_f^{Upwind}-\nabla\phi_f)\cdot \mathbf{d}_{Cf} \\
\text{QUICK scheme}: \phi_f= \phi_f^{Upwind}+\frac{1}{2} (\nabla\phi_f^{Upwind}+\nabla\phi_f) \cdot \mathbf{d}_{Cf} 
$$
where $\phi_f^{Upwind}$ denotes the upstream-cell value at face $f$, $\mathbf{d}_{Cf}$ is the distance between the face center and the upstream cell center, and the calculation of gradient $\nabla \phi_f^{Upwind}, \nabla\phi_f$ will be introduced in the following section.

Because the gradient cannot be discretized implicitly The Deferred Correction (DC) procedure of Khosla and Rubin is a compacting technique that enables the use of HO schemes in codes initially written for low order schemes without violating any of the stability rules. The approach is applicable on any type of structured or unstructured grid systems >Khosla PK, Rubin SG (1974) A diagonally dominant second-order accurate implicit scheme. Comput Fluids 2:207–209

## 3. Discretization of diffusion term

$$
{-\sum_{f}\left(\mu \nabla \phi_f \cdot \mathbf{S_f}\right)
=  V_C \mathbf{b}} \tag{1.1}
$$


## 4. Discretization of transient term
$$
\frac{\partial(\rho \phi)}{\partial t}=Q^\phi
$$

## 5. Pressure correction equation
How to use pressure-based Segregated Method to solve velocity-pressure coupling.<br>


superscript (n) denoting the initial guess or the solution at the starts of any iterationStart;<br> superscript (*) refers to intermediate values at the current iteration;<br> superscript prime (') denoting the correction field.<br>

1. Firstly, we need to discrete and solve the steady and incompressible and Newtonian fluid momtemum equation based on initial/guessed values($$\mathbf{u}^{(n)}, p^{(n)}$$)<br>

The basic momtemum euqation is shown as follows:

$$
\nabla \cdot\{\rho \mathbf{u u}\}=-\nabla p+\nabla \cdot\{\mu \nabla \mathbf{u}\}+\nabla \cdot\ {\mu(\nabla \mathbf{u})^{\mathrm{T}}}+\mathbf{b}
$$

After volume integral, Gauss theorem and midpoint integration rule of the convection, diffusion and pressure gradient term, the equation can be written as:

$$
{\sum_{f}\left(\dot{m}_f \phi_f \right)-\sum_{f}\left(\mu \nabla \phi_f \cdot \mathbf{S_f}\right)
= -V_C(\frac{\partial p^{(n)}}{\partial x_j}) + \sum_{f}\left(\mu (\nabla \phi_f^{(n)})^{\mathrm{T}} \cdot \mathbf{S}_f\right) + V_C \mathbf{b}} \tag{1.1}
$$

where $\phi_f$ represents $u_x, u_y, u_z$ in the faces of cells. The left-hand side of equation is implicitly discrated, and the right-hand side is explicitly calculated.

A HR scheme for the convection term implemented via the deferred correction approach, and decomposing the diffusion flux into an implicit part aligned with the grid and an explicit cross diffusion part, the discretized momentum equation of one cell $C$ can be written as:

$$
{a_C \mathbf{u}_C+\sum a_F \mathbf{u}_F=\mathbf{b}_C}  \tag{1.2}
$$

This algebraic equation is first solved with guessed values or values obtained from the previous iteration to obtain a momentum conserving velocity field: $$\mathbf{u}^*$$, but this velocity field doesn't satisfy the continuity equation, because of the linearization in which pressure and velocity are based on the previous iteration values.

2. And then, we need to solve the continuity (pressure) equation.<br>

The basic incompressible continuity equation and its semi-discretization form are shown as follows:

$$
\begin{aligned}
\nabla \cdot \mathbf{u} = 0 \\
\downarrow  \\
\sum_{f}\dot{m}_f = 0 \\
\mathrm{where} \quad \dot{m}_f=\rho \mathbf{u}_f \cdot \mathbf{S}_f
\end{aligned}
$$

We want to correct the velocity field $\mathbf{u}^*$ to satisfy the continuity equation, so the corrections to the velocity and pressure fields $u', p'$ are used, which means the difference between the exact and computed fields:

$$
\begin{aligned}
\mathbf{u} =\mathbf{u}^*+\mathbf{u}^{\prime} \\
p=p^{(n)}+p^{\prime}
\end{aligned}
$$

Substituting the correction terms, the continuity equation can be written as:

$$
{\sum_{f}\dot{m}_f^{\prime}=-\sum_{f}\dot{m}_f^{*}} \tag{2.1}
$$

where $\dot{m}_f^{*}$ in faces can be Rhie-Chow interpolated by the intermediate velocity filed $\mathbf{u}^*$ in cells.<br> What we need to do is just represent $\dot{m}_f^{\prime}$ in the faces by $u', p'$ in the cells, because we want to implicitlt discrete it.

According to the Rhie-Chow interpolation, the velocity in the faces can be listed as follows:

$$
\begin{aligned}
\mathbf{u_f} =\overline{\mathbf{u_f}}-\overline{\mathbf{D_f}}(\nabla p_f -\overline{\nabla p_f})\\
\mathbf{u_f}^* =\overline{\mathbf{u_f}^*}-\overline{\mathbf{D_f}}(\nabla p_f^{(n)} -\overline{\nabla p_f^{(n)}}) 
\end{aligned}
$$

Subtracting these euqations, we can get the relationship between the correct velocity in the faces and cells:

$$
{\mathbf{u}_f^{\prime} = \overline{\mathbf{u}_f^{\prime}}-\overline{\mathbf{D}_f}(\nabla p_f^{\prime}-\overline{\nabla p_f^{\prime}})} \tag{2.2}
$$

Substituting the equation (2.2) into (2.1), it can be expressed as:

$$
{\sum_{f}\rho (\overline{\mathbf{u_f}^{\prime}}+\overline{\mathbf{D_f}}\overline{\nabla p_f^{\prime}}-\overline{\mathbf{D_f}}\nabla p_f^{\prime}) \cdot \mathbf{S_f}=-\sum_{f}\dot{m}_f^{*}} \tag{2.3}
$$

To simplify the equation (2.3), another formula should be derived. The equation (1.2) can be reformulated as:

$$
\mathbf{u_C}+\mathbf{H_C}[\mathbf{u}]=-\mathbf{D_C}(\nabla p_C)+\mathbf{B_C}
$$

Substituting the corrections and intermediate values,

$$
\begin{aligned}
(\mathbf{u_C^{\ast}}+\mathbf{u_C^{\prime}})+\mathbf{H_C} [\mathbf{u^{\ast}}+\mathbf{u^{\prime}}]=\mathbf{B_C}-\mathbf{D_C}\nabla(p_C^{(n)}+p_C^{\prime}) \\
\mathbf{u_C^{\ast}} + \mathbf{H_C} [\mathbf{u^{\ast}}] = \mathbf{B_C} - \mathbf{D_C} \nabla (p_C^{(n)})
\end{aligned}
$$

Subtracting the two euqations,

$$
\mathbf{u_C^{\prime}} + \mathbf{H_C} [\mathbf{u^{\prime}}] = - \mathbf{D_C} \nabla (p_C^{\prime}) 
$$

A similar equation can be written for cell $F$, which is connected to cell $C$ by face $f$:

$$
\mathbf{u_F^{\prime}} + \mathbf{H_F} [\mathbf{u^{\prime}}] = - \mathbf{D_F} \nabla (p_F^{\prime}) 
$$

Thus, interpolating the above equations, the velocity corrections in the faces can be:

$$
\begin{aligned}
\overline{\mathbf{u_f^{\prime}}} + \overline{\mathbf{H_f}} [\mathbf{u^{\prime}}] = - \overline{\mathbf{D_f}} \overline{\nabla (p_f^{\prime})}\\
\Downarrow \\
{\overline{\mathbf{u_f^{\prime}}}+ \overline{\mathbf{D_f}} \overline{\nabla (p_f^{\prime})}=-\overline{\mathbf{H_f}} [\mathbf{u^{\prime}}]}
\end{aligned}
$$

Substituting it in Equation (2.3), the pressure correction equation is rewritten as

$$
{\sum_{f} -\rho \overline{\mathbf{D_f}} (\nabla p_f^{\prime}) \cdot \mathbf{S_f}=-\sum_{f}\dot{m}_f^{*} + \sum_{f} \rho \overline{\mathbf{H_f}} [\mathbf{u^{\prime}}] \cdot \mathbf{S_f}} \tag{2.4}
$$

In the original SIMPLE algorithm, $\sum_{f}\rho \overline{\mathbf{H_f}} [\mathbf{u^{\prime}}] \cdot \mathbf{S_f}$ is neglected, because the corrections will become zero at convergence. For this correction equation, the modification or dropping of the term will affect the convergence rate rather than the final solution.

Besides, $\overline{\mathbf{D_f}} (\nabla p_f^{\prime}) \cdot \mathbf{S_f}$ in Equation (2.4) can be further simplified:

$$
\begin{aligned}
\overline{\mathbf{D_f}} (\nabla p_f^{\prime}) \cdot \mathbf{S_f}= (\nabla p_f^{\prime}) \cdot (\overline{\mathbf{D_f}}^{\mathrm{T}} \cdot \mathbf{S_f}) \\
=(\nabla p_f^{\prime}) \cdot \mathbf{S_f^{\prime}} \\
=(\nabla p_f^{\prime}) \cdot (\mathbf{E_f^{\prime}}+\mathbf{T_f^{\prime}})
\end{aligned}
$$

where, the orthogonal contribution can be implicitlt discreted as the form of $p_C^{\prime}$, and the cross-diffusion or non-orthogonal contributioin can only be calculated explicitly:

$$
{{\sum_{f} -\rho \frac{E_f}{d_{CF}}  (p_F^{\prime}-p_C^{\prime})=-\sum_{f}\dot{m}_f^{*} + \sum_{f} \rho \overline{\mathbf{H_f}} [\mathbf{u^{\prime}}] \cdot \mathbf{S_f}}+\sum_{f} \rho (\nabla p_f^{\prime}) \cdot \mathbf{T_f^{\prime}}} \tag{2.5}
$$

Samely, $(\nabla p_f^{\prime}) \cdot \mathbf{T_f^{\prime}}$ can be neglected, because it is a correction term. And if it is treated explicitly, that is the non-orthogonal loop in OpenFOAM.

### 0.5 Solver workflow

The programming process follows the finite-volume solution loop:

1. Read the setting files.
2. Calculate mesh geometry data, including `faceCentroids`, `faceSf`, `faceAreas`, `cellCentroids`, `cellVolumes`, `faceCF`, `faceCf`, `faceFf`, and `faceWeights`.
3. Initialize the fields, such as `Ufield` and `pfield`.
4. Apply boundary conditions to the fields.
5. Calculate the face mass flux.
6. Calculate cell gradients and boundary-face gradients.
7. Discretize and solve the momentum equation.
8. Discretize and solve the pressure-correction / continuity equation.
9. Check residuals and repeat the iteration loop until convergence.

The following sections introduce the mesh data, boundary conditions, gradients, and the discretization of each term in the momentum and pressure-correction equations.
