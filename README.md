# Incompressible-CFD-Solver---Python
solve the imcompressible, isothermal, Newtonian fluids N-S equation (with constant viscosity flow and no energy equation)

Momentum equation

$$
\frac{\partial(\rho \phi)}{\partial t}+\nabla \cdot(\rho \mathbf{u} \phi)=\nabla \cdot\left(\mu \nabla \phi\right)+Q^\phi-\nabla p
$$

where $\phi$ represents $U_x, U_y, U_z$

Continuity equation

$$
\nabla \cdot\mathbf{u}=0
$$

Firstly, we will read the setting files<br>
calculate the mesh data, like faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes, faceCF, faceCf, faceFf, faceWeights<br>
initialize fields, Ufield,pfield<br>
set the boundary condition on the fields<br>
calculate the mass flux<br>
calculate the cell gradients and boundary face gradients<br>
discrete the Momentum equation, and solve<br>
discrete the Continuity equation, and solve<br>
