# Incompressible-CFD-Solver---Python
In this project, I want to show how to Solve the imcompressible, isothermal, Newtonian fluids N-S equation (with constant viscosity flow and no energy equation) and explain the related theories. Help understand how the CFD software works.

Momentum equation

$$
\frac{\partial(\rho \phi)}{\partial t}+\nabla \cdot(\rho \mathbf{u} \phi)=\nabla \cdot\left(\mu \nabla \phi\right)+Q^\phi-\nabla p
$$

where $\phi$ represents $U_x, U_y, U_z$. $\Gamma^\phi$ can be directly replaced with constant dynamic viscosity $\mu$. And $Q^\phi$ is generally setted as 0.

Continuity equation

$$
\nabla \cdot\mathbf{u}=0
$$


Programming process:<br>
1. Firstly, read the setting files<br>
2. calculate the mesh data, like faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes, faceCF, faceCf, faceFf, faceWeights<br>
3. initialize fields, like Ufield,pfield<br>
4. set the boundary condition on the fields<br>
5. calculate the mass flux<br>
6. calculate the cell gradients and boundary face gradients<br>
7. discrete the Momentum equation, and solve<br>
8. discrete the Continuity equation, and solve<br>
9. iterations and Residuals


Discretization of convective term and diffusion term in the Momentum equation.<br>
