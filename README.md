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
<br>



-------------------------------------------------------------------------------
How to use pressure-based Segregated Method to solve velocity-pressure coupling.<br>


superscript (n) denoting the initial guess or the solution at the starts of any iterationStart;<br>
superscript (*) refers to intermediate values at the current iteration;<br>
superscript prime (') denoting the correction field.<br>

1. Firstly, we need to discrete and solve the steady and incompressible momtemum equation based on initial/guessed values($$\mathbf{u}^{(n)}, p^{(n)}$$)<br>

The basic momtemum euqation is shown as follows:

$$
\nabla \cdot\{\rho \mathbf{u u}\}=-\nabla p+\nabla \cdot\{\mu \nabla \mathbf{u}\}+\nabla \cdot\{\mu(\nabla \mathbf{u})^{\mathrm{T}}}+\mathbf{f}_b
$$

After volume integral, Gauss theorem and midpoint integration rule of the convection, diffusion and pressure gradient term, the equation can be written as:

$$
\sum_{f}\left(\dot{m_f} \phi_f \right)-\sum_{f}\left(\mu \nabla \phi_f \cdot \mathbf{S_f}\right)
= -V_C(\nabla p_C^{(n)}) + \sum_{f}\left(\mu (\nabla \phi_f^{(n)})^{\mathrm{T}} \cdot \mathbf{S}_f\right) + V_C \mathbf{f}_b
$$

where $\phi_f$ represents $u_x, u_y, u_z$ in the faces of cells.

A HR scheme for the convection term implemented via the deferred correction approach, and decomposing the diffusion flux into an implicit part aligned with the grid and an explicit cross diffusion part, 
the discretized momentum equation of one cell $C$ can be written as:

$$
a_C \mathbf{u}_C+\sum a_F \mathbf{u}_F=\mathbf{b}_C
$$

This algebraic equation is first solved with guessed values or values obtained from the previous iteration to obtain a momentum conserving velocity field: $$\mathbf{u}^*$$, but this velocity field doesn't satisfy the
continuity equation, because of the linearization in which pressure and velocity are based on the previous iteration values.

2. And then, we need to solve the continuity (pressure) equation.<br>

The basic incompressible ontinuity equation is shown as follows:

$$
\begin{aligned}
\nabla \cdot \mathbf{u} = 0
\sum_{f}\dot{m_f} = 0
\end{aligned}
$$
And after semi-discretization it can be written as:
$$
\sum_{f}\dot{m_f} = 0
$$

we want to correct the velocity field $\mathbf{u}^*$ to satisfy the continuity equation, so the corrections to the velocity and pressure fields  $u', p'$ are used, which means the difference between the exact and computed fields:

$$
\begin{aligned}
\mathbf{u} =\mathbf{u}^*+\mathbf{u}^{\prime} \\
p=p^{(n)}+p^{\prime}
\end{aligned}
$$

So 
