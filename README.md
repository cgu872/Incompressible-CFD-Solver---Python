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
{\sum_{f}\left(\dot{m_f} \phi_f \right)-\sum_{f}\left(\mu \nabla \phi_f \cdot \mathbf{S_f}\right)
= -V_C(\frac{\partial p^{(n)}}{\partial x_j}) + \sum_{f}\left(\mu (\nabla \phi_f^{(n)})^{\mathrm{T}} \cdot \mathbf{S}_f\right) + V_C \mathbf{f}_b} \tag{1.1}
$$

where $\phi_f$ represents $u_x, u_y, u_z$ in the faces of cells. The left-hand side of equation is implicitly discrated, and the right-hand side is explicitly calculated.

A HR scheme for the convection term implemented via the deferred correction approach, and decomposing the diffusion flux into an implicit part aligned with the grid and an explicit cross diffusion part, 
the discretized momentum equation of one cell $C$ can be written as:

$$
{a_C \mathbf{u}_C+\sum a_F \mathbf{u}_F=\mathbf{b}_C}  \tag{1.2}
$$

This algebraic equation is first solved with guessed values or values obtained from the previous iteration to obtain a momentum conserving velocity field: $$\mathbf{u}^*$$, but this velocity field doesn't satisfy the
continuity equation, because of the linearization in which pressure and velocity are based on the previous iteration values.

2. And then, we need to solve the continuity (pressure) equation.<br>

The basic incompressible continuity equation and its semi-discretization form are shown as follows:

$$
\begin{aligned}
\nabla \cdot \mathbf{u} = 0 \\
\downarrow  \\
\sum_{f}\dot{m_f} = 0 \\
\mathrm{where} \quad \dot{m_f}=\rho \mathbf{u}_f \cdot \mathbf{S}_f
\end{aligned}
$$

We want to correct the velocity field $\mathbf{u}^*$ to satisfy the continuity equation, so the corrections to the velocity and pressure fields  $u', p'$ are used, which means the difference between the exact and computed fields:

$$
\begin{aligned}
\mathbf{u} =\mathbf{u}^*+\mathbf{u}^{\prime} \\
p=p^{(n)}+p^{\prime}
\end{aligned}
$$

Substituting the correction terms, the continuity equation can be written as:

$$
{\sum_{f}\dot{m_f}^{\prime}=-\sum_{f}\dot{m_f}^{*}} \tag{2.1}
$$

where $\dot{m_f}^{*}$ in faces can be Rhie-Chow interpolated by the intermediate velocity filed $\mathbf{u}^\*$ in cells.<br>
What we need to do is just represent $\dot{m_f}^{\prime}$ in the faces by $u', p'$ in the cells, because we want to implicitlt discrete it.

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
{\sum_{f}\rho (\overline{\mathbf{u_f}^{\prime}}+\overline{\mathbf{D_f}}\overline{\nabla p_f^{\prime}}-\overline{\mathbf{D_f}}\nabla p_f^{\prime}) \cdot \mathbf{S_f}=-\sum_{f}\dot{m_f}^{*}} \tag{2.3}
$$

To simplify the equation (2.3), another formula should be derived. The equation (1.2) can be reformulated as:

$$
\mathbf{u_C}+\mathbf{H_C}[\mathbf{u}]=-\mathbf{D_C}(\nabla p_C)+\mathbf{B_C}
$$

Substituting the corrections and intermediate values,

$$
\mathbf{u_C^*}+\mathbf{H_C}\left [\mathbf{u_C^*}  \right ]=\mathbf{B_C}-\mathbf{D_C}\nabla(p_C^{(n)})
$$

$$
\mathbf{H_C}{[\mathbf{u_C^*}]}
$$


