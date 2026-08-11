**Libraries and Requirements**

Python 3.14.3
SDL 2.32.10

numpy     2.5.1
pip       26.2
pygame-ce 2.5.7
PyOpenGL  3.1.10
scipy     1.18.0

slowkenuinely just use claude to make a requirment.txt file to install these, idk with venv, library and terminal stuff

put the shader text files into a file names "shaders" or remove all shaders/

**idk how to make this easily usable and if there's some way to easily run from github so just download**

**Schrodinger Equation stuff**

If you're actually trying to learn about this don't trust me. There's probably a bunch of mistakes in this explanation.

A **wavefunction** (ψ) represents the probability of a particle existing at some point over all of space. Normally the wavefun
ction is complex and the probability function is given by the **square modulus** ( (a+bi)*(a-bi) ) of the wavefunction but this
method produces real wavefunctions whose probability is just the square of the wavefunction at each point.

A wavefunction is defined by two things:
  -  the total probability of the particle existing anywhere is 1 so the probability function is normalized, meaning its sum
     over all space is equal to 1
  -  the **Schrodinger Equation** (H*ψ = E*ψ) which I imagine as representing the **quantized nature of energy** and **conservation
     of energy**

**H** is the **Hamiltonian Operator** (a function with an input of the wavefunction ψ and an output of the energy at each position
in space) made of the kinetic energy part -(d2/dx2+d2/dy2+d2/dz2)ψ(x,y,z) and the potential energy part V(x,y,z)*ψ(x,y,z)
where V(x,y,z) represents the stabilizing **Coulombic interaction** between two charged ions. The **Born-Oppenheimer Approximantion**
says that nucleons are so much more massive than electrons that they are functionally stationary and can be represented by a single
**potential energy function** and not multiple interacting wavefunctions. (More nuclei can be added by changing the exPotential function.)

**E** is some constant with the physical meaning of the particle's total energy. Only some energy values with give a solution
which represents that energy is quantized.

This means the wavefunction (ψ) is a function that is only scaled by a constant when acted on by the Hamiltonian which is
what an **eigenfunction** is in Linear Algebra. **Eigenvalues** and eigenfunctions come in pairs and are specific to given matrices
so if the Hamiltonian Operator and wavefunction can be represented as a matrix and a vector we can use computer eigenvalue
solvers to find energy values and their associated wavefunctions.

The first step is shrinking the range from -∞ to +∞ to a finite cube which physically represents the particle being stuck
in a box giving another boundary condition: the probability of the particle existing at the edges is 0 so those points in the matrix
and vector are set to 0.

To represent the function within an N point by N point by N point box it could be a 3d matrix but those are unoptimized for
computers so it is best to flatten it into a N^3 long vector.

The Hamiltonian is represented as a N^3 by N^3 vector with the **Finite Difference Method** to approximate the second derivative.
(If you get finite difference method and matrix multiplication this is pretty straight forward but I can't explain it here.)

**Calculating and Rendering**

I have a toggle to calculate new wavefunctions, set calculate = True. It will calculate the lowest n energy values given by
eigens = n.

For rendering, I sum the values and normalize them by dividing everything by the sum then only render points whose probability
sums to a given threshold (I've set to 95%). I have a couple different rendering types, the last two are the same cause I'm
going to add specular and diffuse lighting.

**DFT or Density Functional Theory**

This is my final goal that I want to implement. I'm too lazy to explain it though.
