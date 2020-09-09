.. default-domain:: py

.. _parameters:

DFT-FE Input Parameter File Reference
=====================================

Global parameters
-----------------

.. data:: H REFINED ELECTROSTATICS

   :value: true
   :type:  true | false

   [Advanced] Compute electrostatic energy and forces on a h refined mesh after each ground-state solve. Default: true if cell stress computation is set to false otherwise it is set to false.

.. data:: P REFINED ELECTROSTATICS

   :value: false
   :type:  true | false

   [Advanced] Compute electrostatic energy on a p refined mesh after each ground-state solve. Default: false.

.. data:: REPRODUCIBLE OUTPUT

   :value: false
   :type:  true | false

   [Developer] Limit output to what is reproducible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.

.. data:: VERBOSITY

   :value: 1
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 4`

   [Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users.

.. _Boundary-conditions:

Boundary conditions
-------------------

.. data:: Boundary conditions::CONSTRAINTS FROM SERIAL DOFHANDLER

   :value: true
   :type:  true | false

   [Developer] Check constraints from serial dofHandler.

.. data:: Boundary conditions::CONSTRAINTS PARALLEL CHECK

   :value: true
   :type:  true | false

   [Developer] Check for consistency of constraints in parallel.

.. data:: Boundary conditions::PERIODIC1

   :value: false
   :type:  true | false

   [Standard] Periodicity along the first domain bounding vector.

.. data:: Boundary conditions::PERIODIC2

   :value: false
   :type:  true | false

   [Standard] Periodicity along the second domain bounding vector.

.. data:: Boundary conditions::PERIODIC3

   :value: false
   :type:  true | false

   [Standard] Periodicity along the third domain bounding vector.

.. data:: Boundary conditions::SELF POTENTIAL RADIUS

   :value: 0.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 50`

   [Advanced] The radius (in a.u) of the ball around an atom in which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accommodate the largest radius possible for the given finite element mesh. The default approach works for most problems.

Brillouin zone k point sampling options
---------------------------------------

.. data:: Brillouin zone k point sampling options::USE GROUP SYMMETRY

   :value: false
   :type:  true | false

   [Standard] Flag to control the use of point group symmetries. Currently this feature cannot be used if ION FORCE or CELL STRESS input parameters are set to true.

.. data:: Brillouin zone k point sampling options::USE TIME REVERSAL SYMMETRY

   :value: false
   :type:  true | false

   [Standard] Flag to control the use of time reversal symmetry.

.. data:: Brillouin zone k point sampling options::kPOINT RULE FILE

   :value: 
   :type:  Any string

   [Developer] File providing list of k points on which eigen values are to be computed from converged KS Hamiltonian. The first three columns specify the crystal coordinates of the k points. The fourth column provides weights of the corresponding points, which is currently not used. The eigen values are written on an output file bands.out

Brillouin zone k point sampling options/Monkhorst-Pack (MP) grid generation
---------------------------------------------------------------------------

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING POINTS 1

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 1000`

   [Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 1.

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING POINTS 2

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 1000`

   [Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 2.

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING POINTS 3

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 1000`

   [Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 3.

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING SHIFT 1

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 1`

   [Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 1.

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING SHIFT 2

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 1`

   [Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 2.

.. data:: Brillouin zone k point sampling options::Monkhorst-Pack (MP) grid generation::SAMPLING SHIFT 3

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 1`

   [Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 3.

Checkpointing and Restart
-------------------------

.. data:: Checkpointing and Restart::CHK TYPE

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2`

   [Standard] Checkpoint type, 0 (do not create any checkpoint), 1 (create checkpoint for geometry optimization restart if either ION OPT or CELL OPT is set to true. Currently, checkpointing and restart framework does not work if both ION OPT and CELL OPT are set to true simultaneously- the code will throw an error if attempted.), 2 (create checkpoint for scf restart. Currently, this option cannot be used if geometry optimization is being performed. The code will throw an error if this option is used in conjunction with geometry optimization.)

.. data:: Checkpointing and Restart::RESTART FROM CHK

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying if the current job reads from a checkpoint. The nature of the restart corresponds to the CHK TYPE parameter. Hence, the checkpoint being read must have been created using the CHK TYPE parameter before using this option. RESTART FROM CHK is always false for CHK TYPE 0.

DFT functional parameters
-------------------------

.. data:: DFT functional parameters::EXCHANGE CORRELATION TYPE

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 4`

   [Standard] Parameter specifying the type of exchange-correlation to be used: 1(LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), 2(LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), 3(LDA: Vosko, Wilk \& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), 4(GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]).

.. data:: DFT functional parameters::PSEUDOPOTENTIAL CALCULATION

   :value: true
   :type:  true | false

   [Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed. For all-electron DFT calculation set to false.

.. data:: DFT functional parameters::PSEUDOPOTENTIAL FILE NAMES LIST

   :value: 
   :type:  Any string

   [Standard] Pseudopotential file. This file contains the list of pseudopotential file names in UPF format corresponding to the atoms involved in the calculations. UPF version 2.0 or greater and norm-conserving pseudopotentials(ONCV and Troullier Martins) in UPF format are only accepted. File format (example for two atoms Mg(z=12), Al(z=13)): 12 filename1.upf(row1), 13 filename2.upf (row2). Important Note: ONCV pseudopotentials data base in UPF format can be downloaded from http://www.quantum-simulation.org/potentials/sg15\_oncv.  Troullier-Martins pseudopotentials in UPF format can be downloaded from http://www.quantum-espresso.org/pseudopotentials/fhi-pp-from-abinit-web-site.

.. data:: DFT functional parameters::PSEUDO TESTS FLAG

   :value: false
   :type:  true | false

   [Developer] Boolean parameter specifying the explicit path of pseudopotential upf format files used for ctests

.. data:: DFT functional parameters::SPIN POLARIZATION

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 1`

   [Standard] Spin polarization: 0 for no spin polarization and 1 for collinear spin polarization calculation. Default option is 0.

.. data:: DFT functional parameters::START MAGNETIZATION

   :value: 0.0
   :type:  A floating point number :math:`v` such that :math:`-0.5 \leq v \leq 0.5`

   [Standard] Starting magnetization to be used for spin-polarized DFT calculations (must be between -0.5 and +0.5). Corresponding magnetization per simulation domain will be (2 x START MAGNETIZATION x Number of electrons) a.u.

Finite element mesh parameters
------------------------------

.. data:: Finite element mesh parameters::MESH FILE

   :value: 
   :type:  Any string

   [Developer] External mesh file path. If nothing is given auto mesh generation is performed. The option is only for testing purposes.

.. data:: Finite element mesh parameters::POLYNOMIAL ORDER

   :value: 4
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 12`

   [Standard] The degree of the finite-element interpolating polynomial. Default value is 4. POLYNOMIAL ORDER= 4 or 5 is usually a good choice for most pseudopotential as well as all-electron problems.

Finite element mesh parameters/Auto mesh generation parameters
--------------------------------------------------------------

.. data:: Finite element mesh parameters::Auto mesh generation parameters::ATOM BALL RADIUS

   :value: 2.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 20`

   [Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AROUND ATOM. The default value of 2.0 is good enough for most cases. On rare cases, where the nonlocal pseudopotential projectors have a compact support beyond 2.0, a slightly larger ATOM BALL RADIUS between 2.0 to 2.5 may be required. Standard users do not need to tune this parameter. Units: a.u.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::AUTO USER MESH PARAMS

   :value: false
   :type:  true | false

   [Standard] Except MESH SIZE AROUND ATOM, all other user defined mesh parameters are heuristically set. Default: false.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::BASE MESH SIZE

   :value: 0.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 20`

   [Advanced] Mesh size of the base mesh on which refinement is performed. For the default value of 0.0, a heuristically determined base mesh size is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::ERROR ESTIMATE WAVEFUNCTIONS

   :value: 5
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Developer] Number of wavefunctions to be used for error estimation.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::INNER ATOM BALL RADIUS

   :value: 0.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 20`

   [Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AT ATOM. Standard users do not need to tune this parameter. Units: a.u.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::MESH ADAPTION

   :value: false
   :type:  true | false

   [Standard] Generates adaptive mesh based on a-posteriori mesh adaption strategy using single atom wavefunctions before computing the ground-state. Default: false.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::MESH SIZE AROUND ATOM

   :value: 0.8
   :type:  A floating point number :math:`v` such that :math:`0.0001 \leq v \leq 10`

   [Standard] Mesh size in a ball of radius ATOM BALL RADIUS around every atom. For pseudopotential calculations, a value between 0.5 to 1.0 is usually a good choice. For all-electron calculations, a value between 0.1 to 0.3 would be a good starting choice. In most cases, MESH SIZE AROUND ATOM is the only parameter to be tuned to achieve the desired accuracy in energy and forces with respect to the mesh refinement. Units: a.u.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::MESH SIZE AT ATOM

   :value: 0.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 10`

   [Advanced] Mesh size of the finite elements in the immediate vicinity of the atom. For the default value of 0.0, a heuristically determined MESH SIZE AT ATOM is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::NUM LEVELS

   :value: 10
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 30`

   [Developer] Number of times to be refined.

.. data:: Finite element mesh parameters::Auto mesh generation parameters::TOLERANCE FOR MESH ADAPTION

   :value: 1
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Developer] Tolerance criteria used for stopping the multi-level mesh adaption done apriori using single atom wavefunctions. This is used as Kinetic energy change between two successive iterations

.. data:: Finite element mesh parameters::Auto mesh generation parameters::TOP FRAC

   :value: 0.1
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Developer] Top fraction of elements to be refined.

Geometry
--------

.. data:: Geometry::ATOMIC COORDINATES FILE

   :value: 
   :type:  Any string

   [Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.

.. data:: Geometry::DOMAIN VECTORS FILE

   :value: 
   :type:  Any string

   [Standard] Domain vectors input file name. Domain vectors are the vectors bounding the three edges of the 3D parallelepiped computational domain. File format: v1x v1y v1z (row1), v2x v2y v2z (row2), v3x v3y v3z (row3). Units: a.u. CAUTION: please ensure that the domain vectors form a right-handed coordinate system i.e. dotProduct(crossProduct(v1,v2),v3)>0. Domain vectors are the typical lattice vectors in a fully periodic calculation.

.. data:: Geometry::NATOMS

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Standard] Total number of atoms. This parameter requires a mandatory non-zero input which is equal to the number of rows in the file passed to ATOMIC COORDINATES FILE.

.. data:: Geometry::NATOM TYPES

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Standard] Total number of atom types. This parameter requires a mandatory non-zero input which is equal to the number of unique atom types in the file passed to ATOMIC COORDINATES FILE.

Geometry/Optimization
---------------------

.. data:: Geometry::Optimization::CELL CONSTRAINT TYPE

   :value: 12
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 13`

   [Standard] Cell relaxation constraint type, 1 (isotropic shape-fixed volume optimization), 2 (volume-fixed shape optimization), 3 (relax along domain vector component v1x), 4 (relax along domain vector component v2x), 5 (relax along domain vector component v3x), 6 (relax along domain vector components v2x and v3x), 7 (relax along domain vector components v1x and v3x), 8 (relax along domain vector components v1x and v2x), 9 (volume optimization- relax along domain vector components v1x, v2x and v3x), 10 (2D - relax along x and y components), 11(2D- relax only x and y components with inplane area fixed), 12(relax all domain vector components), 13 automatically decides the constraints based on boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.

.. data:: Geometry::Optimization::CELL OPT

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying if cell needs to be relaxed to achieve zero stress

.. data:: Geometry::Optimization::CELL STRESS

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying if cell stress needs to be computed. Automatically set to true if CELL OPT is true.

.. data:: Geometry::Optimization::FORCE TOL

   :value: 1e-4
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Standard] Sets the tolerance on the maximum force (in a.u.) on an atom during atomic relaxation, when the atoms are considered to be relaxed.

.. data:: Geometry::Optimization::ION FORCE

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying if atomic forces are to be computed. Automatically set to true if ION OPT is true.

.. data:: Geometry::Optimization::ION OPT

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying if atomic forces are to be relaxed.

.. data:: Geometry::Optimization::ION RELAX FLAGS FILE

   :value: 
   :type:  Any string

   [Standard] File specifying the permission flags (1-free to move, 0-fixed) and external forces for the 3-coordinate directions and for all atoms. File format (example for two atoms with atom 1 fixed and atom 2 free and 0.01 Ha/Bohr force acting on atom 2): 0 0 0 0.0 0.0 0.0(row1), 1 1 1 0.0 0.0 0.01(row2). External forces are optional.

.. data:: Geometry::Optimization::NON SELF CONSISTENT FORCE

   :value: false
   :type:  true | false

   [Developer] Boolean parameter specifying whether to include the force contributions arising out of non self-consistency in the Kohn-Sham ground-state calculation. Currently non self-consistent force computation is still in experimental phase. The default option is false.

.. data:: Geometry::Optimization::REUSE WFC

   :value: false
   :type:  true | false

   [Standard] Reuse previous ground-state wavefunctions during geometry optimization. Default setting is false.

.. data:: Geometry::Optimization::STRESS TOL

   :value: 1e-6
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Standard] Sets the tolerance of the cell stress (in a.u.) during cell-relaxation.

Parallelization
---------------

.. data:: Parallelization::BAND PARAL OPT

   :value: true
   :type:  true | false

   [Standard] Uses a more optimal route for band parallelization but at the cost of extra wavefunctions memory.

.. data:: Parallelization::MPI ALLREDUCE BLOCK SIZE

   :value: 100.0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq \text{MAX\_DOUBLE}`

   [Advanced] Block message size in MB used to break a single MPI\_Allreduce call on wavefunction vectors data into multiple MPI\_Allreduce calls. This is useful on certain architectures which take advantage of High Bandwidth Memory to improve efficiency of MPI operations. This variable is relevant only if NPBAND>1. Default value is 100.0 MB.

.. data:: Parallelization::NPBAND

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 2147483647`

   [Standard] Number of groups of MPI tasks across which the work load of the bands is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPBAND must be less than or equal to NUMBER OF KOHN-SHAM WAVEFUNCTIONS.

.. data:: Parallelization::NPKPT

   :value: 1
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 2147483647`

   [Standard] Number of groups of MPI tasks across which the work load of the irreducible k-points is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPKPT must be less than or equal to the number of irreducible k-points.

Poisson problem parameters
--------------------------

.. data:: Poisson problem parameters::MAXIMUM ITERATIONS

   :value: 10000
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 20000`

   [Advanced] Maximum number of iterations to be allowed for Poisson problem convergence.

.. data:: Poisson problem parameters::TOLERANCE

   :value: 1e-10
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Advanced] Absolute tolerance on the residual as stopping criterion for Poisson problem convergence.

Postprocessing
--------------

.. data:: Postprocessing::WRITE DENSITY

   :value: false
   :type:  true | false

   [Standard] Writes DFT ground state electron-density solution fields (FEM mesh nodal values) to densityOutput.vtu file for visualization purposes. The electron-density solution field in densityOutput.vtu is named density. In case of spin-polarized calculation, two additional solution fields- density\_0 and density\_1 are also written where 0 and 1 denote the spin indices. In the case of geometry optimization, the electron-density corresponding to the last ground-state solve is written. Default: false.

.. data:: Postprocessing::WRITE DENSITY OF STATES

   :value: false
   :type:  true | false

   [Standard] Computes density of states using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'dosData.out' containing two columns with first column indicating the energy in eV and second column indicating the density of states

.. data:: Postprocessing::WRITE LOCAL DENSITY OF STATES

   :value: false
   :type:  true | false

   [Standard] Computes local density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'ldosData.out' containing NUMATOM+1 columns with first column indicating the energy in eV and all other NUMATOM columns indicating local density of states for each of the NUMATOM atoms.

.. data:: Postprocessing::WRITE WFC

   :value: false
   :type:  true | false

   [Standard] Writes DFT ground state wavefunction solution fields (FEM mesh nodal values) to wfcOutput.vtu file for visualization purposes. The wavefunction solution fields in wfcOutput.vtu are named wfc\_s\_k\_i in case of spin-polarized calculations and wfc\_k\_i otherwise, where s denotes the spin index (0 or 1), k denotes the k point index starting from 0, and i denotes the Kohn-Sham wavefunction index starting from 0. In the case of geometry optimization, the wavefunctions corresponding to the last ground-state solve are written.  Default: false.

SCF parameters
--------------

.. data:: SCF parameters::COMPUTE ENERGY EACH ITER

   :value: true
   :type:  true | false

   [Advanced] Boolean parameter specifying whether to compute the total energy at the end of every SCF. Setting it to false can lead to some computational time savings.

.. data:: SCF parameters::CONSTRAINT MAGNETIZATION

   :value: false
   :type:  true | false

   [Standard] Boolean parameter specifying whether to keep the starting magnetization fixed through the SCF iterations. Default is FALSE

.. data:: SCF parameters::HIGHER QUAD NLP

   :value: true
   :type:  true | false

   [Advanced] Boolean parameter specifying whether to use a higher order quadrature rule for the calculations involving the non-local part of the pseudopotential. Default setting is true. Could be safely set to false if you are using a very refined mesh.

.. data:: SCF parameters::MAXIMUM ITERATIONS

   :value: 100
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 1000`

   [Standard] Maximum number of iterations to be allowed for SCF convergence

.. data:: SCF parameters::MIXING HISTORY

   :value: 10
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 1000`

   [Standard] Number of SCF iteration history to be considered for density mixing schemes. For metallic systems, a mixing history larger than the default value provides better scf convergence.

.. data:: SCF parameters::MIXING METHOD

   :value: ANDERSON
   :type:  Any one of BROYDEN, ANDERSON

   [Standard] Method for density mixing. ANDERSON is the default option.

.. data:: SCF parameters::MIXING PARAMETER

   :value: 0.1
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Standard] Mixing parameter to be used in density mixing schemes. Default: 0.1.

.. data:: SCF parameters::STARTING WFC

   :value: RANDOM
   :type:  Any one of ATOMIC, RANDOM

   [Standard] Sets the type of the starting Kohn-Sham wavefunctions guess: Atomic(Superposition of single atom atomic orbitals. Atom types for which atomic orbitals are not available, random wavefunctions are taken. Currently, atomic orbitals data is not available for all atoms.), Random(The starting guess for all wavefunctions are taken to be random). Default: RANDOM.

.. data:: SCF parameters::TEMPERATURE

   :value: 500.0
   :type:  A floating point number :math:`v` such that :math:`1e-05 \leq v \leq \text{MAX\_DOUBLE}`

   [Standard] Fermi-Dirac smearing temperature (in Kelvin).

.. data:: SCF parameters::TOLERANCE

   :value: 1e-06
   :type:  A floating point number :math:`v` such that :math:`1e-12 \leq v \leq 1`

   [Standard] SCF iterations stopping tolerance in terms of $L_2$ norm of the electron-density difference between two successive iterations. CAUTION: A tolerance close to 1e-7 or lower can deteriorate the SCF convergence due to the round-off error accumulation.

SCF parameters/Eigen-solver parameters
--------------------------------------

.. data:: SCF parameters::Eigen-solver parameters::ADAPTIVE FILTER STATES

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Advanced] Number of lowest Kohn-Sham eigenstates which are filtered with Chebyshev polynomial degree linearly varying from 50 percent (starting from the lowest) to 80 percent of the value specified by CHEBYSHEV POLYNOMIAL DEGREE. This imposes a step function filtering polynomial order on the ADAPTIVE FILTER STATES as filtering is done with blocks of size WFC BLOCK SIZE. This setting is recommended for large systems (greater than 5000 electrons). Default value is 0 i.e., all states are filtered with the same Chebyshev polynomial degree.

.. data:: SCF parameters::Eigen-solver parameters::ALGO

   :value: NORMAL
   :type:  Any one of NORMAL, FAST

   [Standard] In the FAST mode, spectrum splitting technique is used in Rayleigh-Ritz step, and mixed precision arithmetic algorithms are used in Rayleigh-Ritz and Cholesky factorization based orthogonalization step. For spectrum splitting, 85 percent of the total number of wavefunctions are taken to be core states, which holds good for most systems including metallic systems assuming NUMBER OF KOHN-SHAM WAVEFUNCTIONS to be around 10 percent more than N/2. FAST setting is strongly recommended for large-scale (> 10k electrons) system sizes. Both NORMAL and FAST setting use Chebyshev filtered subspace iteration technique. Currently, FAST setting is only enabled for the real executable and with ScaLAPACK linking. If manual options for mixed precision and spectum splitting are being used, please use NORMAL setting for ALGO. Default setting is NORMAL.

.. data:: SCF parameters::Eigen-solver parameters::BATCH GEMM

   :value: true
   :type:  true | false

   [Advanced] Boolean parameter specifying whether to use gemm batch blas routines to perform matrix-matrix multiplication operations with groups of matrices, processing a number of groups at once using threads instead of the standard serial route. CAUTION: gemm batch blas routines will only be activated if the CHEBY WFC BLOCK SIZE is less than 1000, and only if intel mkl blas library is linked with the dealii installation. Default option is true.

.. data:: SCF parameters::Eigen-solver parameters::CHEBYSHEV FILTER TOLERANCE

   :value: 2e-02
   :type:  A floating point number :math:`v` such that :math:`1e-10 \leq v \leq \text{MAX\_DOUBLE}`

   [Advanced] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure. Default value is sufficient for most purposes

.. data:: SCF parameters::Eigen-solver parameters::CHEBYSHEV POLYNOMIAL DEGREE

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2000`

   [Advanced] Chebyshev polynomial degree to be employed for the Chebyshev filtering subspace iteration procedure to dampen the unwanted spectrum of the Kohn-Sham Hamiltonian. If set to 0, a default value depending on the upper bound of the eigen-spectrum is used. See Phani Motamarri et.al., J. Comp. Phys. 253, 308-343 (2013).

.. data:: SCF parameters::Eigen-solver parameters::CHEBY WFC BLOCK SIZE

   :value: 400
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 2147483647`

   [Advanced] Chebyshev filtering procedure involves the matrix-matrix multiplication where one matrix corresponds to the discretized Hamiltonian and the other matrix corresponds to the wavefunction matrix. The matrix-matrix multiplication is accomplished in a loop over the number of blocks of the wavefunction matrix to reduce the memory footprint of the code. This parameter specifies the block size of the wavefunction matrix to be used in the matrix-matrix multiplication. The optimum value is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting CHEBY WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/CHEBY WFC BLOCK SIZE equals an integer value. Default value is 400.

.. data:: SCF parameters::Eigen-solver parameters::ENABLE SUBSPACE ROT PGS OPT

   :value: true
   :type:  true | false

   [Developer] Turns on subspace rotation optimization for Pseudo-Gram-Schimdt orthogonalization. Default option is true.

.. data:: SCF parameters::Eigen-solver parameters::ENABLE SWITCH TO GS

   :value: true
   :type:  true | false

   [Developer] Controls automatic switching to Gram-Schimdt orthogonalization if Lowden Orthogonalization or Pseudo-Gram-Schimdt orthogonalization are unstable. Default option is true.

.. data:: SCF parameters::Eigen-solver parameters::LOWER BOUND UNWANTED FRAC UPPER

   :value: 0
   :type:  A floating point number :math:`v` such that :math:`0 \leq v \leq 1`

   [Developer] The value of the fraction of the upper bound of the unwanted spectrum, the lower bound of the unwanted spectrum will be set. Default value is 0.

.. data:: SCF parameters::Eigen-solver parameters::LOWER BOUND WANTED SPECTRUM

   :value: -10.0
   :type:  A floating point number :math:`v` such that :math:`-\text{MAX\_DOUBLE} \leq v \leq \text{MAX\_DOUBLE}`

   [Developer] The lower bound of the wanted eigen spectrum. It is only used for the first iteration of the Chebyshev filtered subspace iteration procedure. A rough estimate based on single atom eigen values can be used here. Default value is good enough for most problems.

.. data:: SCF parameters::Eigen-solver parameters::NUMBER OF KOHN-SHAM WAVEFUNCTIONS

   :value: 10
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Standard] Number of Kohn-Sham wavefunctions to be computed. For spin-polarized calculations, this parameter denotes the number of Kohn-Sham wavefunctions to be computed for each spin. A recommended value for this parameter is to set it to N/2+Nb where N is the number of electrons. Use Nb to be 5-10 percent of N/2 for insulators and for metals use Nb to be 10-15 percent of N/2. If 5-15 percent of N/2 is less than 10 wavefunctions, set Nb to be atleast 10.

.. data:: SCF parameters::Eigen-solver parameters::ORTHOGONALIZATION TYPE

   :value: Auto
   :type:  Any one of GS, LW, PGS, Auto

   [Advanced] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization using SLEPc library), LW(Lowden Orthogonalization implemented using LAPACK/BLAS routines, extension to use ScaLAPACK library not implemented yet), PGS(Pseudo-Gram-Schmidt Orthogonalization: if dealii library is compiled with ScaLAPACK and if you are using the real executable, parallel ScaLAPACK functions are used, otherwise serial LAPACK functions are used.) Auto is the default option, which chooses GS for all-electron case and PGS for pseudopotential case. GS and LW options are only available if RR GEP is set to false.

.. data:: SCF parameters::Eigen-solver parameters::RR GEP

   :value: true
   :type:  true | false

   [Advanced] Solve generalized eigenvalue problem instead of standard eignevalue problem in Rayleigh-Ritz step. This approach is not extended yet to complex executable. Default value is true for real executable and false for complex executable.

.. data:: SCF parameters::Eigen-solver parameters::SCALAPACKPROCS

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 300`

   [Advanced] Uses a processor grid of SCALAPACKPROCS times SCALAPACKPROCS for parallel distribution of the subspace projected matrix in the Rayleigh-Ritz step and the overlap matrix in the Pseudo-Gram-Schmidt step. Default value is 0 for which a thumb rule is used (see http://netlib.org/scalapack/slug/node106.html). If ELPA is used, twice the value obtained from the thumb rule is used as ELPA scales much better than ScaLAPACK. This parameter is only used if dealii library is compiled with ScaLAPACK.

.. data:: SCF parameters::Eigen-solver parameters::SCALAPACK BLOCK SIZE

   :value: 50
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 300`

   [Advanced] ScaLAPACK process grid block size.

.. data:: SCF parameters::Eigen-solver parameters::SPECTRUM SPLIT CORE EIGENSTATES

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Advanced] Number of lowest Kohn-Sham eigenstates which should not be included in the Rayleigh-Ritz diagonalization.  In other words, only the eigenvalues and eigenvectors corresponding to the higher eigenstates (Number of Kohn-Sham wavefunctions minus the specified core eigenstates) are computed in the diagonalization of the projected Hamiltonian. This value is usually chosen to be the sum of the number of core eigenstates for each atom type multiplied by number of atoms of that type. This setting is recommended for large systems (greater than 5000 electrons). Default value is 0 i.e., no core eigenstates are excluded from the Rayleigh-Ritz projection step. Currently this optimization is not implemented for the complex executable and ScaLAPACK linking is also needed.

.. data:: SCF parameters::Eigen-solver parameters::SPECTRUM SPLIT STARTING SCF ITER

   :value: 0
   :type:  An integer :math:`n` such that :math:`0\leq n \leq 2147483647`

   [Advanced] SCF iteration no beyond which spectrum splitting based can be used.

.. data:: SCF parameters::Eigen-solver parameters::SUBSPACE ROT DOFS BLOCK SIZE

   :value: 5000
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 2147483647`

   [Developer] This block size is used for memory optimization purposes in subspace rotation step in Pseudo-Gram-Schmidt orthogonalization and Rayleigh-Ritz steps. This optimization is only activated if dealii library is compiled with ScaLAPACK. Default value is 5000.

.. data:: SCF parameters::Eigen-solver parameters::USE ELPA

   :value: false
   :type:  true | false

   [Standard] Use ELPA instead of ScaLAPACK for diagonalization of subspace projected Hamiltonian and Pseudo-Gram-Schmidt orthogonalization. Currently this setting is only available for real executable. Default setting is false.

.. data:: SCF parameters::Eigen-solver parameters::USE MIXED PREC PGS O

   :value: false
   :type:  true | false

   [Advanced] Use mixed precision arithmetic in overlap matrix computation step of PGS orthogonalization, if ORTHOGONALIZATION TYPE is set to PGS. Currently this optimization is only enabled for the real executable and with ScaLAPACK linking. Default setting is false.

.. data:: SCF parameters::Eigen-solver parameters::USE MIXED PREC PGS SR

   :value: false
   :type:  true | false

   [Advanced] Use mixed precision arithmetic in subspace rotation step of PGS orthogonalization, if ORTHOGONALIZATION TYPE is set to PGS. Currently this optimization is only enabled for the real executable and with ScaLAPACK linking. Default setting is false.

.. data:: SCF parameters::Eigen-solver parameters::USE MIXED PREC RR\_SR SPECTRUM SPLIT

   :value: false
   :type:  true | false

   [Advanced] Use mixed precision arithmetic in Rayleigh-Ritz subspace rotation step when SPECTRUM SPLIT CORE EIGENSTATES>0. Currently this optimization is only enabled for the real executable and with ScaLAPACK linking. Default setting is false.

.. data:: SCF parameters::Eigen-solver parameters::USE MIXED PREC XTHX SPECTRUM SPLIT

   :value: false
   :type:  true | false

   [Advanced] Use mixed precision arithmetic in computing subspace projected Kohn-Sham Hamiltonian when SPECTRUM SPLIT CORE EIGENSTATES>0. Currently this optimization is only enabled for the real executable and with ScaLAPACK linking. Default setting is false.

.. data:: SCF parameters::Eigen-solver parameters::WFC BLOCK SIZE

   :value: 400
   :type:  An integer :math:`n` such that :math:`1\leq n \leq 2147483647`

   [Advanced]  This parameter specifies the block size of the wavefunction matrix to be used for memory optimization purposes in the orthogonalization, Rayleigh-Ritz, and density computation steps. The feature is activated only if dealii library is compiled with ScaLAPACK. The optimum block size is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/WFC BLOCK SIZE equals an integer value. Default value is 400.

