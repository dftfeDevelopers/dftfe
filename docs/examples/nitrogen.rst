Nitrogen Gas
------------

Now we will walk you through a few demo examples in the `/dftfe/demo` folder.
The demo examples do not cover all the input parameter options. To get full list of input parameters see :ref:`parameters`.

In the folder `/dftfe/demo/ex1`, we compute the ground state of the
nitrogen molecule using pseudopotential DFT with a real-space mesh
under fully non-periodic boundary conditions.

There are two input parameter files:

  * `parameterFile_a.prm` computes the ground-state and forces of the nitrogen molecule

  *  `parameterFile_b.prm` additionally does atomic relaxation

Below, we provide a step by step procedure on setting up the above input parameter files,
doing a total energy and force convergence study with respect to finite-element mesh discretization,
and finally doing the atomic relaxation of nitrogen molecule.

1. The geometry of the nitrogen molecule (:math:`{\rm N}_{2}`) system is set using input parameters under `Geometry` subsection::

    subsection Geometry
      set NATOMS = 2
      set NATOM TYPES = 1
      set ATOMIC COORDINATES FILE = coordinates.inp 
      set DOMAIN VECTORS FILE = domainVectors.inp
    end

  where

  * `NATOMS` is the total number of atoms, and `NATOM TYPES` is the total number of atom types.
	
  * ``domainVectors.inp`` is the input file we have included with the example to list the three domain vectors (in Bohr)
    describing the 3D parallelepiped computational domain.

  For the current example we take a cubodial domain with 80 a.u as the edge length. 
  Accordingly, the ``domainVectors.inp`` file is formatted as::

    80.0 0.0 0.0
    0.0 80.0 0.0
    0.0 0.0 80.0

  Each row corresponds to a domain vector.
  It is a requirement that the above vectors must form a right-handed coordinate system i.e. :math:`(v_1 \times v_2)\cdot v_3 >0`.

  * ``coordinates.inp`` is the input file we have included with the example to list the
    Cartesian coordinates of the atoms (in a.u.) with respect to origin at the center of the domain.
    For this example, ``coordinates.inp`` is described as::

        7    5   -1.30000000E+00   0.00000000E+00   0.00000000E+00
        7    5    1.30000000E+00   0.00000000E+00   0.00000000E+00

  Each line corresponds to "<atomic-number> <valence-charge> <x> <y> <z>".
  Since this is a pseudopotential calculation, the valence-charge must correspond
  to the pseudopotential input, which we disuss in the later steps.

  For an all-electron calculation, the valence charge would be equal to the atomic number.
  Here, the pseudopotential describes 2 core electrons, leaving DFT to handle the remaining 5.

  .. note::
    We require Cartesian coordinates (in Bohr) for fully non-periodic simulation domain like above.
    Fractional coordinates are mandatory for periodic and semi-periodic simulation domains.

2. Set the fully non-periodic boundary conditions for the problem using the subsection :ref:`Boundary-conditions`::

    subsection Boundary conditions
      set PERIODIC1                       = false
      set PERIODIC2                       = false
      set PERIODIC3                       = false
    end

  where `PERIODIC1/2/3` sets the periodicity along the first, second, and third domain vectors.
  We note that `DFT-FE` allows for arbitrary boundary conditions.

3.  Set the required DFT functional input parameters for pseudopotential calculation::

        subsection DFT functional parameters
          set EXCHANGE CORRELATION TYPE   = 4
          set PSEUDOPOTENTIAL CALCULATION = true
          set PSEUDOPOTENTIAL FILE NAMES LIST = pseudo.inp
        end

    where

    * The choice of "4" for `EXCHANGE CORRELATION TYPE` corresponds to "GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]" functional. 
		
    * ``pseudo.inp`` is the input file we have included with the example to list out all pseudopotential file names
      (in `UPF` format) corresponding to the atom types involved in the calculations. The file is formatted as::

        7 N_ONCV_PBE-1.0.upf

    where "7" is the atomic number of nitrogen, and `N_ONCV_PBE-1.0.upf` is the
    Optimized Norm-Conserving Vanderbilt pseudopotential (ONCV) file obtained
    from http://www.quantum-simulation.org/potentials/sg15_oncv/upf/.

    .. note::
        Presently, we only support Norm-Conserving pseudopotential (Troullier-Martins, ONCV) files in `UPF`
        format (version 2.0 or greater).

4. Set the input parameters for self-consistent field (SCF) iterative procedure.::

    subsection SCF parameters
      set MIXING HISTORY   = 70
      set MIXING PARAMETER = 0.5
      set MAXIMUM ITERATIONS               = 40
      set TEMPERATURE                      = 500
      set TOLERANCE                        = 5e-5
      subsection Eigen-solver parameters
          set NUMBER OF KOHN-SHAM WAVEFUNCTIONS = 12
      end
    end	

  where

    * "70" set for `MIXING HISTORY` is the number of SCF iteration history to be considered for mixing the electron-density.
    * "0.5" set for `MIXING PARAMETER` is the mixing parameter to be used in the mixing scheme.
    * "40" set for `MAXIMUM ITERATIONS` is the maximum number of iterations allowed in SCF iterative procedure.
    * "500" set for `TEMPERATURE` is the Fermi-Dirac smearing temperature in Kelvin.
    * "1e-5" set for `TOLERANCE` is the SCF stopping tolerance in terms of L2 norm of the electron-density
       difference between two successive iterations.
    * "12" set for `NUMBER OF KOHN-SHAM WAVEFUNCTIONS` is the number of Kohn-Sham wavefunctions to be computed
      in the eigen-solve (using Chebyshev subspace iteration solve) for every SCF iteration step.
      This parameter is set inside the subsection `Eigen-solver parameters`, which is nested within `SCF parameters`.

5. As we are also computing the force on the atoms in this example, update the `Geometry` subsection in the first step to::

    subsection Geometry
      set NATOMS = 2
      set NATOM TYPES = 1
      set ATOMIC COORDINATES FILE = coordinates.inp 
      set DOMAIN VECTORS FILE = domainVectors.inp
      subsection Optimization
        set ION FORCE = true
      end
    end

  where the `ION FORCE` is set to true inside the nested subsection `Optimization`.
  This computes and prints the forces on the atoms at the end of the ground-state solve.

6. `DFT-FE` employs finite-element basis. These basis are piecewise polynomial functions.
   Hence `DFT-FE` allows for two approaches (**H** and **P** refinement) for systematic converge
   of the ground-state energy and forces. **H** refinement is done primarily by tuning the input parameter
   `MESH SIZE AROUND ATOM`, which controls the finite-element mesh size (grid spacing) around all atoms.
   **P** refinement is controlled by `POLYNOMIAL ORDER`, which is the degree of polynomial associated with
   the finite-element basis.

   In this example, we will first tune
   `MESH SIZE AROUND ATOM` for convergence while keeping `POLYNOMIAL ORDER` to 4.
   Then, while keeping the `MESH SIZE AROUND ATOM` constant, we will increase the `POLYNOMIAL ORDER` to 5.
   `POLYNOMIAL ORDER` 4 or 5 is a good choice for most pseudopotential as well as
   all-electron problems. Let us take the following input parameters to start with::

    subsection Finite element mesh parameters
      set POLYNOMIAL ORDER = 4
      subsection Auto mesh generation parameters
        set AUTO USER MESH PARAMS = true
        set MESH SIZE AROUND ATOM  = 0.8
      end
    end

  and now run the problem using the `/build/release/real/dftfe` executable::

   mpirun -n 32 ../../build/release/real/dftfe parameterFile_a.prm > outputMesh1 &

  From the ``outputMesh1`` file, you can obtain information on the number of degrees of
  freedom in the auto-generated finite-element mesh and the ground-state energy and forces. 

  Repeat the above step thrice, once with `MESH SIZE AROUND ATOM  = 0.6`, then with
  `MESH SIZE AROUND ATOM  = 0.5`, and finally with `MESH SIZE AROUND ATOM  = 0.4`.
  Now run one more time with `POLYNOMIAL ORDER  = 5` while keeping `MESH SIZE AROUND ATOM  = 0.4`.
  We recommend to run this final simulation with around 64 MPI tasks for faster computational times.

  The ground-state energy per atom and force on the atomId 0 is tabulated :ref:`below <table1>` for all the above cases.
  Upon comparing the errors in the energy and force with respect the most refined mesh (*Mesh No. 5*), we observe that for *Mesh No. 3* we obtain convergence in energy per atom to :math:`\mathcal{O}(10^{-5})` accuracy, and convergence in force to :math:`\mathcal{O}(10^{-5})` accuracy. For your reference, we have provided the output file for *Mesh No. 3* at `/demo/ex1/ex1_a.output`. `DFT-FE` also has the capability to write finite-element meshes with electron-density or wavefunction information to .vtu format which can be visualized using software like ParaView or Visit. As an example, Figure~\ref{fig:N2} shows the finite-element mesh and electron-density contours for *Mesh No. 3*, which are obtained via setting::

    subsection Postprocessing
       set WRITE DENSITY=true
    end

  in the input parameter file, and visualizing the ``densityOutput.vtu`` file in ParaView.

  .. _table1:

    Nitrogen molecule ground-state energy and force convergence for demo example 1

    ========  =============  ===========   ========================  ===============   =================
    Mesh No.   POLYNOMIAL    MESH SIZE     Total degrees of freedom  Energy per atom   Force on atomId 0
               ORDER         AROUND ATOM   per atom                  (Hartree)         (Hartree/Bohr)
    ========  =============  ===========   ========================  ===============   =================
    1             4           0.8          50,665                    -9.89788996         0.2936143
    2             4           0.6          70,997                    -9.89905556         0.2944262
    3             4           0.5          180,865                   -9.89919209         0.2941677
    4             4           0.4          201,969                   -9.89920596         0.2941331
    5             5           0.4          385,577                   -9.89921195         0.2941327
    ========  =============  ===========   ========================  ===============   =================

7. Finally we discuss how to set up the input parameter file for atomic relaxation. 
   Use the same finite-element mesh input parameters as used for *Mesh No. 3*,
   and update the input parameters in `Geometry` subsection from the fifth step to::

    subsection Geometry
      set NATOMS = 2
      set NATOM TYPES = 1
      set ATOMIC COORDINATES FILE = coordinates.inp 
      set DOMAIN VECTORS FILE = domainVectors.inp
      subsection Optimization
        set ION OPT              = true
        set FORCE TOL            = 1e-4    
        set ION RELAX FLAGS FILE = relaxationFlags.inp
      end
    end

  where

    * `ION OPT` is set to true to enable atomic relaxation.  		
    * "1e-4" for `FORCE TOL` sets the tolerance of the maximum force (in Hartree/Bohr) on an atom when atoms are
      considered to be relaxed.
    * ``relaxationFlags.inp``, is the input file we have included in the example to specify
      the permission flags (1-- free to move, 0-- fixed) for each coordinate axis and for all atoms.
      This file contains one line per atom::

        1 0 0
        1 0 0

  This marks both nitrogen atoms to move freely along the x axis, while remaining at its starting values for y and z.

  Now, run the atomic relaxation problem. From the output file, you should observe that the Nitrogen molecule geometry relaxed to
  an equilibrium bond length of 2.0819 Bohr after 10 geometry update steps.
  For your reference, we have provided an output file at `/demo/ex1/ex1_b.output`, which was run using 32 MPI tasks.

.. _figN2:

  Finite-element mesh used in Nitrogen molecule pseudopotential DFT calculation (See \ref{sec:example1}).}

.. image:: /_static/zoomedOutDemo1.png
    :alt: Zoomed-out electron density of nitrogen.

.. image:: /_static/zoomedInDemo1.png
    :alt: Zoomed-in with electron-density contours.

