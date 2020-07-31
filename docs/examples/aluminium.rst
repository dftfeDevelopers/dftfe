FCC Aluminium
=============

In the previous example, we discussed how to setup and run a fully non-periodic problem.
Here we briefly discuss how to setup and run the fully periodic problem (FCC Aluminium unit cell)
in the folder `/dftfe/demo/ex2`. There are again two input parameter files:

  * ``parameterFile_a.prm`` computes the ground-state and cell stress of the FCC Al unit cell
  * ``parameterFile_b.prm`` additionally does cell stress relaxation

For this example, the `coordinates.inp`` (given as input to `ATOMIC COORDINATES FILE`)
lists the fractional (reduced) coordinates of the atoms::

    13   3   0.00000000E+00   0.00000000E+00   0.00000000E+00
    13   3   0.00000000E+00   0.50000000E+00   0.50000000E+00
    13   3   0.50000000E+00   0.00000000E+00   0.50000000E+00
    13   3   0.50000000E+00   0.50000000E+00   0.00000000E+00

where each line corresponds to "<atomic-number> <valence-charge> <fracx> <fracy> <fracz>".

.. note::

  We require fractional coordinates for fully periodic or semi-periodic simulation
  domains while Cartesian coordinates are mandatory for fully non-periodic simulation domain.


There are several new steps needed to describe periodic simulation domains.

1. The input parameter file should specify periodic boundary conditions::

    subsection Boundary conditions
      set PERIODIC1                       = true
      set PERIODIC2                       = true
      set PERIODIC3                       = true
    end

2. We also want to compute cell stress, so (nested
in the `Geometry` / `Optimization` subsections), set::

    set CELL STRESS = true

3. The wavefunctions gain a :math:`\exp(i k\cdot r)` factor due to Bloch's theorem,
   where :math:`k = 2\pi L^{-1} m` is an electron momentum, :math:`m` is
   a triple of integers, and :math:`L` is the 3x3 matrix whose rows are lattice vectors.
   We choose the set of :math:`m` to calculate by specifying a "k-point" grid::

    subsection Brillouin zone k point sampling options
      set USE TIME REVERSAL SYMMETRY = true
      subsection Monkhorst-Pack (MP) grid generation
        set SAMPLING POINTS 1 = 2
        set SAMPLING POINTS 2 = 2
        set SAMPLING POINTS 3 = 2
        set SAMPLING SHIFT 1  = 1
        set SAMPLING SHIFT 2  = 1
        set SAMPLING SHIFT 3  = 1
      end
    end

  where

  * `SAMPLING POINTS 1/2/3` sets the number of Monkhorst-Pack grid points to be used along reciprocal lattice
    vectors 1, 2, and 3.  		

  * Setting `SAMPLING SHIFT 1/2/3` to 1 enables fractional shifting to be used along reciprocal lattice vectors.

  * Setting `USE TIME REVERSAL SYMMETRY` to true enables use of time reversal symmetry to reduce number
    of k points to be solved for. For this option to work, `SAMPLING SHIFT 1/2/3` must be set to 1 as done above. 


4. We may as well calculate wavefunctions at each *k*-point in parallel.  To do this, set::

    subsection Parallelization
      set NPKPT = 2
    end

  which parallelizes the work load of the irreducible k-points across two groups of MPI tasks.

5. The same strategy for convergence of the ground state energy and force discussed
   in the previous example is applied to the current example to get convergence in ground state energy and cell stress. 

   The ground-state energy per atom and hydrostatic cell stress for finite-element meshes with increasing
   level of refinement is tabulated :ref:`below <table2>`.
   Upon comparing the errors in the energy and force with respect the most refined mesh (*Mesh No. 3*),
   we observe that for *Mesh No. 2* we have obtained convergence in energy per atom to
   :math:`\mathcal{O}(10^{-6})` accuracy, and convergence in cell stress to
   :math:`\mathcal{O}(10^{-7})` accuracy.

  .. _table2: FCC Al ground-state energy and hydrostatic cell-stress convergence for demo example 2

    ========  ========== ===========  ========================  ===============  ==============================
    Mesh No.  POLYNOMIAL MESH SIZE    Total degrees of freedom  Energy per atom  Hydrostatic Cell stress
              ORDER      AROUND ATOM  per atom                  (Hartree)        (Hartree/:math:`{\rm Bohr}^3`)
    ========  ========== ===========  ========================  ===============  ==============================
    1         4          0.8          24,169                    -2.09083548      0.000028917
    2         4          0.5          108,689                   -2.09322424      0.000037932
    3         5          0.4          204,957                   -2.09322860      0.000038135
    ========  ========== ===========  ========================  ===============  ==============================

The output file using the mesh parameters for *Mesh No.2* is provided at `/demo/ex2/ex2_a.output` (for `parameterFile_a.prm`).

8. For cell stress relaxation, use ``parameterFile_b.prm``, where we set within
   the `Optimization` subsection nested under `Geometry`::

    set STRESS TOL            = 4e-6
    set CELL OPT              = true
    set CELL CONSTRAINT TYPE  = 1

  where

  * `CELL OPT` is set to true which enables cell stress relaxation.
  * "4e-6" for `STRESS TOL` sets the tolerance of the cell stress (in a.u.) for cell stress relaxation.
  * Choice of "1" for `CELL CONSTRAINT TYPE` enforces isotropic shape-fixed
    volume optimization constraint during cell stress relaxation.

For your reference, the output file for the cell stress relaxation is provided at
`/demo/ex2/ex2_b.output`. From the output file, you should observe that you obtain a
relaxed lattice constant of 7.563 Bohr after two geometry updates. 


