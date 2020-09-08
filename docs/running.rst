Running DFT-FE
==============

After compiling `DFT-FE` as described in :ref:`installation`, we have now two executables:

  * `build/release/real/dftfe`, which uses real data-structures is sufficient for fully non-periodic problems.
    This executable can also be used for periodic and semi-periodic problems involving a Gamma point calculation.

  * `build/release/complex/dftfe`, which uses complex data-structures is required for periodic and
    semi-periodic problems with multiple k point sampling for Brillouin zone integration.

These executables are to be used as follows.
For a serial run use::

  ./dftfe parameterFile.prm

or, for a parallel run use::

  mpirun -n N ./dftfe parameterFile.prm

to run with N processors. 


Input File Syntax
-----------------

In the above, an input file with `.prm` extension is used.
This file contains input parameters as described in :ref:`parameters`.
Paramter files contain both *Global Parameters* and 
*Nested Parameters*.
For example::

  TOLERANCE = 1e-6

  subsection A
    set PARAMETER SUBSECTION xyzA1= value1
    set PARAMETER SUBSECTION xyzA2=value2
    subsection B
      set PARAMETER SUBSUBSECTION xyzAB1   =value1
      set PARAMETER SUBSUBSECTION xyzAB2 = value2
    end
  end

The parameter files are not sensitive to indentation or whitespace around the "=" sign.

