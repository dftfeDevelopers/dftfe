.. _installation:

Installing DFT-FE
=================

All the underlying installation instructions assume a Linux operating system. We assume standard tools and libraries like CMake, compilers- (C, C++ and Fortran), and MPI libraries are pre-installed. Most high-performance computers would have the latest version of these libraries in the default environment. However, in many cases you would have to use `Environment Modules <http://modules.sourceforge.net/}{Environment Modules>`_ to set the correct environment variables for compilers-(C, C++ and Fortran), MPI libraries, and compilation tools like `CMake <http://www.cmake.org/>`_.
*For example, on one of the high-performance computers we develop and test the `DFT-FE` code, we can use the following commands to set the desired environment variables*::

    $ module load cmake
    $ module load intel
    $ module load mpilibrary

In the above mpilibrary denotes the MPI library you are using in your system(for eg: openmpi, mpich or intel-mpi). 
We strongly recommend using the latest stable version of compilers-(C, C++ and Fortran), and MPI libraries available on your high-performance computer. **Our experience shows that Intel compilers provide the best performance in comparison to GNU compilers**. Furthermore, for the installations which use `CMake <http://www.cmake.org/>`_, version 2.8.12 or later is required.   

Compiling and installing external libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`DFT-FE` is primarily based on the open source finite element library `deal.II <http://www.dealii.org/>`_,
through which external dependencies
on `p4est <http://p4est.org/>`_, `PETSc <https://www.mcs.anl.gov/petsc/>`_, `SLEPc <http://slepc.upv.es/>`_, and `ScaLAPACK <http://www.netlib.org/scalapack/>`_ are set. ScaLAPACK is an optional requirement, but strongly recommended for large problem sizes with 5000 electrons or more. The other required external libraries, which are
not interfaced via deal.II are `ALGLIB <http://www.alglib.net/>`_, `Libxc <http://www.tddft.org/programs/libxc/>`_, `spglib <https://atztogo.github.io/spglib/>`_, `Libxml2 <http://www.xmlsoft.org/>`_ and optionally `ELPA <https://elpa.mpcdf.mpg.de/>`_.  ELPA is strongly recommended for large problem sizes with 10000 electrons or more. Some of the above libraries (PETSc, SLEPc, ScaLAPACK, and Libxml2) might already be installed on most high-performance computers.

Below, we give brief installation and/or linking instructions for each of the above libraries.

Instructions for ALGLIB, Libxc, spglib, Libxml2, and ELPA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   * **ALGLIB**: Used by `DFT-FE` for spline fitting for various radial data. Download the current release of the Alglib free C++ edition from \url{http://www.alglib.net/download.php}. After downloading and unpacking, go to `cpp/src`, and create a shared library using a C++ compiler. For example, using Intel compiler do::

      $ icpc -c -fPIC -O3 *.cpp
      $ icpc *.o -shared -o libAlglib.so

 * **Libxc**: Used by `DFT-FE` for exchange-correlation functionals. Download the current release from \url{http://www.tddft.org/programs/libxc/download/}, and do::

      $ ./configure --prefix=libxc_install_dir_path
                    CC=c_compiler CXX=c++_compiler FC=fortran_compiler
                    CFLAGS=-fPIC FCFLAGS=-fPIC
      
      $ make
      $ make install

  Do not forget to replace `libxc_install_dir_path` by some appropriate path on your file system and make sure that you have write permissions. Also replace `c_compiler, c++_compiler` and `fortran_compiler` with compilers on your system.

 * **spglib**: Used by `DFT-FE` to find crystal symmetries. To install spglib, first obtain the development version of spglib from their github repository by::

      $ git clone https://github.com/atztogo/spglib.git	

  and next follow the ``Compiling using cmake`` installation procedure described in \url{https://atztogo.github.io/spglib/install.html}.   	

 * **Libxml2**: Libxml2 is used by `DFT-FE` to read `.xml` files. Most likely, Libxml2 might be already installed in the high-performance computer you are working with. It is usually installed in the default locations like `/usr/lib64` (library path which contains `.so` file for Libxml2, something like `libxml2.so.2`) and `/usr/include/libxml2` (include path). 

   Libxml2 can also be installed by doing (Do not use these instructions if you have already have Libxml2 on your system)::

      $ git clone git://git.gnome.org/libxml2
      $ ./autogen.sh --prefix=Libxml_install_dir_path
      $ make
      $ make install 

   There might be errors complaining that it can not create regular file libxml2.py in /usr/lib/python2.7/site-packages, but that should not matter.

 * **ELPA**: Download elpa-2018.05.001 from \url{https://elpa.mpcdf.mpg.de/elpa-tar-archive} and follow the installation instructions in there.

Instructions for deal.II's dependencies: p4est, PETSc, SLEPc, and ScaLAPACK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * **p4est**: This library is used by deal.II to create and distribute finite-element meshes across multiple processors. Download the current release tarball of p4est from \url{http://www.p4est.org/}. Also download the script from \url{https://github.com/dftfeDevelopers/dftfe/raw/manual/p4est-setup.sh} if using Intel compilers, or from \url{https://dealii.org/developer/external-libs/p4est.html} if using GCC compilers. Use the script to automatically compile and install a debug and optimized version of p4est by doing::

      $ chmod u+x p4est-setup.sh
      $ ./p4est-setup.sh p4est-x-y-z.tar.gz p4est_install_dir_path

 * **PETSc**: PETSc is a parallel linear algebra library. `DFT-FE` needs two variants of the PETSc installation- one with real scalar type and the another with complex scalar type. Also both the installation variants must have 64-bit indices and optimized mode enabled during the installation. To install PETSc, first download the current release (tested up to 3.9.2) tarball from \url{https://www.mcs.anl.gov/petsc/download/index.html}, unpack it, and follow the installation instructions in \url{https://www.mcs.anl.gov/petsc/documentation/installation.html}. 

  Below, we show an example installation for the real scalar type variant. 
  This example should be used only as a reference.::

      $ ./configure --prefix=petscReal_install_dir_path --with-debugging=no 
                    --with-64-bit-indices=true --with-cc=c_compiler
                    --with-cxx=c++_compiler --with-fc=fortran_compiler
                    --with-blas-lapack-lib=(optimized BLAS-LAPACK library path) 
                    CFLAGS=c_compilter_flags CXXFLAGS=c++_compiler_flags
                            FFLAGS=fortran_compiler_flags

      $ make PETSC_DIR=prompted by PETSc 
             PETSC_ARCH=prompted by PETSc

      $ make PETSC_DIR=prompted by PETSc 
             PETSC_ARCH=prompted by PETSc
             install

  For the complex installation variant, unpack a fresh PETSc directory from the tarball and repeat the above steps with the only changes being adding  `--with-scalar-type=complex` and `--with-clanguage=cxx` to the configuration step (`./configure`) as well as providing a new installation path to `--prefix`.

  Please notice that we have used place holders for values of some of the above configuration flags. You have to use the correct values specific to the compilers and MPI libraries you are working with. Also make sure to follow compiling recommendations for the high-performance computer you are compiling on. For example, if using Intel compilers and Intel MKL for BLAS-LAPACK, it is **very important** to use `Intel MKL Link Line Advisor <https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_ to set the appropriate path for ```--with-blas-lapack-lib=```. It can be something like::

      --with-blas-lapack-lib="-Wl,--start-group 
      ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a 
      ${MKLROOT}/lib/intel64/libmkl_intel_thread.a 
      ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
      -liomp5 -lpthread -lm -ldl" 

 *  **SLEPc**: The SLEPc library is built on top of PETSc, and it is used in DFT-FE for Gram-Schmidt Orthogonalization. To install SLEPc, first download the current release (tested up to 3.9.1) tarball from \url{http://slepc.upv.es/download/}, and then follow the installation procedure described in \url{http://slepc.upv.es/documentation/instal.htm}.

   .. note::

       SLEPc installation requires PETSc to be installed first. You also need to create two separate SLEPc
       installations: one for PETSc installed with `--with-scalar-type=real`, and the second for PETSc installed with
       `--with-scalar-type=complex`. 
	
For your reference you provide here an example installation of SLEPc for real scalar type::

    $ export PETSC_DIR=petscReal_install_dir_path
    $ unset PETSC_ARCH
    $ cd downloaded_slepc_dir
    $ ./configure --prefix=slepcReal_install_dir_path
    $ make
    $ make install

 *  **ScaLAPACK**: ScaLAPACK library is used by DFT-FE via deal.II for its parallel linear algebra routines involving dense matrices. ScaLAPACK is already installed in most high-performance computers. For example, in case of Intel MKL, linking to pre-installed ScaLAPACK libraries would be something like (obtained via `Intel MKL Link Line Advisor <https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_)::

    ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.so
    ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.so

  where `$MKLROOT` points to the directory path for Intel MKL library. It is important to note that the second line above points to the BLACS library, which ScaLAPACK requires to be linked with, and the choice of the BLACS library depends upon the MPI library one is using. For instance, the above example is shown for Intel MPI library. For Open MPI library, the BLACS path would become something like::

      ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.so
      \end{verbatim}

 * **Installing ScaLAPACK from scratch**
   Do not use these instructions if you already have pre-installed ScaLAPACK libraries on your high-performance computer.
   Download the current release version from \url{http://www.netlib.org/scalapack/#\_software}, and build a shared library (use `BUILD_SHARED_LIBS=ON` and `BUILD_STATIC_LIBS=OFF`  during the cmake configuration) installation of ScaLAPACK using cmake. BLACS library, which is required for linking to Intel MKL ScaLAPACK, is not required to be installed separately as it is compiled along with the ScaLAPACK library. Hence you just have to link to `/your_scalapack_installation_dir/lib/libscalapack.so` for using the ScaLAPACK library. For best performance, ScaLAPACK must be linked to optimized BLAS-LAPACK libraries by using\\ `USE_OPTIMIZED_LAPACK_BLAS=ON`, and providing external paths to BLAS-LAPACK during the cmake configuration.   	

Instructions for deal.II
^^^^^^^^^^^^^^^^^^^^^^^^

Assuming the above dependencies (p4est, PETSc, SLEPc, and ScaLAPACK) are installed, we now briefly discuss the steps to compile and install deal.II library linked with the above dependencies. You need to install two variants of the deal.II library: one variant linked with real scalar type PETSc and SLEPc installations, and the other variant linked with complex scalar type PETSc and SLEPc installations. 

 * Obtain the development version of deal.II library via::

        $ git clone -b dealiiStable https://github.com/dftfeDevelopers/dealii.git

 * In addition to requiring C, C++ and Fortran compilers, MPI library, and CMake, deal.II additionaly requires BOOST library. If not found externally, cmake will resort to the bundled BOOST that comes along with deal.II. Based on our experience, we recommend to use the bundled boost (enforced by unsetting/unloading external BOOST library environment paths) to avoid compilation issues.

 * Build deal.II for real numbers::

        $ mkdir buildReal
        $ cd buildReal
        $ cmake -DCMAKE_INSTALL_PREFIX=dealii_petscReal_install_dir_path \
                -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_64BIT_INDICES=ON \
                -DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=p4est_install_dir_path \
                -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR=petscReal_install_dir_path \
                -DDEAL_II_WITH_SLEPC=ON -DSLEPC_DIR=slepcReal_install_dir_path \
                -DDEAL_II_WITH_LAPACK=ON \
                -DLAPACK_DIR=lapack_dir_path \
                -DLAPACK_LIBRARIES=lapack_lib_path \
                -DLAPACK_FOUND=true \
                -DSCALAPACK_DIR=scalapack_dir_path \
                -DSCALAPACK_LIBRARIES=scalapack_lib_path \
                ../deal.II
        $ make install


  .. note::
     Linking with ScaLAPACK is optional, but strongly recommended for systems with 5000 electrons or more.

*  Repeat above step for installing deal.II linked with complex scalar type PETSc and SLEPc installations. 


For more information about installing deal.II library refer to \url{https://dealii.org/developer/readme.html}.
We also provide here an example of deal.II installation, which we did on a high-performance computer
(`STAMPEDE2 <https://www.tacc.utexas.edu/systems/stampede2>`_) using Intel compilers and Intel MPI library
(CXX flags used below are specific to the architecture)::

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_Fortran_COMPILER=mpif90 \
      -DCMAKE_CXX_FLAGS="-xMIC-AVX512" -DCMAKE_C_FLAGS="-xMIC-AVX512" \
      -DDEAL_II_CXX_FLAGS_RELEASE=-O3 -DDEAL_II_COMPONENT_EXAMPLES=OFF \
      -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_64BIT_INDICES=ON \
      -DDEAL_II_WITH_P4EST=ON \
      -DP4EST_DIR=p4est_install_dir_path \
      -DDEAL_II_WITH_PETSC=ON  \
      -DPETSC_DIR=petsc_install_dir_path \
      -DDEAL_II_WITH_SLEPC=ON \
      -DSLEPC_DIR=petsc_install_dir_path \
      -DDEAL_II_WITH_LAPACK=ON \
      -DLAPACK_DIR="${MKLROOT}/lib/intel64" -DLAPACK_FOUND=true \
      -DLAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so; \
      ${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so"  \
      -DLAPACK_LINKER_FLAGS="-liomp5 -lpthread -lm -ldl" \
      -DSCALAPACK_DIR="${MKLROOT}/lib/intel64" \
      -DSCALAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.so; \
      ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.so" \
      -DCMAKE_INSTALL_PREFIX=dealii_install_dir_path \
      ../dealii
    $ make -j 8
    $ make install

The values for `-DLAPACK_DIR`,`-DLAPACK_LIBRARIES`, `-DLAPACK_LINKER_FLAGS`,`-DSCALAPACK_DIR`, and
`-DSCALAPACK_LIBRARIES` were obtained with the help of
`Intel MKL Link Line Advisor <https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_. 

.. note::
    Note that in the above procedure one is installing the development version of deal.II library and this version is continuously updated by deal.II developers, which can sometimes lead to installation issues on certain compilers. If you face any issues during the installation procedure of deal.II development version as explained above, you may alternatively obtain the current release version of deal.II by downloading and unpacking the .tar.gz file from \url{https://www.dealii.org/download.html} and following the same procedure as above. If you still face installation issues, and/or if you have any questions about the deal.II installation, please contact the deal.II developers at `deal.II mailing lists <https://groups.google.com/d/forum/dealii>`_.

Using AVX, AVX-512 instructions in deal.II:
*******************************************

deal.II compilation will automatically try to pick the available vector instructions on the sytem like SSE2, AVX and AVX-512, and generate the following output message during compilation::

    -- Performing Test DEAL_II_HAVE_SSE2
    -- Performing Test DEAL_II_HAVE_SSE2 - Success/Failed
    -- Performing Test DEAL_II_HAVE_AVX
    -- Performing Test DEAL_II_HAVE_AVX - Success/Failed
    -- Performing Test DEAL_II_HAVE_AVX512
    -- Performing Test DEAL_II_HAVE_AVX512 - Success/Failed
    -- Performing Test DEAL_II_HAVE_OPENMP_SIMD
    -- Performing Test DEAL_II_HAVE_OPENMP_SIMD - Success/Failed

``Success``, means deal.II was able to use the corresponding vector instructions, and ``Failed`` would mean otherwise. If deal.II is not able to pick an available vector instruction on your high-performance computer, please contact the deal.II developers at `deal.II mailing lists <https://groups.google.com/d/forum/dealii>`_ and/or contact your high-performance computer support for guidance on how to use the correct compiler flags for AVX or AVX-512. 

Ensure that deal.II picks up AVX-512, which is strongly recommended for obtaining maximum performance on the new Intel Xeon Phi (KNL) and Skylake processors, both of which support Intel AVX-512 instructions.

Important generic instructions for deal.II and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 *  We strongly recommend to link to optimized BLAS-LAPACK library. If using Intel MKL for BLAS-LAPACK library, it is **very important** to use `Intel MKL Link Line Advisor <https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_ to correctly link with Intel MKL for installations of PETSc, ScaLAPACK, and deal.II. To exploit performance benefit from threads, we recommend (strongly recommended for the new Intel Xeon Phi (KNL) and Skylake processors) linking to threaded versions of Intel MKL libraries by using the options ``threading layer`` and  ``OpenMP library`` in `Intel MKL Link Line Advisor <https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_.

 *  Use `-fPIC` compiler flag for compilation of deal.II and its dependencies, to prevent linking errors during `DFT-FE` compilation.	

 .. warning::
   It is  highly recommended to compile deal.II and its dependencies (p4est, PETSc, SLEPc, and ScaLAPACK),  with the same compilers, same BLAS-LAPACK libraries, and same MPI libraries. This prevents deal.II compilation issues, occurence of run time crashes, and `DFT-FE` performance degradation.

Obtaining and Compiling `DFT-FE`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that you have already installed the above external dependencies, next follow the steps below to obtain and compile `DFT-FE`.

 *  Obtain the source code of the current release of `DFT-FE` with all current patches using `Git <https://git-scm.com/>`_::

        $ git clone -b release https://github.com/dftfeDevelopers/dftfe.git
        $ cd dftfe

 * Do `git pull` in the `dftfe` directory any time to obtain new patches that have been added since your `git clone` or last `git pull`.
   If you are not familiar with Git, you may download the current release tarball from the `Downloads <https://sites.google.com/umich.edu/dftfe/download>`_ page in our website, but downloading via Git is recommended to avail new patches seamlessly. 


 * **Obtaining previous releases:** (Skip this part if you are using the current release version)::
        $ git clone https://github.com/dftfeDevelopers/dftfe.git 
        $ cd dftfe
        $ git checkout tags/<tag_name> 

   Alternatively, you could download the appropriate tarball from `github-releases <https://github.com/dftfeDevelopers/dftfe/releases>`_.

 * Set paths to external libraries (deal.II, ALGLIB, Libxc, spglib, and Libxml2), compiler options, and compiler flags in `setup.sh`, which is a script to compile `DFT-FE` using cmake. For your reference, a few example `setup.sh` scripts are provided in the `/helpers` folder. If you are using GCC compilers, please add `-fpermissive` to the compiler flags (see for example `UMCAEN/setupCAEN.sh`). Also if you have installed deal.II by linking with Intel MKL library, set `withIntelMkl=ON` in setup.sh , otherwise set it to `OFF`. 

 * To compile `DFT-FE` in release mode (the fast version), set `optimizedFlag=1` in `setup.sh` and do::

        $ ./setup.sh

   If compilation is successful, a `/build` directory will be created with the following executables:

      * /build/release/real/dftfe
      * /build/release/complex/dftfe

   To compile `DFT-FE` in debug mode (much slower but useful for debugging), set `optimizedFlag=0` in `setup.sh` and do::

        $ ./setup.sh

    which will create the following debug mode executables:

      * /build/debug/real/dftfe
      * /build/debug/complex/dftfe
