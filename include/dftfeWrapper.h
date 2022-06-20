// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//

#ifndef dftfeWrapper_H_
#define dftfeWrapper_H_

#include <mpi.h>
#include <string>
#include <vector>

namespace dftfe
{
  class dftBase;
  class dftParameters;
  /**
   * @brief wrapper class for dftfe
   *
   * @author Sambit Das
   */
  class dftfeWrapper
  {
  public:
    /**
     * @brief must be called only once at start of program from all processors
     * after calling MPI_Init
     */
    static void
    globalHandlesInitialize();

    /**
     * @brief must be called only once at end of program from all processors
     * but before calling MPI_Finalize
     */
    static void
    globalHandlesFinalize();

    /**
     * @brief constructor based on input parameter_file
     */
    dftfeWrapper(const std::string parameter_file,
                 const MPI_Comm &  mpi_comm_parent,
                 const bool        printParams                      = false,
                 const bool        setGPUToMPITaskBindingInternally = false);

    /**
     * @brief constructor based on input parameter_file and restart
     * coordinates and domain vectors file paths
     */
    dftfeWrapper(const std::string parameter_file,
                 const std::string restartCoordsFile,
                 const std::string restartDomainVectorsFile,
                 const MPI_Comm &  mpi_comm_parent,
                 const bool        printParams                      = false,
                 const bool        setGPUToMPITaskBindingInternally = false);

    /**
     * @brief constructor based on input list of atomic coordinates,
     * list of atomic numbers,cell, boundary conditions,
     * Monkhorst-Pack k-point grid, and other optional parameters.
     * This constructor currently only sets up GGA PBE pseudopotential
     * DFT calculations using ONCV pseudopotentials in .upf format
     * (read from DFTFE_PSP_PATH folder provided as an environment
     * variable). The pseudpotential directory must contain files in the
     * format: AtomicSymbol.upf
     *
     * @param[in] mpi_comm_parent mpi communicator to be used by the
     * dftfeWrapper.
     * @param[in] useGPU toggle use of GPU accelerated DFT-FE
     * @param[in] atomicPositionsCart vector of atomic positions for
     * each atom (in Bohr units), Origin is at cell corner
     * @param[in] atomicNumbers vector of atomic numbers
     * @param[in] cell 3 \times 3 matrix in Bohr units, cell[i] denotes the ith
     * cell vector. DFT-FE requires the cell vectors to form a
     * right-handed coordinate system i.e.
     * dotProduct(crossProduct(cell[0],cell[1]),cell[2])>0
     * @param[in] pbc vector of bools denoting periodic boundary conditions
     * along the three cell vectors, false denotes non-periodic and true is
     * periodic
     * @param[in] mpgrid vector of Monkhorst-Pack grid points along the
     * reciprocal lattice vector directions for sampling the Brillouin zone
     * along periodic directions. Default value is a Gamma point.
     * @param[in] mpgridShift vector of bools where false denotes no shift and
     * true denotes shift by half the Monkhost-Pack grid spacing. Default value
     * is no shift.
     * @param[in] spinPolarizedDFT toggles spin-polarized DFT calculations.
     * Default value is false
     * @param[in] startMagnetization Starting magnetization to be used for
     * spin-polarized DFT calculations (must be between -0.5 and +0.5).
     * Corresponding magnetization per simulation domain will be
     * (2 x START MAGNETIZATION x Number of electrons) in Bohr magneton units.
     * @param[in] fermiDiracSmearingTemp Fermi-Dirac smearing temperature in
     * Kelvin. Default value is 500 K.
     * @param[in] npkpt Number of groups of MPI tasks across which the work load
     * of the irreducible k-points is parallelised. npkpt must be a divisor of
     * total number of MPI tasks. Default value of 0 internally sets npkt to an
     * heuristically determined value.
     * @param[in] meshSize Finite-element mesh size around the atoms in Bohr
     * units. The default value of 0.8 is sufficient to achieve chemical
     * accuracy in energy (0.1 mHa/atom discretization error) and forces (0.1
     * mHa/Bohr discretization error) for the ONCV pseudo-dojo
     * pseudopotentials. Note that this function assumes a sixth order
     * finite-element interpolating polynomial
     * @param[in] scfMixingParameter mixing paramter for SCF fixed point
     * iteration. Currently the Anderson mixing strategy is used.
     * @param[in] verbosity printing verbosity. Default value is -1: no printing
     * @param[in] setGPUToMPITaskBindingInternally This option is only valid for
     * GPU runs. If set to true GPU to MPI task binding is set inside the DFT-FE
     * code. Default behaviour is false which assumes the binding has been
     * externally set.
     */
    dftfeWrapper(const MPI_Comm &                       mpi_comm_parent,
                 const bool                             useGPU,
                 const std::vector<std::vector<double>> atomicPositionsCart,
                 const std::vector<unsigned int>        atomicNumbers,
                 const std::vector<std::vector<double>> cell,
                 const std::vector<bool>                pbc,
                 const std::vector<unsigned int>        mpGrid =
                   std::vector<unsigned int>{1, 1, 1},
                 const std::vector<bool> mpGridShift = std::vector<bool>{false,
                                                                         false,
                                                                         false},
                 const bool              spinPolarizedDFT       = false,
                 const double            startMagnetization     = 0.0,
                 const double            fermiDiracSmearingTemp = 500.0,
                 const unsigned int      npkpt                  = 0,
                 const double            meshSize               = 0.8,
                 const double            scfMixingParameter     = 0.2,
                 const int               verbosity              = -1,
                 const bool setGPUToMPITaskBindingInternally    = false);


    ~dftfeWrapper();

    /**
     * @brief clear and reinitialize based on input parameter_file
     */
    void
    reinit(const std::string parameter_file,
           const MPI_Comm &  mpi_comm_parent,
           const bool        printParams                      = false,
           const bool        setGPUToMPITaskBindingInternally = false);

    /**
     * @brief clear and reinitialize based on input parameter_file and restart
     * coordinates and domain vectors file paths
     */
    void
    reinit(const std::string parameter_file,
           const std::string restartCoordsFile,
           const std::string restartDomainVectorsFile,
           const MPI_Comm &  mpi_comm_parent,
           const bool        printParams                      = false,
           const bool        setGPUToMPITaskBindingInternally = false);

    void
    reinit(const MPI_Comm &                       mpi_comm_parent,
           const bool                             useGPU,
           const std::vector<std::vector<double>> atomicPositionsCart,
           const std::vector<unsigned int>        atomicNumbers,
           const std::vector<std::vector<double>> cell,
           const std::vector<bool>                pbc,
           const std::vector<unsigned int>        mpGrid =
             std::vector<unsigned int>{1, 1, 1},
           const std::vector<bool> mpGridShift        = std::vector<bool>{false,
                                                                   false,
                                                                   false},
           const bool              spinPolarizedDFT   = false,
           const double            startMagnetization = 0.0,
           const double            fermiDiracSmearingTemp           = 500.0,
           const unsigned int      npkpt                            = 0,
           const double            meshSize                         = 0.8,
           const double            scfMixingParameter               = 0.2,
           const int               verbosity                        = -1,
           const bool              setGPUToMPITaskBindingInternally = false);

    void
    clear();

    /**
     * @brief Legacy function (to be deprecated)
     */
    void
    run();

    /**
     * @brief solve ground-state and return DFT free energy which is sum of internal
     * energy and negative of electronic entropic energy (in Hartree units)
     */
    double
    computeDFTFreeEnergy(const bool computeIonForces  = true,
                         const bool computeCellStress = false);

    /**
     * @brief Get electronic entropic energy (in Hartree units). This function can
     * only be called after calling computeDFTFreeEnergy
     */
    double
    getElectronicEntropicEnergy() const;

    /**
     * @brief Get ionic forces: negative of gradient of DFT free energy with
     * respect to ionic positions (in Hartree/Bohr units). This function can
     * only be called after calling computeDFTFreeEnergy
     *
     *  @return vector of forces on each atom
     */
    std::vector<std::vector<double>>
    getForcesAtoms() const;

    /**
     * @brief Get cell stress: negative of gradient of DFT free energy
     * with respect to affine strain components scaled by volume
     * (Hartree/Bohr^3) units. This function can only
     * be called after calling computeDFTFreeEnergy
     *
     * @return cell stress 3 \times 3 matrix given by
     *  sigma[i][j]=\frac{1}{\Omega}\frac{\partial E}{\partial \epsilon_{ij}}
     */
    std::vector<std::vector<double>>
    getCellStress() const;

    /**
     * @brief update atom positions and reinitialize all related  data-structures
     *
     * @param[in] atomsDisplacements vector of displacements for
     * each atom (in Bohr units)
     */
    void
    updateAtomPositions(
      const std::vector<std::vector<double>> atomsDisplacements);


    /**
     *@brief Deforms the cell by applying the given affine deformation gradient and
     * reinitializes the underlying data-structures.
     *
     *@param[in] deformationGradient deformation gradient
     * matrix given by F[i][j]=\frac{\partial x_i}{\partial X_j}
     */
    void
    deformCell(const std::vector<std::vector<double>> deformationGradient);

    /**
     * @brief Gets the current atom Positions in cartesian form (in Bohr units)
     * (origin at corner of cell against which the cell vectors are defined)
     *
     *  @return array of coords for each atom
     */
    std::vector<std::vector<double>>
    getAtomPositionsCart() const;

    /**
     * @brief Gets the current atom Positions in fractional form
     * (only applicable for periodic and semi-periodic BCs).
     * CAUTION: during relaxation and MD fractional coordinates may have negaive
     * values
     *
     *  @return array of coords for each atom
     */
    std::vector<std::vector<double>>
    getAtomPositionsFrac() const;



    /**
     * @brief Gets the current cell vectors
     *
     *  @return  3 \times 3 matrix, cell[i][j] corresponds to jth component of
     *  ith cell vector (in Bohr units)
     */
    std::vector<std::vector<double>>
    getCell() const;


    /**
     * @brief Gets the boundary conditions for each cell vector direction
     *
     *  @return vector of bools, false denotes non-periodic BC and true denotes periodic BC
     */
    std::vector<bool>
    getPBC() const;

    /**
     * @brief Gets the atomic numbers vector
     *
     *  @return vector of atomic numbers
     */
    std::vector<int>
    getAtomicNumbers() const;


    /**
     * @brief Gets the number of valence electrons for each atom
     *
     *  @return array of number of valence for each atom
     */
    std::vector<int>
    getValenceElectronNumbers() const;


    dftBase *
    getDftfeBasePtr();

  private:
    void
    createScratchFolder();

    void
    initialize(const bool setGPUToMPITaskBindingInternally);

    MPI_Comm       d_mpi_comm_parent;
    dftBase *      d_dftfeBasePtr;
    dftParameters *d_dftfeParamsPtr;
    std::string    d_scratchFolderName;
    bool           d_isGPUToMPITaskBindingSetInternally;
  };
} // namespace dftfe
#endif
