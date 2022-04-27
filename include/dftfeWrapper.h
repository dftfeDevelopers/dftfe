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
    static void
    globalHandlesInitialize();

    static void
    globalHandlesFinalize();

    dftfeWrapper(const std::string parameter_file,
                 const MPI_Comm &  mpi_comm_parent,
                 const bool        printParams                      = false,
                 const bool        setGPUToMPITaskBindingInternally = false);

    ~dftfeWrapper();

    void
    reinit(const std::string parameter_file,
           const MPI_Comm &  mpi_comm_parent,
           const bool        printParams                      = false,
           const bool        setGPUToMPITaskBindingInternally = false);

    void
    clear();

    /**
     * @brief Legacy function (to be deprecated)
     */
    void
    run();

    /**
     * @brief solve ground-state and return DFT free energy which is sum of internal
     * energy and negative of electronic entropic energy
     */
    double
    computeDFTFreeEnergy();


    /**
     * @brief Get negative of gradient of DFT free energy with respect to ionic positions
     *
     *  @return std::vector<std::vector<double>> vector of forces on each atom
     */
    std::vector<std::vector<double>>
    getForcesAtoms() const;

    /**
     * @brief Get negative of gradient of DFT free energy with respect to affine strain components
     *
     * @return std::vector<std::vector<double> > 3 \times 3 matrix given by
     *  sigma[i][j]=\frac{1}{\Omega}\frac{\partial E}{\partial \epsilon_{ij}}
     */
    std::vector<std::vector<double>>
    getCellStress() const;

    /**
     * @brief update atom positions and reinitialize all related  datastructures
     *
     * @param[in] std::vector<double> vector of displacements for each atom
     */
    void
    updateAtomPositions(
      const std::vector<std::vector<double>> atomsDisplacements);


    /**
     *@brief Deforms the domain by the given affine deformation gradient and
     * reinitializes the underlying datastructures.
     *
     *@param[in] std::vector<std::vector<double>> deformation gradient
     * matrix given by F[i][j]=\frac{\partial x_i}{\partial X_j}
     */
    void
    deformDomain(const std::vector<std::vector<double>> deformationGradient);

    /**
     * @brief Gets the current atom Locations in cartesian form
     * (origin at center of domain)
     *
     *  @return std::vector<std::vector<double>> array of coords for each atom
     */
    std::vector<std::vector<double>>
    getAtomLocationsCart() const;

    /**
     * @brief Gets the current atom Locations in fractional form
     * (only applicable for periodic and semi-periodic BCs).
     * CAUTION: during relaxation and MD fractional coordinates may have negaive
     * values
     *
     *  @return std::vector<std::vector<double>> array of coords for each atom
     */
    std::vector<std::vector<double>>
    getAtomLocationsFrac() const;



    /**
     * @brief Gets the current cell lattice vectors
     *
     *  @return std::vector<std::vector<double>> 3 \times 3 matrix,lattice[i][j] corresponds to jth component of
     *  ith lattice vector
     */
    std::vector<std::vector<double>>
    getCell() const;


    /**
     * @brief Gets the boundary conditions for each lattice vector direction
     *
     *  @return std::vector<bool> false denotes non-periodic BC and true denotes periodic BC
     */
    std::vector<bool>
    getPBC() const;

    /**
     * @brief Gets the atomic numbers vector
     *
     *  @return std::vector<double> array of atomic number for each atom
     */
    std::vector<int>
    getAtomicNumbers() const;


    /**
     * @brief Gets the number of valence electrons for each atom
     *
     *  @return std::vector<double> array of number of valence for each atom
     */
    std::vector<int>
    getValenceElectronNumbers() const;


    dftBase *
    getDftfeBasePtr();

  private:
    dftBase *      d_dftfeBasePtr;
    dftParameters *d_dftfeParamsPtr;
  };
} // namespace dftfe
#endif
