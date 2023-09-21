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

#ifndef dftBase_H_
#define dftBase_H_

#include <vector>
#include <tuple>
#include <deal.II/base/tensor_function.h>
#include "dftParameters.h"

namespace dftfe
{
  /**
   * @brief abstract base class for dft
   *
   * @author Sambit Das
   */
  class dftBase
  {
  public:
    /**
     * @brief This is required to correctly delete the derived class object
     * through the base class ptr
     */
    virtual ~dftBase(){};

    virtual void
    set() = 0;

    virtual void
    init() = 0;

    virtual void
    run() = 0;

    virtual void
    writeMesh() = 0;

    virtual void
    updateAtomPositionsAndMoveMesh(
      const std::vector<dealii::Tensor<1, 3, double>> &globalAtomsDisplacements,
      const double maxJacobianRatioFactor         = 1.25,
      const bool   useSingleAtomSolutionsOverride = false) = 0;

    /**
     *@brief Deforms the domain by the given deformation gradient and reinitializes the
     * dftClass datastructures.
     */
    virtual void
    deformDomain(const dealii::Tensor<2, 3, double> &deformationGradient,
                 const bool vselfPerturbationUpdateForStress   = false,
                 const bool useSingleAtomSolutionsInitialGuess = false,
                 const bool print                              = true) = 0;


    virtual std::tuple<bool, double>
    solve(const bool computeForces                   = true,
          const bool computeStress                   = true,
          const bool isRestartGroundStateCalcFromChk = false) = 0;

    virtual void
    computeStress() = 0;

    virtual void
    trivialSolveForStress() = 0;

    virtual double
    getInternalEnergy() const = 0;

    virtual double
    getEntropicEnergy() const = 0;

    virtual double
    getFreeEnergy() const = 0;

    virtual distributedCPUVec<double>
    getRhoNodalOut() const = 0;

    virtual distributedCPUVec<double>
    getRhoNodalSplitOut() const = 0;

    virtual double
    getTotalChargeforRhoSplit() = 0;

    virtual void
    resetRhoNodalIn(distributedCPUVec<double> &OutDensity) = 0;

    virtual void
    resetRhoNodalSplitIn(distributedCPUVec<double> &OutDensity) = 0;

    /**
     * @brief Gets the current atom Locations in cartesian form
     * (origin at center of domain) from dftClass
     */
    virtual std::vector<std::vector<double>>
    getAtomLocationsCart() const = 0;

    /**
     * @brief Gets the current atom Locations in fractional form
     * from dftClass (only applicable for periodic and semi-periodic BCs)
     */
    virtual std::vector<std::vector<double>>
    getAtomLocationsFrac() const = 0;



    /**
     * @brief Gets the current cell lattice vectors
     *
     *  @return std::vector<std::vector<double>> 3 \times 3 matrix,lattice[i][j]
     *  corresponds to jth component of ith lattice vector
     */
    virtual std::vector<std::vector<double>>
    getCell() const = 0;

    /**
     * @brief Gets the current cell volume
     *
     */
    virtual double
    getCellVolume() const = 0;

    /**
     * @brief Gets the current atom types from dftClass
     */
    virtual std::set<unsigned int>
    getAtomTypes() const = 0;

    /**
     * @brief Gets the current atomic forces (configurational forces) from dftClass
     */
    virtual std::vector<double>
    getForceonAtoms() const = 0;


    /**
     * @brief Gets the current cell stress from dftClass
     */
    virtual dealii::Tensor<2, 3, double>
    getCellStress() const = 0;

    /**
     * @brief Get reference to dftParameters object
     */
    virtual dftParameters &
    getParametersObject() const = 0;

    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files, which are required for
     * geometry relaxation restart

     */
    virtual void
    writeDomainAndAtomCoordinates() = 0;


    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files for
     * structural optimization and dynamics restarts. The coordinates are stored
     * as: 1. fractional for semi-periodic/periodic 2. Cartesian for
     * non-periodic.
     * @param[in] Path The folder path to store the atom coordinates required
     * during restart.
     */
    virtual void
    writeDomainAndAtomCoordinates(const std::string Path) const = 0;


    /**
     * @brief writes atomistics data for subsequent post-processing. Related to
     * WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS input parameter.
     * @param[in] Path The folder path to store the atomistics data.
     */
    virtual void
    writeStructureEnergyForcesDataPostProcess(const std::string Path) const = 0;

    /**
     * @brief writes quadrature grid information and associated spin-up
     * and spin-down electron-density for post-processing
     * @param[in] Path The folder path to store the atomistics data.
     */
    virtual void
    writeGSElectronDensity(const std::string Path) const = 0;
  };

} // namespace dftfe

#endif
