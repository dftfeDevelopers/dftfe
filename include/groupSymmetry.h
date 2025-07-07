// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#ifndef groupSymmetry_H_
#define groupSymmetry_H_
#include <complex>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include "constants.h"
#include "headers.h"
#include "BLASWrapper.h"
namespace dftfe
{
  /**
   * @brief density symmetrization based on irreducible Brillouin zone calculation,
   * only relevant for calculations using point group symmetries
   *
   * @author Nikhil Kodali
   */
  enum class pointSet
  {
    cellCentroids,
    densityQuad,
    densityNodal,
    atomicCoord
  };

  class groupSymmetryClass
  {
  public:
    /**
     * groupSymmetryClass constructor
     */
    groupSymmetryClass(const MPI_Comm &mpi_comm_parent,
                       const MPI_Comm &mpi_comm_domain,
                       const bool      isGroupSymmetry,
                       const bool      isTimeReversal);

    void
    initGroupSymmetry(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                        dftfe::utils::MemorySpace::HOST>> &BLASWrapperPtrHost,
                      std::vector<std::vector<double>>    &atomLocations,
                      std::vector<std::vector<double>> &domainBoundingVectors,
                      std::vector<bool> &periodicBoundaryConditions,
                      const bool         isCollinearSpin = false);


    void
    computePointMapFromLocalCartesianCoordinates(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtrHost,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                           &localPointCoords,
      const dftfe::pointSet pointSetType,
      const bool            cellOrdered = false);

    void
    computePointMapsFromGlobalFractionalCoordinates(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                           &globalPointCoords,
      const dftfe::pointSet pointSetType,
      const bool            cellOrdered = false);

    void
    symmetrizeScalarFieldFromLocalValues(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                           &scalarFieldValues,
      const dftfe::pointSet pointSetType) const;

    void
    symmetrizeVectorFieldFromLocalValues(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                           &vectorFieldValues,
      const dftfe::pointSet pointSetType) const;

    void
    symmetrizeVectorFieldFromGlobalValues(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                           &vectorFieldValues,
      const dftfe::pointSet pointSetType) const;

    void
    symmetrizeRank2Tensor(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tensorValues) const;

    void
    reduceKPointGrid(std::vector<double> &kPointCoordinatesFrac,
                     std::vector<double> &kPointWeights) const;


  private:
    /**
     * compute-time logger
     */
    dealii::TimerOutput computing_timer;
    /**
     * parallel objects
     */
    const MPI_Comm             d_mpiCommParent, d_mpiCommDomain;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
    /**
     * Space group symmetry related data
     */
    dftfe::uInt                      d_numSymm;
    std::vector<std::vector<double>> d_symmMat;
    std::vector<std::vector<double>> d_translation;

    dftfe::uInt d_numAtoms;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_atomicCoordsCart;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_atomicCoordsFrac;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_domainBoundingVectors;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_domainBoundingVectorsInverse;


    std::vector<bool> d_periodicBoundaryConditions;

    std::map<dftfe::pointSet, std::vector<std::vector<dftfe::uInt>>>
      d_pointMapsForSymmetry;

    const bool d_isTimeReversal;
    const bool d_isGroupSymmetry;
  };
} // namespace dftfe
#endif
