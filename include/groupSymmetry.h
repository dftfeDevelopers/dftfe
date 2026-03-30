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
#include <deal.II/base/mpi_remote_point_evaluation.h>
namespace dftfe
{
  /**
   * @brief density symmetrization based on irreducible Brillouin zone calculation,
   * only relevant for calculations using point group symmetries
   *
   * @author Nikhil Kodali
   */

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
    initGroupSymmetry(std::vector<std::vector<double>> &atomLocations,
                      std::vector<std::vector<double>> &domainBoundingVectors,
                      std::vector<bool> &periodicBoundaryConditions,
                      const bool         isCollinearSpin    = false,
                      const bool         isNonCollinearSpin = false);

    void
    reinitGroupSymmetry(
      std::vector<std::vector<double>> &atomLocations,
      std::vector<std::vector<double>> &domainBoundingVectors);

    void
    setupCommPatternForNodalField(const dealii::DoFHandler<3> &dofHandler);

    bool
    computeAtomIdMapsFromGlobalFractionalCoordinates(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &globalPointCoords);

    void
    symmetrizeScalarFieldFromLocalValues(
      distributedCPUVec<double>   &scalarField,
      const dealii::DoFHandler<3> &dofHandler);

    void
    symmetrizeVectorFieldFromLocalValues(
      distributedCPUVec<double>   &vectorFieldComponentx,
      distributedCPUVec<double>   &vectorFieldComponenty,
      distributedCPUVec<double>   &vectorFieldComponentz,
      const dealii::DoFHandler<3> &dofHandler);

    void
    symmetrizeForce(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &vectorFieldValues) const;

    void
    symmetrizeStress(
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
    std::vector<std::vector<double>> d_symmMatCart;
    std::vector<std::vector<double>> d_symmMatInverse;
    std::vector<std::vector<double>> d_translation;

    dftfe::uInt d_numAtoms;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_atomicCoordsFrac;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_domainBoundingVectors;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_domainBoundingVectorsInverse;


    std::vector<bool> d_periodicBoundaryConditions;

    std::vector<std::vector<dftfe::uInt>> d_pointMapsForSymmetry;

    const bool d_isTimeReversal;
    const bool d_isGroupSymmetry;

    std::vector<std::vector<dftfe::uInt>> localDoFIndexToPointIndexMap;
    mutable dealii::Utilities::MPI::RemotePointEvaluation<3, 3>
                                  remotePointCache;
    std::vector<dealii::Point<3>> requiredPointCoordinates;
    dealii::MappingQ1<3>          mapping;
  };
} // namespace dftfe
#endif
