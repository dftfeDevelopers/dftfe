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
// ---------------------------------------------------------------------
//

#if defined(DFTFE_WITH_DEVICE)
#  ifndef solveVselfInBinsDevice_H_
#    define solveVselfInBinsDevice_H_

#    include <constraintMatrixInfo.h>
#    include <headers.h>
#    include <BLASWrapper.h>
namespace dftfe
{
  namespace poissonDevice
  {
    void
    solveVselfInBins(
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellGradNIGradNJIntergralDevice,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                              &BLASWrapperPtr,
      const dealii::MatrixFree<3, double>     &matrixFreeData,
      const dftfe::uInt                        mfDofHandlerIndex,
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const double                            *rhsFlattenedH,
      const double                            *diagonalAH,
      const double                            *inhomoIdsColoredVecFlattenedH,
      const dftfe::uInt                        localSize,
      const dftfe::uInt                        ghostSize,
      const dftfe::uInt                        numberBins,
      const MPI_Comm                          &mpiCommParent,
      const MPI_Comm                          &mpiCommDomain,
      double                                  *xH,
      const dftfe::Int                         verbosity,
      const dftfe::uInt                        maxLinearSolverIterations,
      const double                             absLinearSolverTolerance,
      const bool isElectroFEOrderDifferentFromFEOrder = false);

    void
    cgSolver(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
                   &constraintsMatrixDataInfoDevice,
      const double *bD,
      const double *diagonalAD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &poissonCellStiffnessMatricesD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &inhomoIdsColoredVecFlattenedD,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
                                   &cellLocalProcIndexIdMapD,
      const dftfe::uInt             localSize,
      const dftfe::uInt             ghostSize,
      const dftfe::uInt             numberBins,
      const dftfe::uInt             totalLocallyOwnedCells,
      const dftfe::uInt             numberNodesPerElement,
      const dftfe::Int              debugLevel,
      const dftfe::uInt             maxIter,
      const double                  absTol,
      const MPI_Comm               &mpiCommParent,
      const MPI_Comm               &mpiCommDomain,
      distributedDeviceVec<double> &x);
  } // namespace poissonDevice
} // namespace dftfe
#  endif
#endif
