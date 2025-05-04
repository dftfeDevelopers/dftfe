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

#ifndef DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H
#define DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H


#include "InterpolateCellWiseDataToPoints.h"
#include "headers.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "vectorUtilities.h"


namespace dftfe
{
  /**
   * @brief This class provides the interface for the transfer between the meshes
   *
   * @tparam memorySpace
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class TransferDataBetweenMeshesIncompatiblePartitioning
  {
  public:
    TransferDataBetweenMeshesIncompatiblePartitioning(
      const dealii::MatrixFree<3, double> &matrixFreeMesh1,
      const dftfe::uInt                    matrixFreeMesh1VectorComponent,
      const dftfe::uInt                    matrixFreeMesh1QuadratureComponent,
      const dealii::MatrixFree<3, double> &matrixFreeMesh2,
      const dftfe::uInt                    matrixFreeMesh2VectorComponent,
      const dftfe::uInt                    matrixFreeMesh2QuadratureComponent,
      const dftfe::uInt                    verbosity,
      const MPI_Comm                      &mpiComm,
      const bool useMemOptForCellWiseInterpolation = false);

    void
    interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                           &BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                              memorySpace> &inputVec,
      const dftfe::uInt                                     numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
        &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
                       &outputQuadData,
      const dftfe::uInt blockSizeOfInputData,
      const dftfe::uInt blockSizeOfOutputData,
      const dftfe::uInt startIndexOfInputData,
      bool              resizeOutputVec); // override;

    void
    interpolateMesh2DataToMesh1QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                           &BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                              memorySpace> &inputVec,
      const dftfe::uInt                                     numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
        &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
                       &outputQuadData,
      const dftfe::uInt blockSizeOfInputData,
      const dftfe::uInt blockSizeOfOutputData,
      const dftfe::uInt startIndexOfInputData,
      bool              resizeOutputVec); // override;


    void
    interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
      const distributedCPUVec<dftfe::dataTypes::number> &inputVec,
      const dftfe::uInt                                  numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::HOST>
        &fullFlattenedArrayCellLocalProcIndexIdMapParent,
      dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
                       &outputQuadData,
      const dftfe::uInt blockSizeOfInputData,
      const dftfe::uInt blockSizeOfOutputData,
      const dftfe::uInt startIndexOfInputData,
      bool              resizeOutputVec); // override;

    void
    interpolateMesh2DataToMesh1QuadPoints(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
      const distributedCPUVec<dftfe::dataTypes::number> &inputVec,
      const dftfe::uInt                                  numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::HOST>
        &mapVecToCells,
      dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
                       &outputQuadData,
      const dftfe::uInt blockSizeOfInputData,
      const dftfe::uInt blockSizeOfOutputData,
      const dftfe::uInt startIndexOfInputData,
      bool              resizeOutputVec); // override;

  private:
    const dealii::MatrixFree<3, double> *d_matrixFreeMesh1Ptr;
    const dealii::MatrixFree<3, double> *d_matrixFreeMesh2Ptr;

    dftfe::uInt d_matrixFreeMesh1VectorComponent,
      d_matrixFreeMesh1QuadratureComponent;

    dftfe::uInt d_matrixFreeMesh2VectorComponent,
      d_matrixFreeMesh2QuadratureComponent;

    std::shared_ptr<
      InterpolateCellWiseDataToPoints<dftfe::dataTypes::number, memorySpace>>
      d_mesh1toMesh2;
    std::shared_ptr<
      InterpolateCellWiseDataToPoints<dftfe::dataTypes::number, memorySpace>>
      d_mesh2toMesh1;

    const MPI_Comm d_mpiComm;
  };
} // namespace dftfe
#endif // DFTFE_TRANSFERDATABETWEENMESHESINCOMPATIBLEPARTITIONING_H
