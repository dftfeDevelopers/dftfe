// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef dftfeFEBasisOperations_h
#define dftfeFEBasisOperations_h

#include <MultiVector.h>
#include <headers.h>
#include <constraintMatrixInfo.h>
#include <constraintMatrixInfoDevice.h>
namespace dftfe
{
  namespace basis
  {
    enum UpdateFlags
    {
      update_default = 0,

      update_values = 0x0001,

      update_gradients = 0x0002,

      update_macrocell_map = 0x0004
    };
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    class FEBasisOperations
    {
    public:
      FEBasisOperations(
        dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
        std::vector<const dealii::AffineConstraints<ValueTypeBasisCoeff> *>
          &constraintsVector);

      ~FEBasisOperations() = default;

      void
      reinit(const unsigned int &blockSize,
             const unsigned int &dofHandlerID,
             const unsigned int &quadratureID,
             const UpdateFlags   updateFlags = update_values);

      void
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *  quadratureGradients         = NULL,
        bool useMacroCellSubCellOrdering = false) const;


      void
      interpolate(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                    memorySpace> &nodalData,
                  dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
                    *quadratureValues,
                  dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
                    *  quadratureGradients         = NULL,
                  bool useMacroCellSubCellOrdering = false) const;

      void
      integrateWithBasis(
        const std::map<
          dealii::CellId,
          dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                      dftfe::utils::MemorySpace::HOST>>
          &quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering = false) const;


      void
      integrateWithBasis(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering = false) const;

      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *  cellNodalDataPtr,
        bool useMacroCellSubCellOrdering = false) const;

      void
      accumulateFromCellNodalData(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering = false) const;

    private:
#if defined(DFTFE_WITH_DEVICE)
      using constraintInfoClass =
        typename std::conditional<memorySpace ==
                                    dftfe::utils::MemorySpace::DEVICE,
                                  dftUtils::constraintMatrixInfoDevice,
                                  dftUtils::constraintMatrixInfo>::type;
#else
      using constraintInfoClass = dftUtils::constraintMatrixInfo;
#endif



      void
      initializeIndexMaps();

      void
      initializeConstraints();

      void
      initializeShapeFunctionAndJacobianData();

      void
      interpolateHostKernel(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                   quadratureGradients,
        std::pair<unsigned int, unsigned int> cellRange,
        bool useMacroCellSubCellOrdering = false) const;

      void
      integrateWithBasisHostKernel(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                   nodalData,
        std::pair<unsigned int, unsigned int> cellRange,
        bool useMacroCellSubCellOrdering = false) const;


      void
      extractToCellNodalDataHostKernel(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                   cellNodalDataPtr,
        std::pair<unsigned int, unsigned int> cellRange,
        bool useMacroCellSubCellOrdering = false) const;

      void
      accumulateFromCellNodalDataHostKernel(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                   nodalData,
        std::pair<unsigned int, unsigned int> cellRange,
        bool useMacroCellSubCellOrdering = false) const;



      constraintInfoClass d_constraintInfo;
      std::vector<const dealii::AffineConstraints<ValueTypeBasisCoeff> *>
        *                                              d_constraintsVector;
      const dealii::MatrixFree<3, ValueTypeBasisData> *d_matrixFreeDataPtr;
      std::vector<size_type> d_cellDofIndexToProcessDofIndexMap;
      std::vector<size_type> d_macroCellSubCellDofIndexToProcessDofIndexMap;
      std::vector<size_type> d_cellIndexToMacroCellSubCellIndexMap;
      std::vector<dealii::CellId> d_cellIndexToCellIdMap;
      std::map<dealii::CellId,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_inverseJacobianData;
      std::map<dealii::CellId,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_JxWData;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_shapeFunctionData;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
                   d_shapeFunctionGradientData;
      unsigned int d_quadratureID;
      unsigned int d_dofHandlerID;
      unsigned int d_nVectors;
      unsigned int d_nCells;
      unsigned int d_nMacroCells;
      unsigned int d_nDofsPerCell;
      unsigned int d_nQuadsPerCell;
      bool         areAllCellsAffine;
      UpdateFlags  d_updateFlags;

    }; // end of FEBasisOperations
  }    // end of namespace basis
} // end of namespace dftfe
#endif // dftfeBasisOperations_h
