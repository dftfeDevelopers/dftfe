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

#include <FEBasisOperations.h>
namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolateHostKernel(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                   quadratureGradients,
        std::pair<unsigned int, unsigned int> cellRange,
        bool                                  useMacroCellSubCellOrdering) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        cellNodalData, tempQuadratureGradientsData;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell * d_nCells);

      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(d_nVectors * d_nQuadsPerCell * 3);


      extractToCellNodalDataHostKernel(nodalValues,
                                       &cellNodalData,
                                       cellRange,
                                       useMacroCellSubCellOrdering);

      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        {
          const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                    scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
          const char transA = 'N', transB = 'N';

          xgemm(&transA,
                &transB,
                &d_nVectors,
                &d_nQuadsPerCell,
                &d_nDofsPerCell,
                &scalarCoeffAlpha,
                cellNodalData.data() + d_nDofsPerCell * iCell,
                &d_nVectors,
                d_shapeFunctionData.data(),
                &d_nDofsPerCell,
                &scalarCoeffBeta,
                quadratureValues->data() + d_nQuadsPerCell * iCell,
                &d_nVectors);
          if (quadratureGradients != NULL)
            {
              const unsigned int d_nQuadsPerCellTimesThree =
                d_nQuadsPerCell * 3;
              xgemm(&transA,
                    &transB,
                    &d_nVectors,
                    &d_nQuadsPerCellTimesThree,
                    &d_nDofsPerCell,
                    &scalarCoeffAlpha,
                    cellNodalData.data() + d_nDofsPerCell * iCell,
                    &d_nVectors,
                    d_shapeFunctionGradientData.data(),
                    &d_nDofsPerCell,
                    &scalarCoeffBeta,
                    tempQuadratureGradientsData.data(),
                    &d_nVectors);
              const unsigned int d_nQuadsPerCellTimesnVectors =
                d_nQuadsPerCell * d_nVectors;
              const unsigned int three = 3;
              xgemm(&transA,
                    &transB,
                    &d_nQuadsPerCellTimesnVectors,
                    &three,
                    &three,
                    &scalarCoeffAlpha,
                    tempQuadratureGradientsData.data(),
                    &d_nQuadsPerCellTimesnVectors,
                    d_inverseJacobianData.data() + 9 * iCell,
                    &three,
                    &scalarCoeffBeta,
                    quadratureGradients->data() + d_nQuadsPerCell * 3 * iCell,
                    &d_nQuadsPerCellTimesnVectors);
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
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
        bool                                  useMacroCellSubCellOrdering) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        cellNodalData, tempQuadratureGradientsData;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell * d_nCells);
      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(3 * d_nVectors * d_nQuadsPerCell);



      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        {
          const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                    scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
          const char transA = 'N', transB = 'T';

          xgemm(&transA,
                &transB,
                &d_nVectors,
                &d_nDofsPerCell,
                &d_nQuadsPerCell,
                &scalarCoeffAlpha,
                quadratureValues->data() + d_nQuadsPerCell * iCell,
                &d_nVectors,
                d_shapeFunctionData.data(),
                &d_nQuadsPerCell,
                &scalarCoeffBeta,
                cellNodalData.data() + d_nDofsPerCell * iCell,
                &d_nVectors);
          if (quadratureGradients != NULL)
            {
              const unsigned int d_nQuadsPerCellTimesThree =
                d_nQuadsPerCell * 3;
              const unsigned int d_nQuadsPerCellTimesnVectors =
                d_nQuadsPerCell * d_nVectors;
              const unsigned int three = 3;
              xgemm(&transA,
                    &transB,
                    &d_nQuadsPerCellTimesnVectors,
                    &three,
                    &three,
                    &scalarCoeffAlpha,
                    quadratureGradients->data() + d_nQuadsPerCell * 3 * iCell,
                    &d_nQuadsPerCellTimesnVectors,
                    d_inverseJacobianData.data() + 9 * iCell,
                    &three,
                    &scalarCoeffBeta,
                    tempQuadratureGradientsData.data(),
                    &d_nQuadsPerCellTimesnVectors);
              xgemm(&transA,
                    &transB,
                    &d_nVectors,
                    &d_nDofsPerCell,
                    &d_nQuadsPerCellTimesThree,
                    &scalarCoeffAlpha,
                    tempQuadratureGradientsData.data(),
                    &d_nVectors,
                    d_shapeFunctionGradientData.data(),
                    &d_nQuadsPerCellTimesThree,
                    &scalarCoeffAlpha,
                    cellNodalData.data() + d_nDofsPerCell * iCell,
                    &d_nVectors);
            }
        }
      accumulateFromCellNodalDataHostKernel(&cellNodalData,
                                            nodalData,
                                            cellRange,
                                            useMacroCellSubCellOrdering);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      extractToCellNodalDataHostKernel(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                   cellNodalDataPtr,
        std::pair<unsigned int, unsigned int> cellRange,
        bool                                  useMacroCellSubCellOrdering) const
    {
      auto &cellDofIndexToProcessDofIndexMap =
        useMacroCellSubCellOrdering ?
          d_macroCellSubCellDofIndexToProcessDofIndexMap :
          d_cellDofIndexToProcessDofIndexMap;

      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          std::memcpy(
            cellNodalDataPtr + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors,
            nodalData.data() +
              d_nVectors *
                cellDofIndexToProcessDofIndexMap[iCell * d_nDofsPerCell + iDof],
            d_nVectors * sizeof(ValueTypeBasisCoeff));
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      accumulateFromCellNodalDataHostKernel(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                   nodalData,
        std::pair<unsigned int, unsigned int> cellRange,
        bool                                  useMacroCellSubCellOrdering) const
    {
      auto &cellDofIndexToProcessDofIndexMap =
        useMacroCellSubCellOrdering ?
          d_macroCellSubCellDofIndexToProcessDofIndexMap :
          d_cellDofIndexToProcessDofIndexMap;

      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          std::transform(
            cellNodalDataPtr + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors,
            cellNodalDataPtr + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors + d_nVectors,
            nodalData.data() +
              d_nVectors *
                cellDofIndexToProcessDofIndexMap[iCell * d_nDofsPerCell + iDof],
            nodalData.data() +
              d_nVectors *
                cellDofIndexToProcessDofIndexMap[iCell * d_nDofsPerCell + iDof],
            std::plus<ValueTypeBasisCoeff>());
    }

  } // namespace basis
} // namespace dftfe
