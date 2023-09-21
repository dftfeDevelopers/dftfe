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
#include <linearAlgebraOperations.h>
namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        *quadratureValuesCurrentCell;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        *quadratureGradientsCurrentCell;

      for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
        {
          dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
          quadratureValuesCurrentCell  = &(quadratureValues->at(currentCellId));
          quadratureGradientsCurrentCell =
            quadratureGradients ? &(quadratureGradients->at(currentCellId)) :
                                  NULL;
          interpolateKernel(nodalData,
                            quadratureValuesCurrentCell,
                            quadratureGradientsCurrentCell,
                            std::pair<unsigned int, unsigned int>(iCell,
                                                                  iCell + 1));
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureGradients) const
    {
      interpolateKernel(nodalData,
                        quadratureValues,
                        quadratureGradients,
                        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      integrateWithBasis(
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const
    {
      for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
        {
          dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
          const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                            dftfe::utils::MemorySpace::HOST>
            *quadratureValuesCurrentCell =
              &(quadratureValues->at(currentCellId));
          const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                            dftfe::utils::MemorySpace::HOST>
            *quadratureGradientsCurrentCell =
              quadratureGradients ? &(quadratureGradients->at(currentCellId)) :
                                    NULL;
          integrateWithBasisKernel(
            quadratureValuesCurrentCell,
            quadratureGradientsCurrentCell,
            nodalData,
            std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      integrateWithBasis(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const
    {
      integrateWithBasisKernel(quadratureValues,
                               quadratureGradients,
                               nodalData,
                               std::pair<unsigned int, unsigned int>(0,
                                                                     d_nCells));
    }


    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *cellNodalDataPtr) const
    {
      extractToCellNodalDataKernel(
        nodalData,
        cellNodalDataPtr,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      accumulateFromCellNodalData(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const
    {
      accumulateFromCellNodalDataKernel(
        cellNodalDataPtr,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &nodalValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                         quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        cellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell);

      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(
          areAllCellsCartesian ? 0 : (d_nVectors * d_nQuadsPerCell * 3));

      if (quadratureGradients != NULL)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ? 0 : (d_nVectors * d_nQuadsPerCell * 3));


      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        {
          extractToCellNodalDataKernel(
            nodalValues,
            &cellNodalData,
            std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
          const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                    scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
          const char transA = 'N', transB = 'N';

          xgemm(&transA,
                &transB,
                &d_nVectors,
                &d_nQuadsPerCell,
                &d_nDofsPerCell,
                &scalarCoeffAlpha,
                cellNodalData.data() +
                  d_nDofsPerCell * (iCell - cellRange.first) * d_nVectors,
                &d_nVectors,
                d_shapeFunctionData.data(),
                &d_nDofsPerCell,
                &scalarCoeffBeta,
                quadratureValues->data() +
                  d_nQuadsPerCell * (iCell - cellRange.first) * d_nVectors,
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
                    cellNodalData.data() +
                      d_nDofsPerCell * (iCell - cellRange.first) * d_nVectors,
                    &d_nVectors,
                    d_shapeFunctionGradientData.data(),
                    &d_nDofsPerCell,
                    &scalarCoeffBeta,
                    areAllCellsCartesian ? (quadratureGradients->data() +
                                            d_nQuadsPerCell * d_nVectors * 3 *
                                              (iCell - cellRange.first)) :
                                           (tempQuadratureGradientsData.data()),
                    &d_nVectors);
              if (areAllCellsCartesian)
                {
                  const unsigned int d_nQuadsPerCellTimesnVectors =
                    d_nQuadsPerCell * d_nVectors;
                  const unsigned int one = 1;
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    xscal(&d_nQuadsPerCellTimesnVectors,
                          d_inverseJacobianData.data() + 3 * iCell + iDim,
                          quadratureGradients->data() +
                            d_nQuadsPerCell * d_nVectors * 3 *
                              (iCell - cellRange.first) +
                            d_nQuadsPerCell * d_nVectors * iDim,
                          &one);
                }
              else if (areAllCellsAffine)
                {
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
                        quadratureGradients->data() +
                          d_nQuadsPerCell * d_nVectors * 3 *
                            (iCell - cellRange.first),
                        &d_nQuadsPerCellTimesnVectors);
                }
              else
                {
                  const unsigned int three = 3;
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                    xgemm(&transA,
                          &transB,
                          &d_nVectors,
                          &three,
                          &three,
                          &scalarCoeffAlpha,
                          tempQuadratureGradientsData.data() +
                            iQuad * d_nVectors * 3,
                          &d_nVectors,
                          d_inverseJacobianData.data() +
                            9 * d_nQuadsPerCell * iCell + 9 * iQuad,
                          &three,
                          &scalarCoeffBeta,
                          tempQuadratureGradientsDataNonAffine.data() +
                            iQuad * d_nVectors * 3,
                          &d_nVectors);
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      std::memcpy(quadratureGradients->data() +
                                    d_nVectors * 3 * d_nQuadsPerCell *
                                      (iCell - cellRange.first) +
                                    d_nVectors * d_nQuadsPerCell * iDim +
                                    d_nVectors * iQuad,
                                  tempQuadratureGradientsDataNonAffine.data() +
                                    d_nVectors * 3 * iQuad + d_nVectors * iDim,
                                  d_nVectors * sizeof(ValueTypeBasisCoeff));
                }
            }
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      integrateWithBasisKernel(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          *quadratureValues,
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        cellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell * d_nCells);
      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(3 * d_nVectors * d_nQuadsPerCell);

      if (quadratureGradients != NULL)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ? 0 : (3 * d_nVectors * d_nQuadsPerCell));



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
              if (areAllCellsCartesian)
                {
                  const unsigned int d_nQuadsPerCellTimesnVectors =
                    d_nQuadsPerCell * d_nVectors;
                  const unsigned int one = 1;
                  std::memcpy(tempQuadratureGradientsData.data(),
                              quadratureGradients->data() +
                                d_nQuadsPerCell * d_nVectors * 3 * iCell,
                              3 * d_nQuadsPerCellTimesnVectors *
                                sizeof(ValueTypeBasisCoeff));
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    xscal(&d_nQuadsPerCellTimesnVectors,
                          d_inverseJacobianData.data() + 3 * iCell + iDim,
                          tempQuadratureGradientsData.data() +
                            d_nQuadsPerCell * d_nVectors * iDim,
                          &one);
                }
              else if (areAllCellsAffine)
                {
                  const unsigned int d_nQuadsPerCellTimesnVectors =
                    d_nQuadsPerCell * d_nVectors;
                  const unsigned int three = 3;
                  xgemm(&transA,
                        &transB,
                        &d_nQuadsPerCellTimesnVectors,
                        &three,
                        &three,
                        &scalarCoeffAlpha,
                        quadratureGradients->data() +
                          d_nQuadsPerCell * d_nVectors * 3 * iCell,
                        &d_nQuadsPerCellTimesnVectors,
                        d_inverseJacobianData.data() + 9 * iCell,
                        &three,
                        &scalarCoeffBeta,
                        tempQuadratureGradientsData.data(),
                        &d_nQuadsPerCellTimesnVectors);
                }
              else
                {
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      std::memcpy(tempQuadratureGradientsDataNonAffine.data() +
                                    d_nVectors * 3 * iQuad + d_nVectors * iDim,
                                  quadratureGradients->data() +
                                    d_nVectors * 3 * d_nQuadsPerCell * iCell +
                                    d_nVectors * d_nQuadsPerCell * iDim +
                                    d_nVectors * iQuad,
                                  d_nVectors * sizeof(ValueTypeBasisCoeff));
                  const unsigned int three = 3;
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                    xgemm(&transA,
                          &transB,
                          &d_nVectors,
                          &three,
                          &three,
                          &scalarCoeffAlpha,
                          tempQuadratureGradientsDataNonAffine.data() +
                            d_nVectors * 3 * iQuad,
                          &d_nVectors,
                          d_inverseJacobianData.data() +
                            9 * d_nQuadsPerCell * iCell + 9 * iQuad,
                          &three,
                          &scalarCoeffBeta,
                          tempQuadratureGradientsData.data() +
                            d_nVectors * 3 * iQuad,
                          &d_nVectors);
                }
              const unsigned int d_nQuadsPerCellTimesThree =
                d_nQuadsPerCell * 3;
              xgemm(&transA,
                    &transB,
                    &d_nVectors,
                    &d_nQuadsPerCellTimesThree,
                    &d_nDofsPerCell,
                    &scalarCoeffAlpha,
                    tempQuadratureGradientsData.data(),
                    &d_nVectors,
                    d_shapeFunctionGradientData.data(),
                    &d_nDofsPerCell,
                    &scalarCoeffBeta,
                    cellNodalData.data() + d_nDofsPerCell * iCell,
                    &d_nVectors);
            }
          accumulateFromCellNodalDataKernel(
            &cellNodalData,
            nodalData,
            std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::HOST>
          *                                         cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          {
            std::memcpy(cellNodalDataPtr->data() +
                          (iCell - cellRange.first) * d_nVectors *
                            d_nDofsPerCell +
                          iDof * d_nVectors,
                        nodalData.data() +
                          d_flattenedCellDofIndexToProcessDofIndexMap
                            [iCell * d_nDofsPerCell + iDof],
                        d_nVectors * sizeof(ValueTypeBasisCoeff));
          }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      accumulateFromCellNodalDataKernel(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          std::transform(
            cellNodalDataPtr->data() + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors,
            cellNodalDataPtr->data() + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors + d_nVectors,
            nodalData.data() + d_flattenedCellDofIndexToProcessDofIndexMap
                                 [iCell * d_nDofsPerCell + iDof],
            nodalData.data() + d_flattenedCellDofIndexToProcessDofIndexMap
                                 [iCell * d_nDofsPerCell + iDof],
            std::plus<ValueTypeBasisCoeff>());
    }
  } // namespace basis
} // namespace dftfe
