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
// @author Nikhil Kodali
//

#include <dftfe/config.h>
#include <dftfe/KohnShamDFTBaseOperator.h>
#include <dftfe/KohnShamDFTOperatorKernels.h>
#include <dftfe/ExcDFTPlusU.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
#if defined(DFTFE_WITH_DEVICE)
#  include <dftfe/DeviceAPICalls.h>
#endif

namespace dftfe
{
  namespace internal
  {
    template <>
    void
    computeCellHamiltonianMatrixNonCollinearFromBlocks(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         nDofsPerCell,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tempHamMatrixRealBlock,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tempHamMatrixImagBlock,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tempHamMatrixBZBlockNonCollin,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tempHamMatrixBYBlockNonCollin,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &tempHamMatrixBXBlockNonCollin,
      dftfe::utils::MemoryStorage<std::complex<double>,
                                  dftfe::utils::MemorySpace::HOST>
        &cellHamiltonianMatrix)
    {
      for (dftfe::uInt iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (dftfe::uInt iDoF = 0; iDoF < nDofsPerCell; ++iDoF)
          for (dftfe::uInt jDoF = 0; jDoF < nDofsPerCell; ++jDoF)
            {
              const double H_realIJ =
                tempHamMatrixRealBlock[jDoF + nDofsPerCell * iDoF +
                                       (iCell - cellRange.first) *
                                         nDofsPerCell * nDofsPerCell];
              const double H_imagIJ =
                tempHamMatrixImagBlock[jDoF + nDofsPerCell * iDoF +
                                       (iCell - cellRange.first) *
                                         nDofsPerCell * nDofsPerCell];
              const double H_bzIJ =
                tempHamMatrixBZBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                              (iCell - cellRange.first) *
                                                nDofsPerCell * nDofsPerCell];
              const double H_byIJ =
                tempHamMatrixBYBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                              (iCell - cellRange.first) *
                                                nDofsPerCell * nDofsPerCell];
              const double H_bxIJ =
                tempHamMatrixBXBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                              (iCell - cellRange.first) *
                                                nDofsPerCell * nDofsPerCell];
              cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                    2 * nDofsPerCell * (2 * iDoF + 1) +
                                    2 * jDoF + 1] =
                std::complex<double>(H_realIJ - H_bzIJ, H_imagIJ);
              cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                    2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF] =
                std::complex<double>(H_realIJ + H_bzIJ, H_imagIJ);
              cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                    2 * nDofsPerCell * (2 * iDoF + 1) +
                                    2 * jDoF] =
                std::complex<double>(H_bxIJ, H_byIJ);
              cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                    2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF +
                                    1] = std::complex<double>(H_bxIJ, -H_byIJ);
            }
    }
    template <>
    void
    computeVeffJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &VeffJxW)
    {
      dftfe::uInt startCell = cellRange.first;
      dftfe::uInt endCell   = cellRange.second;
      dftfe::uInt iCell     = 0;
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *tempPhi =
            phiVector.data() + cellIndex * numQuadsPerCell;
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
            {
              VeffJxW[cellIndex * numQuadsPerCell + iQuad] =
                (tempPhi[iQuad] + pdecVector[iCell * numQuadsPerCell + iQuad] +
                 pdexVector[iCell * numQuadsPerCell + iQuad]) *
                cellJxWPtr[iQuad];
            }
          iCell++;
        }
    }
    template <>
    void
    computeVeffBeffJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVectorSpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVectorSpinDown,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVectorSpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVectorSpinDown,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &magAxis,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &VeffJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &BeffxJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &BeffyJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &BeffzJxW)
    {
      dftfe::uInt startCell = cellRange.first;
      dftfe::uInt endCell   = cellRange.second;
      dftfe::uInt iCell     = 0;
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *tempPhi =
            phiVector.data() + cellIndex * numQuadsPerCell;
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
            {
              VeffJxW[cellIndex * numQuadsPerCell + iQuad] =
                (tempPhi[iQuad] +
                 0.5 * (pdecVectorSpinUp[iCell * numQuadsPerCell + iQuad] +
                        pdecVectorSpinDown[iCell * numQuadsPerCell + iQuad] +
                        pdexVectorSpinUp[iCell * numQuadsPerCell + iQuad] +
                        pdexVectorSpinDown[iCell * numQuadsPerCell + iQuad])) *
                cellJxWPtr[iQuad];
              const double temp =
                0.5 *
                (pdecVectorSpinUp[iCell * numQuadsPerCell + iQuad] -
                 pdecVectorSpinDown[iCell * numQuadsPerCell + iQuad] +
                 pdexVectorSpinUp[iCell * numQuadsPerCell + iQuad] -
                 pdexVectorSpinDown[iCell * numQuadsPerCell + iQuad]) *
                cellJxWPtr[iQuad];
              BeffxJxW[cellIndex * numQuadsPerCell + iQuad] =
                temp * magAxis[iCell * numQuadsPerCell * 3 + 3 * iQuad + 0];
              BeffyJxW[cellIndex * numQuadsPerCell + iQuad] =
                temp * magAxis[iCell * numQuadsPerCell * 3 + 3 * iQuad + 1];
              BeffzJxW[cellIndex * numQuadsPerCell + iQuad] =
                temp * magAxis[iCell * numQuadsPerCell * 3 + 3 * iQuad + 2];
            }
          iCell++;
        }
    }
    template <>
    void
    computeInvJacderExcWithSigmaTimesGradRhoJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          spinIndex,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacobianEntries,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradientRhoSpinIndex,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradientRhoOtherSpinIndex,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacderExcWithSigmaTimesGradRhoJxW)
    {
      dftfe::uInt startCell = cellRange.first;
      dftfe::uInt endCell   = cellRange.second;
      dftfe::uInt iCell     = 0;
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          if (cellsTypeFlag != 2)
            {
              {
                for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                  {
                    const double *inverseJacobiansQuadPtr =
                      invJacobianEntries.data() +
                      (cellsTypeFlag == 0 ?
                         cellIndex * 9 * numQuadsPerCell + iQuad * 9 :
                         cellIndex * 9);
                    const double *gradDensityQuadPtr =
                      gradientRhoSpinIndex.data() +
                      iCell * 3 * numQuadsPerCell + iQuad * 3;
                    const double *gradDensityOtherQuadPtr =
                      gradientRhoOtherSpinIndex.data() +
                      iCell * 3 * numQuadsPerCell + iQuad * 3;
                    const double term =
                      (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 +
                                  2 * spinIndex] +
                       pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 +
                                  2 * spinIndex]) *
                      cellJxWPtr[iQuad];
                    const double termOff =
                      (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1] +
                       pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 +
                                  1]) *
                      cellJxWPtr[iQuad];
                    for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                        {
                          invJacderExcWithSigmaTimesGradRhoJxW
                            [3 * cellIndex * numQuadsPerCell + iQuad * 3 +
                             iDim] += inverseJacobiansQuadPtr[3 * jDim + iDim] *
                                      (2 * gradDensityQuadPtr[jDim] * term +
                                       gradDensityOtherQuadPtr[jDim] * termOff);
                        } // iDim
                  }       // iQuad
              }           // iQuad
            }
          else if (cellsTypeFlag == 2)
            {
              const double *inverseJacobiansQuadPtr =
                invJacobianEntries.data() + cellIndex * 3;
              for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                {
                  const double *gradDensityQuadPtr =
                    gradientRhoSpinIndex.data() + iCell * 3 * numQuadsPerCell +
                    iQuad * 3;
                  const double *gradDensityOtherQuadPtr =
                    gradientRhoOtherSpinIndex.data() +
                    iCell * 3 * numQuadsPerCell + iQuad * 3;
                  const double term = (pdecVector[iCell * 3 * numQuadsPerCell +
                                                  iQuad * 3 + 2 * spinIndex] +
                                       pdexVector[iCell * 3 * numQuadsPerCell +
                                                  iQuad * 3 + 2 * spinIndex]) *
                                      cellJxWPtr[iQuad];
                  const double termOff =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1]) *
                    cellJxWPtr[iQuad];
                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                    {
                      invJacderExcWithSigmaTimesGradRhoJxW[3 * cellIndex *
                                                             numQuadsPerCell +
                                                           iQuad * 3 + iDim] =
                        inverseJacobiansQuadPtr[iDim] *
                        (2 * gradDensityQuadPtr[iDim] * term +
                         gradDensityOtherQuadPtr[iDim] * termOff);
                    } // iDim
                }     // iQuad
            }
          iCell++;
        } // cellIndex
    }
    template <>
    void
    computeInvJacderExcWithSigmaTimesGradRhoMagJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          spinIndex,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &magAxis,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacobianEntries,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradientRhoSpinUp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradientRhoSpinDown,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacderExcWithSigmaTimesGradRhoJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost)
    {
      dftfe::uInt startCell = cellRange.first;
      dftfe::uInt endCell   = cellRange.second;
      dftfe::uInt iCell     = 0;
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          if (cellsTypeFlag != 2)
            {
              for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                {
                  const double *inverseJacobiansQuadPtr =
                    invJacobianEntries.data() +
                    (cellsTypeFlag == 0 ?
                       cellIndex * 9 * numQuadsPerCell + iQuad * 9 :
                       cellIndex * 9);
                  const double *magAxisQuadPtr =
                    magAxis.data() + iCell * 3 * numQuadsPerCell + iQuad * 3;
                  const double *gradDensitySpinUpQuadPtr =
                    gradientRhoSpinUp.data() + iCell * 3 * numQuadsPerCell +
                    iQuad * 3;
                  const double *gradDensitySpinDownQuadPtr =
                    gradientRhoSpinDown.data() + iCell * 3 * numQuadsPerCell +
                    iQuad * 3;
                  const double termSpinUp =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3]) *
                    cellJxWPtr[iQuad];
                  const double termSpinDown =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 2] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 2]) *
                    cellJxWPtr[iQuad];
                  const double termSpinCross =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1]) *
                    cellJxWPtr[iQuad];
                  for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                    for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                      {
                        invJacderExcWithSigmaTimesGradRhoJxW[3 * cellIndex *
                                                               numQuadsPerCell +
                                                             iQuad * 3 +
                                                             iDim] +=
                          inverseJacobiansQuadPtr[3 * jDim + iDim] *
                          (gradDensitySpinUpQuadPtr[jDim] * termSpinUp +
                           gradDensitySpinDownQuadPtr[jDim] * termSpinDown +
                           0.5 *
                             (gradDensitySpinUpQuadPtr[jDim] +
                              gradDensitySpinDownQuadPtr[jDim]) *
                             termSpinCross);
                      } // iDim
                  for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                    {
                      const double term =
                        (gradDensitySpinUpQuadPtr[jDim] * termSpinUp -
                         gradDensitySpinDownQuadPtr[jDim] * termSpinDown +
                         0.5 *
                           (gradDensitySpinDownQuadPtr[jDim] -
                            gradDensitySpinUpQuadPtr[jDim]) *
                           termSpinCross);
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        {
                          const double termJac =
                            inverseJacobiansQuadPtr[3 * jDim + iDim] * term;
                          invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost
                            [3 * cellIndex * numQuadsPerCell + iQuad * 3 +
                             iDim] += termJac * magAxisQuadPtr[0];
                          invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost
                            [3 * cellIndex * numQuadsPerCell + iQuad * 3 +
                             iDim] += termJac * magAxisQuadPtr[1];
                          invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost
                            [3 * cellIndex * numQuadsPerCell + iQuad * 3 +
                             iDim] += termJac * magAxisQuadPtr[2];
                        }
                    }
                } // iQuad
            }
          else if (cellsTypeFlag == 2)
            {
              const double *inverseJacobiansQuadPtr =
                invJacobianEntries.data() + cellIndex * 3;
              for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                {
                  const double *magAxisQuadPtr =
                    magAxis.data() + iCell * 3 * numQuadsPerCell + iQuad * 3;
                  const double *gradDensitySpinUpQuadPtr =
                    gradientRhoSpinUp.data() + iCell * 3 * numQuadsPerCell +
                    iQuad * 3;
                  const double *gradDensitySpinDownQuadPtr =
                    gradientRhoSpinDown.data() + iCell * 3 * numQuadsPerCell +
                    iQuad * 3;
                  const double termSpinUp =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3]) *
                    cellJxWPtr[iQuad];
                  const double termSpinDown =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 2] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 2]) *
                    cellJxWPtr[iQuad];
                  const double termSpinCross =
                    (pdecVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1] +
                     pdexVector[iCell * 3 * numQuadsPerCell + iQuad * 3 + 1]) *
                    cellJxWPtr[iQuad];
                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                    {
                      const double termMinus =
                        (gradDensitySpinUpQuadPtr[iDim] * termSpinUp -
                         gradDensitySpinDownQuadPtr[iDim] * termSpinDown +
                         0.5 *
                           (gradDensitySpinDownQuadPtr[iDim] -
                            gradDensitySpinUpQuadPtr[iDim]) *
                           termSpinCross);
                      const double termPlus =
                        (gradDensitySpinUpQuadPtr[iDim] * termSpinUp +
                         gradDensitySpinDownQuadPtr[iDim] * termSpinDown +
                         0.5 *
                           (gradDensitySpinDownQuadPtr[iDim] +
                            gradDensitySpinUpQuadPtr[iDim]) *
                           termSpinCross);
                      const double termMinusJac =
                        inverseJacobiansQuadPtr[iDim] * termMinus;
                      const double termPlusJac =
                        inverseJacobiansQuadPtr[iDim] * termPlus;
                      invJacderExcWithSigmaTimesGradRhoJxW[3 * cellIndex *
                                                             numQuadsPerCell +
                                                           iQuad * 3 + iDim] =
                        termPlusJac;
                      invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost
                        [3 * cellIndex * numQuadsPerCell + iQuad * 3 + iDim] =
                          termMinusJac * magAxisQuadPtr[0];
                      invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost
                        [3 * cellIndex * numQuadsPerCell + iQuad * 3 + iDim] =
                          termMinusJac * magAxisQuadPtr[1];
                      invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost
                        [3 * cellIndex * numQuadsPerCell + iQuad * 3 + iDim] =
                          termMinusJac * magAxisQuadPtr[2];
                    } // iDim
                }     // iQuad
            }
          iCell++;
        } // cellIndex
    }

    template <>
    void
    computeHalfInvJacinvJacderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacinvJacderExcWithTauJxW)
    {
      dftfe::uInt startCell = cellRange.first;
      dftfe::uInt endCell   = cellRange.second;
      dftfe::uInt iCell     = 0;
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          if (cellsTypeFlag != 2)
            {
              {
                for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                  {
                    const double *inverseJacobiansQuadPtr =
                      invJacobianEntries.data() +
                      (cellsTypeFlag == 0 ?
                         cellIndex * 9 * numQuadsPerCell + iQuad * 9 :
                         cellIndex * 9);
                    const double termTau =
                      0.5 *
                      (pdecVector[iCell * numQuadsPerCell + iQuad] +
                       pdexVector[iCell * numQuadsPerCell + iQuad]) *
                      cellJxWPtr[iQuad];

                    double *jacobianFactorForTauPtr =
                      invJacinvJacderExcWithTauJxW.data() +
                      cellIndex * 9 * numQuadsPerCell + iQuad * 9;

                    for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                              {
                                jacobianFactorForTauPtr[3 * jDim + iDim] +=
                                  inverseJacobiansQuadPtr[3 * kDim + iDim] *
                                  inverseJacobiansQuadPtr[3 * kDim + jDim];
                              }
                            jacobianFactorForTauPtr[3 * jDim + iDim] *= termTau;
                          }
                      }
                  } // iQuad
              }     // iQuad
            }
          else if (cellsTypeFlag == 2)
            {
              const double *inverseJacobiansQuadPtr =
                invJacobianEntries.data() + cellIndex * 3;
              for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                {
                  const double termTau =
                    0.5 *
                    (pdecVector[iCell * numQuadsPerCell + iQuad] +
                     pdexVector[iCell * numQuadsPerCell + iQuad]) *
                    cellJxWPtr[iQuad];
                  double *jacobianFactorForTauPtr =
                    invJacinvJacderExcWithTauJxW.data() +
                    cellIndex * 9 * numQuadsPerCell + iQuad * 9;
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    {
                      jacobianFactorForTauPtr[3 * iDim + iDim] +=
                        inverseJacobiansQuadPtr[iDim] *
                        inverseJacobiansQuadPtr[iDim] * termTau;
                    }
                } // iQuad
            }
          iCell++;
        } // cellIndex
    }

    template <>
    void
    computeKPointDependenderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::uInt                         offset,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &kPointCoordinate,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &halfKSquareTimesDerExcwithTauJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &invJacKpointTimesderExcwithTauJxW)
    {
      dftfe::uInt  startCell = cellRange.first;
      dftfe::uInt  endCell   = cellRange.second;
      dftfe::uInt  iCell     = 0;
      const double kSquareTimesHalf =
        0.5 * (kPointCoordinate[0] * kPointCoordinate[0] +
               kPointCoordinate[1] * kPointCoordinate[1] +
               kPointCoordinate[2] * kPointCoordinate[2]);
      for (dftfe::uInt cellIndex = startCell; cellIndex < endCell; cellIndex++)
        {
          const double *cellJxWPtr =
            jxwVector.data() + cellIndex * numQuadsPerCell;
          for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
            {
              halfKSquareTimesDerExcwithTauJxW[cellIndex * numQuadsPerCell +
                                               iQuad + offset] =
                (pdecVector[iCell * numQuadsPerCell + iQuad] +
                 pdexVector[iCell * numQuadsPerCell + iQuad]) *
                kSquareTimesHalf * cellJxWPtr[iQuad];
            }
          if (cellsTypeFlag != 2)
            {
              {
                for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                  {
                    const double *inverseJacobiansQuadPtr =
                      invJacobianEntries.data() +
                      (cellsTypeFlag == 0 ?
                         cellIndex * 9 * numQuadsPerCell + iQuad * 9 :
                         cellIndex * 9);
                    const double termTau =
                      (pdecVector[iCell * numQuadsPerCell + iQuad] +
                       pdexVector[iCell * numQuadsPerCell + iQuad]) *
                      cellJxWPtr[iQuad];



                    for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            invJacKpointTimesderExcwithTauJxW
                              [cellIndex * numQuadsPerCell * 3 + iQuad * 3 +
                               iDim + offset * 3] +=
                              -0.5 * inverseJacobiansQuadPtr[3 * jDim + iDim] *
                              kPointCoordinate[jDim] * termTau;
                          }
                      }
                  } // iQuad
              }     // iQuad
            }
          else if (cellsTypeFlag == 2)
            {
              const double *inverseJacobiansQuadPtr =
                invJacobianEntries.data() + cellIndex * 3;
              for (dftfe::uInt iQuad = 0; iQuad < numQuadsPerCell; ++iQuad)
                {
                  const double termTau =
                    (pdecVector[iCell * numQuadsPerCell + iQuad] +
                     pdexVector[iCell * numQuadsPerCell + iQuad]) *
                    cellJxWPtr[iQuad];
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    {
                      invJacKpointTimesderExcwithTauJxW[cellIndex *
                                                          numQuadsPerCell * 3 +
                                                        iQuad * 3 + iDim +
                                                        offset * 3] =
                        -0.5 * inverseJacobiansQuadPtr[iDim] *
                        kPointCoordinate[iDim] * termTau;
                    }
                } // iQuad
            }
          iCell++;
        } // cellIndex
    }
  }; // namespace internal


  //
  // constructor
  //
  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamDFTBaseOperator<memorySpace>::KohnShamDFTBaseOperator(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      basisOperationsPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrHost,
    std::shared_ptr<
      dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                             pseudopotentialClassPtr,
    std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    dftParameters                           *dftParamsPtr,
    const dftfe::uInt                        densityQuadratureID,
    const dftfe::uInt                        lpspQuadratureID,
    const dftfe::uInt                        feOrderPlusOneQuadratureID,
    const MPI_Comm                          &mpi_comm_parent,
    const MPI_Comm                          &mpi_comm_domain)
    : d_kPointIndex(0)
    , d_spinIndex(0)
    , d_HamiltonianIndex(0)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_BLASWrapperPtrHost(BLASWrapperPtrHost)
    , d_basisOperationsPtr(basisOperationsPtr)
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_pseudopotentialClassPtr(pseudopotentialClassPtr)
    , d_excManagerPtr(excManagerPtr)
    , d_dftParamsPtr(dftParamsPtr)
    , d_densityQuadratureID(densityQuadratureID)
    , d_lpspQuadratureID(lpspQuadratureID)
    , d_feOrderPlusOneQuadratureID(feOrderPlusOneQuadratureID)
    , d_isExternalPotCorrHamiltonianComputed(false)
    , d_mpiCommParent(mpi_comm_parent)
    , d_mpiCommDomain(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
  {
    d_nOMPThreads = 1;
#ifdef _OPENMP
    if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
      {
        try
          {
            d_nOMPThreads = std::stoi(std::string(penv));
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                std::string(
                  "When specifying the <DFTFE_NUM_THREADS> environment "
                  "variable, it needs to be something that can be interpreted "
                  "as an integer. The text you have in the environment "
                  "variable is <") +
                penv + ">"));
          }

        AssertThrow(d_nOMPThreads > 0,
                    dealii::ExcMessage(
                      "When specifying the <DFTFE_NUM_THREADS> environment "
                      "variable, it needs to be a positive number."));
      }
#endif
    d_nOMPThreads =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 1 : d_nOMPThreads;
    if (d_dftParamsPtr->isPseudopotential)
      d_pseudopotentialNonLocalOperator =
        pseudopotentialClassPtr->getNonLocalOperator();

    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      d_pseudopotentialNonLocalOperatorSinglePrec =
        pseudopotentialClassPtr->getNonLocalOperatorSinglePrec();
    d_cellsBlockSizeHamiltonianConstruction =
      memorySpace == dftfe::utils::MemorySpace::HOST ? 1 : 50;
    d_cellsBlockSizeHX =
      memorySpace == dftfe::utils::MemorySpace::HOST ?
        1 :
        (d_dftParamsPtr->memOptMode ? 50 : d_basisOperationsPtr->nCells());
    d_numVectorsInternal = 0;
    d_useHubbard         = false;
    if (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
        ExcFamilyType::DFTPlusU)
      {
        d_useHubbard = true;
        std::shared_ptr<ExcDFTPlusU<dataTypes::number, memorySpace>>
          excHubbPtr = std::dynamic_pointer_cast<
            ExcDFTPlusU<dataTypes::number, memorySpace>>(
            d_excManagerPtr->getSSDSharedObj());

        d_hubbardClassPtr = excHubbPtr->getHubbardClass();
      }
  }

  //
  // initialize KohnShamDFTBaseOperator object
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::init(
    const std::vector<double> &kPointCoordinates,
    const std::vector<double> &kPointWeights)
  {
    computing_timer.enter_subsection("KohnShamDFTBaseOperator setup");
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr =
      std::make_shared<dftUtils::constraintMatrixInfo<memorySpace>>(
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr
      ->initializeScaledConstraints(
        d_basisOperationsPtr->inverseSqrtMassVectorBasisData());
    inverseMassVectorScaledConstraintsNoneDataInfoPtr =
      std::make_shared<dftUtils::constraintMatrixInfo<memorySpace>>(
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
    inverseMassVectorScaledConstraintsNoneDataInfoPtr
      ->initializeScaledConstraints(
        d_basisOperationsPtr->inverseMassVectorBasisData());
    d_kPointCoordinates = kPointCoordinates;
    d_kPointWeights     = kPointWeights;
    d_invJacKPointTimesJxW.resize(d_kPointWeights.size());
    d_halfKSquareTimesDerExcwithTauJxW.resize(d_kPointWeights.size());
    d_derExcwithTauTimesinvJacKpointTimesJxW.resize(d_kPointWeights.size());
    d_cellHamiltonianMatrix.resize(
      d_dftParamsPtr->memOptMode ?
        1 :
        (d_kPointWeights.size() * (d_dftParamsPtr->spinPolarized + 1)));
    d_cellHamiltonianMatrixSinglePrec.resize(
      d_dftParamsPtr->useSinglePrecCheby ? d_cellHamiltonianMatrix.size() : 0);

    const dftfe::uInt nCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    tempHamMatrixRealBlock.resize(nDofsPerCell * nDofsPerCell *
                                  d_cellsBlockSizeHamiltonianConstruction);
    if (d_dftParamsPtr->noncolin)
      {
        tempHamMatrixBXBlockNonCollin.resize(
          nDofsPerCell * nDofsPerCell *
          d_cellsBlockSizeHamiltonianConstruction);
        tempHamMatrixBYBlockNonCollin.resize(
          nDofsPerCell * nDofsPerCell *
          d_cellsBlockSizeHamiltonianConstruction);
        tempHamMatrixBZBlockNonCollin.resize(
          nDofsPerCell * nDofsPerCell *
          d_cellsBlockSizeHamiltonianConstruction);
      }
    if constexpr (std::is_same<dataTypes::number, std::complex<double>>::value)
      tempHamMatrixImagBlock.resize(nDofsPerCell * nDofsPerCell *
                                    d_cellsBlockSizeHamiltonianConstruction);
    for (dftfe::uInt iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrix.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrix[iHamiltonian].resize(
        nDofsPerCell * nDofsPerCell * nCells *
        (d_dftParamsPtr->noncolin ? 4 : 1));
    for (dftfe::uInt iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrixSinglePrec.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrixSinglePrec[iHamiltonian].resize(
        nDofsPerCell * nDofsPerCell * nCells *
        (d_dftParamsPtr->noncolin ? 4 : 1));

    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID, false);
    const dftfe::uInt numberQuadraturePointsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      for (dftfe::uInt kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
#if defined(DFTFE_WITH_DEVICE)
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            d_invJacKPointTimesJxWHost;
#else
          auto &d_invJacKPointTimesJxWHost =
            d_invJacKPointTimesJxW[kPointIndex];
#endif
          d_invJacKPointTimesJxWHost.resize(nCells *
                                              numberQuadraturePointsPerCell * 3,
                                            0.0);
          for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
            {
              auto cellJxWPtr =
                d_basisOperationsPtrHost->JxWBasisData().data() +
                iCell * numberQuadraturePointsPerCell;
              const double *kPointCoordinatesPtr =
                kPointCoordinates.data() + 3 * kPointIndex;

              if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                {
                  for (dftfe::uInt iQuad = 0;
                       iQuad < numberQuadraturePointsPerCell;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePointsPerCell * 9 +
                             iQuad * 9 :
                           iCell * 9);
                      for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          d_invJacKPointTimesJxWHost
                            [iCell * numberQuadraturePointsPerCell * 3 +
                             iQuad * 3 + iDim] +=
                            -inverseJacobiansQuadPtr[3 * jDim + iDim] *
                            kPointCoordinatesPtr[jDim] * cellJxWPtr[iQuad];
                    }
                }
              else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                {
                  for (dftfe::uInt iQuad = 0;
                       iQuad < numberQuadraturePointsPerCell;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                        d_invJacKPointTimesJxWHost
                          [iCell * numberQuadraturePointsPerCell * 3 +
                           iQuad * 3 + iDim] =
                            -inverseJacobiansQuadPtr[iDim] *
                            kPointCoordinatesPtr[iDim] * cellJxWPtr[iQuad];
                    }
                }
            }
#if defined(DFTFE_WITH_DEVICE)
          d_invJacKPointTimesJxW[kPointIndex].resize(
            d_invJacKPointTimesJxWHost.size());
          d_invJacKPointTimesJxW[kPointIndex].copyFrom(
            d_invJacKPointTimesJxWHost);
#endif
        }
    computing_timer.leave_subsection("KohnShamDFTBaseOperator setup");
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::resetExtPotHamFlag()
  {
    d_isExternalPotCorrHamiltonianComputed = false;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::resetKohnShamOp()
  {
    resetExtPotHamFlag();
    if (d_useHubbard)
      {
        std::shared_ptr<ExcDFTPlusU<dataTypes::number, memorySpace>>
          excHubbPtr = std::dynamic_pointer_cast<
            ExcDFTPlusU<dataTypes::number, memorySpace>>(
            d_excManagerPtr->getSSDSharedObj());

        d_hubbardClassPtr = excHubbPtr->getHubbardClass();

        d_hubbardClassPtr->initialiseFlattenedDataStructure(
          d_numVectorsInternal);
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::computeVEff(
    std::shared_ptr<AuxDensityMatrix<memorySpace>> auxDensityXCRepresentation,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     &phiValues,
    const dftfe::uInt spinIndex)
  {
    bool isIntegrationByPartsGradDensityDependenceVxc =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);
    const bool isGGA = isIntegrationByPartsGradDensityDependenceVxc;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const dftfe::uInt totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const dftfe::uInt nCellsPerBatch =
      (memorySpace == dftfe::utils::MemorySpace::HOST) ?
        1 :
        (d_dftParamsPtr->useLibXCForXCEvaluation ? 1 : 50);
    const dftfe::uInt numberQuadraturePointsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffxJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffyJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffzJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesGradRhoJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacinvJacderExcWithTauJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost;
#else
    auto &d_VeffJxWHost  = d_VeffJxW;
    auto &d_BeffxJxWHost = d_BeffxJxW;
    auto &d_BeffyJxWHost = d_BeffyJxW;
    auto &d_BeffzJxWHost = d_BeffzJxW;
    auto &d_invJacderExcWithSigmaTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesGradRhoJxW;
    auto &d_invJacinvJacderExcWithTauJxWHost = d_invJacinvJacderExcWithTauJxW;
    auto &d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxW;
    auto &d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxW;
    auto &d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxW;
#endif
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePointsPerCell,
                         0.0);
    d_BeffxJxWHost.resize(d_dftParamsPtr->noncolin ?
                            totalLocallyOwnedCells *
                              numberQuadraturePointsPerCell :
                            0,
                          0.0);
    d_BeffyJxWHost.resize(d_dftParamsPtr->noncolin ?
                            totalLocallyOwnedCells *
                              numberQuadraturePointsPerCell :
                            0,
                          0.0);
    d_BeffzJxWHost.resize(d_dftParamsPtr->noncolin ?
                            totalLocallyOwnedCells *
                              numberQuadraturePointsPerCell :
                            0,
                          0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffGGA(isGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell :
                        0,
                0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffxGGA(isGGA && d_dftParamsPtr->noncolin ?
                   totalLocallyOwnedCells * numberQuadraturePointsPerCell :
                   0,
                 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffyGGA(isGGA && d_dftParamsPtr->noncolin ?
                   totalLocallyOwnedCells * numberQuadraturePointsPerCell :
                   0,
                 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_BeffzGGA(isGGA && d_dftParamsPtr->noncolin ?
                   totalLocallyOwnedCells * numberQuadraturePointsPerCell :
                   0,
                 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.clear();
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(
      isGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 : 0,
      0.0);

    d_invJacinvJacderExcWithTauJxWHost.clear();
    d_invJacinvJacderExcWithTauJxWHost.resize(
      isTauMGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell * 9 :
                  0,
      0.0);
    d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost.resize(
      d_dftParamsPtr->noncolin && isGGA ?
        totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 :
        0,
      0.0);
    d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost.resize(
      d_dftParamsPtr->noncolin && isGGA ?
        totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 :
        0,
      0.0);
    d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost.resize(
      d_dftParamsPtr->noncolin && isGGA ?
        totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 :
        0,
      0.0);

    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      xDataOut;
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      cDataOut;


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdexDensitySpinUp =
        xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdexDensitySpinDown =
        xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdecDensitySpinUp =
        cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdecDensitySpinDown =
        cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    pdexDensitySpinUp.resize(numberQuadraturePointsPerCell * nCellsPerBatch,
                             0.0);
    pdecDensitySpinUp.resize(numberQuadraturePointsPerCell * nCellsPerBatch,
                             0.0);
    pdexDensitySpinDown.resize(numberQuadraturePointsPerCell * nCellsPerBatch,
                               0.0);
    pdecDensitySpinDown.resize(numberQuadraturePointsPerCell * nCellsPerBatch,
                               0.0);

    if (isGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            3 * numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            3 * numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
      }
    if (isTauMGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
        xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            numberQuadraturePointsPerCell * nCellsPerBatch, 0.0);
      }

#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_halfKSquareTimesDerExcwithTauJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_derExcwithTauTimesinvJacKpointTimesJxWHost;
    d_halfKSquareTimesDerExcwithTauJxWHost.clear();
    d_halfKSquareTimesDerExcwithTauJxWHost.resize(
      isTauMGGA ? d_kPointWeights.size() * totalLocallyOwnedCells *
                    numberQuadraturePointsPerCell :
                  0,
      0.0);
    d_derExcwithTauTimesinvJacKpointTimesJxWHost.clear();
    d_derExcwithTauTimesinvJacKpointTimesJxWHost.resize(
      isTauMGGA ? d_kPointWeights.size() * totalLocallyOwnedCells *
                    numberQuadraturePointsPerCell * 3 :
                  0,
      0.0);
#else
    for (dftfe::uInt kPointIndex = 0; kPointIndex < d_kPointWeights.size();
         kPointIndex++)
      {
        auto &d_halfKSquareTimesDerExcwithTauJxWHost =
          d_halfKSquareTimesDerExcwithTauJxW[kPointIndex];

        auto &d_derExcwithTauTimesinvJacKpointTimesJxWHost =
          d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex];
        d_halfKSquareTimesDerExcwithTauJxWHost.clear();
        d_halfKSquareTimesDerExcwithTauJxWHost.resize(
          isTauMGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell :
                      0,
          0.0);
        d_derExcwithTauTimesinvJacKpointTimesJxWHost.clear();
        d_derExcwithTauTimesinvJacKpointTimesJxWHost.resize(
          isTauMGGA ?
            totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 :
            0,
          0.0);
      }
#endif


    auto quadPointsAll = d_basisOperationsPtrHost->quadPoints();

    auto quadWeightsAll = d_basisOperationsPtrHost->JxW();

    for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
         iCell += nCellsPerBatch)
      {
        int numCells = std::min(totalLocallyOwnedCells - iCell, nCellsPerBatch);
        d_excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCRepresentation,
          std::make_pair<dftfe::uInt, dftfe::uInt>(
            iCell * numberQuadraturePointsPerCell,
            (iCell + numCells) * numberQuadraturePointsPerCell),
          xDataOut,
          cDataOut);

        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>
          &pdexDensitySpinIndex =
            spinIndex == 0 ? pdexDensitySpinUp : pdexDensitySpinDown;
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>
          &pdecDensitySpinIndex =
            spinIndex == 0 ? pdecDensitySpinUp : pdecDensitySpinDown;

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexSigma;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecSigma;
        if (isGGA)
          {
            pdexSigma = xDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            pdecSigma = cDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexTauSpinIndex;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecTauSpinIndex;
        if (isTauMGGA)
          {
            pdexTauSpinIndex =
              spinIndex == 0 ?
                xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown];

            pdecTauSpinIndex =
              spinIndex == 0 ?
                cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown];
          }

        std::unordered_map<
          DensityDescriptorDataAttributes,
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          densityData;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &densitySpinUp =
            densityData[DensityDescriptorDataAttributes::valuesSpinUp];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &densitySpinDown =
            densityData[DensityDescriptorDataAttributes::valuesSpinDown];
        if (d_dftParamsPtr->noncolin)
          densityData[DensityDescriptorDataAttributes::magAxisValues];
        if (isGGA)
          {
            densityData[DensityDescriptorDataAttributes::gradValuesSpinUp];
            densityData[DensityDescriptorDataAttributes::gradValuesSpinDown];
          }

        if (isGGA || d_dftParamsPtr->noncolin)
          auxDensityXCRepresentation->applyLocalOperations(
            std::make_pair(iCell * numberQuadraturePointsPerCell,
                           (iCell + numCells) * numberQuadraturePointsPerCell),
            densityData);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &gradDensitySpinUp =
            densityData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &gradDensitySpinDown =
            densityData[DensityDescriptorDataAttributes::gradValuesSpinDown];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
          magAxis = densityData[DensityDescriptorDataAttributes::magAxisValues];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &gradDensityXCSpinIndex =
            spinIndex == 0 ? gradDensitySpinUp : gradDensitySpinDown;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &gradDensityXCOtherSpinIndex =
            spinIndex == 0 ? gradDensitySpinDown : gradDensitySpinUp;
        if (!d_dftParamsPtr->noncolin)
          dftfe::internal::computeVeffJxWEntries(
            std::make_pair(iCell, iCell + numCells),
            numberQuadraturePointsPerCell,
            phiValues,
            pdecDensitySpinIndex,
            pdexDensitySpinIndex,
            d_basisOperationsPtrHost->JxWBasisData(),
            d_VeffJxWHost);
        else
          dftfe::internal::computeVeffBeffJxWEntries(
            std::make_pair(iCell, iCell + numCells),
            numberQuadraturePointsPerCell,
            phiValues,
            pdecDensitySpinUp,
            pdecDensitySpinDown,
            pdexDensitySpinUp,
            pdexDensitySpinDown,
            magAxis,
            d_basisOperationsPtrHost->JxWBasisData(),
            d_VeffJxWHost,
            d_BeffxJxWHost,
            d_BeffyJxWHost,
            d_BeffzJxWHost);
        if (isGGA)
          {
            if (!d_dftParamsPtr->noncolin)
              dftfe::internal::
                computeInvJacderExcWithSigmaTimesGradRhoJxWEntries(
                  std::make_pair(iCell, iCell + numCells),
                  numberQuadraturePointsPerCell,
                  spinIndex,
                  d_basisOperationsPtrHost->cellsTypeFlag(),
                  pdecSigma,
                  pdexSigma,
                  d_basisOperationsPtrHost->JxWBasisData(),
                  d_basisOperationsPtrHost->inverseJacobiansBasisData(),
                  gradDensityXCSpinIndex,
                  gradDensityXCOtherSpinIndex,
                  d_invJacderExcWithSigmaTimesGradRhoJxWHost);
            else
              dftfe::internal::
                computeInvJacderExcWithSigmaTimesGradRhoMagJxWEntries(
                  std::make_pair(iCell, iCell + numCells),
                  numberQuadraturePointsPerCell,
                  spinIndex,
                  d_basisOperationsPtrHost->cellsTypeFlag(),
                  pdecSigma,
                  pdexSigma,
                  magAxis,
                  d_basisOperationsPtrHost->JxWBasisData(),
                  d_basisOperationsPtrHost->inverseJacobiansBasisData(),
                  gradDensitySpinUp,
                  gradDensitySpinDown,
                  d_invJacderExcWithSigmaTimesGradRhoJxWHost,
                  d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost,
                  d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost,
                  d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost);
          }

        if (isTauMGGA)
          {
            dftfe::internal::computeHalfInvJacinvJacderExcWithTauJxWEntries(
              std::make_pair(iCell, iCell + numCells),
              numberQuadraturePointsPerCell,
              d_basisOperationsPtrHost->cellsTypeFlag(),
              pdecTauSpinIndex,
              pdexTauSpinIndex,
              d_basisOperationsPtrHost->JxWBasisData(),
              d_basisOperationsPtrHost->inverseJacobiansBasisData(),
              d_invJacinvJacderExcWithTauJxWHost);
          }

        if (isTauMGGA &&
            std::is_same<dataTypes::number, std::complex<double>>::value)
          {
            // The Hamiltonian operator for the MGGA case is dependent on the k
            // point.
            dftfe::uInt offsetFactor = 0;
            for (dftfe::uInt kPointIndex = 0;
                 kPointIndex < d_kPointWeights.size();
                 kPointIndex++)
              {
#if defined(DFTFE_WITH_DEVICE)
                offsetFactor = kPointIndex * totalLocallyOwnedCells *
                               numberQuadraturePointsPerCell;
#else
                auto &d_halfKSquareTimesDerExcwithTauJxWHost =
                  d_halfKSquareTimesDerExcwithTauJxW[kPointIndex];

                auto &d_derExcwithTauTimesinvJacKpointTimesJxWHost =
                  d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex];
                offsetFactor = 0;

#endif


                const std::vector<double> kPointCoordsVector = {
                  *(d_kPointCoordinates.data() + 3 * kPointIndex),
                  *(d_kPointCoordinates.data() + 3 * kPointIndex + 1),
                  *(d_kPointCoordinates.data() + 3 * kPointIndex + 2)};
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>
                  kPointCoords(3);
                kPointCoords.copyFrom(kPointCoordsVector);
                dftfe::internal::computeKPointDependenderExcWithTauJxWEntries(
                  std::make_pair(iCell, iCell + numCells),
                  numberQuadraturePointsPerCell,
                  d_basisOperationsPtrHost->cellsTypeFlag(),
                  offsetFactor,
                  kPointCoords,
                  pdecTauSpinIndex,
                  pdexTauSpinIndex,
                  d_basisOperationsPtrHost->JxWBasisData(),
                  d_basisOperationsPtrHost->inverseJacobiansBasisData(),
                  d_halfKSquareTimesDerExcwithTauJxWHost,
                  d_derExcwithTauTimesinvJacKpointTimesJxWHost);
              } // Kpoint Loop
          }     // TauMGGA
      }         // cell loop

    // GGA gradient correction to the noncollinear effective magnetic field.
    // Pure host BLAS/loop computation with no device dependency - must run
    // regardless of DFTFE_WITH_DEVICE, unlike the device-copy steps below.
    if (d_dftParamsPtr->noncolin && isGGA)
      {
        const double scalarCoeffOne  = 1.0;
        const double scalarCoeffZero = 0.0;
        d_BLASWrapperPtrHost->xgemm(
          'T',
          'N',
          numberQuadraturePointsPerCell,
          totalLocallyOwnedCells,
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffOne,
          d_basisOperationsPtrHost->collocationShapeFunctionGradientBasisData()
            .data(),
          3 * numberQuadraturePointsPerCell,
          d_invJacderExcWithSigmaTimesGradRhoJxWHost.data(),
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffOne,
          d_VeffJxWHost.data(),
          numberQuadraturePointsPerCell);
        d_BLASWrapperPtrHost->xgemm(
          'T',
          'N',
          numberQuadraturePointsPerCell,
          totalLocallyOwnedCells,
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffOne,
          d_basisOperationsPtrHost->collocationShapeFunctionGradientBasisData()
            .data(),
          3 * numberQuadraturePointsPerCell,
          d_invJacderExcWithSigmaTimesMagXTimesGradRhoJxWHost.data(),
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffZero,
          d_BeffxGGA.data(),
          numberQuadraturePointsPerCell);
        d_BLASWrapperPtrHost->xgemm(
          'T',
          'N',
          numberQuadraturePointsPerCell,
          totalLocallyOwnedCells,
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffOne,
          d_basisOperationsPtrHost->collocationShapeFunctionGradientBasisData()
            .data(),
          3 * numberQuadraturePointsPerCell,
          d_invJacderExcWithSigmaTimesMagYTimesGradRhoJxWHost.data(),
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffZero,
          d_BeffyGGA.data(),
          numberQuadraturePointsPerCell);
        d_BLASWrapperPtrHost->xgemm(
          'T',
          'N',
          numberQuadraturePointsPerCell,
          totalLocallyOwnedCells,
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffOne,
          d_basisOperationsPtrHost->collocationShapeFunctionGradientBasisData()
            .data(),
          3 * numberQuadraturePointsPerCell,
          d_invJacderExcWithSigmaTimesMagZTimesGradRhoJxWHost.data(),
          3 * numberQuadraturePointsPerCell,
          &scalarCoeffZero,
          d_BeffzGGA.data(),
          numberQuadraturePointsPerCell);

        std::unordered_map<
          DensityDescriptorDataAttributes,
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          densityData;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
          magAxis = densityData[DensityDescriptorDataAttributes::magAxisValues];
        auxDensityXCRepresentation->applyLocalOperations(
          std::make_pair(0,
                         totalLocallyOwnedCells *
                           numberQuadraturePointsPerCell),
          densityData);

        for (dftfe::uInt iQuad = 0;
             iQuad < totalLocallyOwnedCells * numberQuadraturePointsPerCell;
             ++iQuad)
          {
            const double temp = magAxis[3 * iQuad + 0] * d_BeffxGGA[iQuad] +
                                magAxis[3 * iQuad + 1] * d_BeffyGGA[iQuad] +
                                magAxis[3 * iQuad + 2] * d_BeffzGGA[iQuad];
            d_BeffxJxWHost[iQuad] += temp * magAxis[3 * iQuad + 0];
            d_BeffyJxWHost[iQuad] += temp * magAxis[3 * iQuad + 1];
            d_BeffzJxWHost[iQuad] += temp * magAxis[3 * iQuad + 2];
          }
      }

#if defined(DFTFE_WITH_DEVICE)
    if (isTauMGGA)
      {
        for (dftfe::uInt kPointIndex = 0; kPointIndex < d_kPointWeights.size();
             kPointIndex++)
          {
            dftfe::uInt size =
              totalLocallyOwnedCells * numberQuadraturePointsPerCell;
            d_halfKSquareTimesDerExcwithTauJxW[kPointIndex].resize(size);
            d_halfKSquareTimesDerExcwithTauJxW[kPointIndex].copyFrom(
              d_halfKSquareTimesDerExcwithTauJxWHost,
              size,
              size * kPointIndex,
              0);
            d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex].resize(size *
                                                                         3);
            d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex].copyFrom(
              d_derExcwithTauTimesinvJacKpointTimesJxWHost,
              size * 3,
              size * 3 * kPointIndex,
              0);
          }
      }
#endif

#if defined(DFTFE_WITH_DEVICE)
    if (!d_dftParamsPtr->noncolin)
      {
        d_VeffJxW.resize(d_VeffJxWHost.size());
        d_VeffJxW.copyFrom(d_VeffJxWHost);
        d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
          d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
        d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
          d_invJacderExcWithSigmaTimesGradRhoJxWHost);
        d_invJacinvJacderExcWithTauJxW.resize(
          d_invJacinvJacderExcWithTauJxWHost.size());
        d_invJacinvJacderExcWithTauJxW.copyFrom(
          d_invJacinvJacderExcWithTauJxWHost);
      }
    else
      {
        d_VeffJxW.resize(d_VeffJxWHost.size());
        d_VeffJxW.copyFrom(d_VeffJxWHost);
        d_BeffxJxW.resize(d_BeffxJxWHost.size());
        d_BeffxJxW.copyFrom(d_BeffxJxWHost);
        d_BeffyJxW.resize(d_BeffyJxWHost.size());
        d_BeffyJxW.copyFrom(d_BeffyJxWHost);
        d_BeffzJxW.resize(d_BeffzJxWHost.size());
        d_BeffzJxW.copyFrom(d_BeffzJxWHost);
      }
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::setVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                     &vKS_quadValues,
    const dftfe::uInt spinIndex)
  {
    const dftfe::uInt spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const dftfe::uInt totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const dftfe::uInt numberQuadraturePointsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxWHost;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesGradRhoJxWHost;
#else
    auto &d_VeffJxWHost = d_VeffJxW;

    auto &d_invJacderExcWithSigmaTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesGradRhoJxW;
#endif
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePointsPerCell,
                         0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(0, 0.0);

    for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      {
        auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                          iCell * numberQuadraturePointsPerCell;
        for (dftfe::uInt qPoint = 0; qPoint < numberQuadraturePointsPerCell;
             ++qPoint)
          {
            // TODO extend to spin polarised case
            d_VeffJxWHost[qPoint + iCell * numberQuadraturePointsPerCell] =
              vKS_quadValues[0]
                            [qPoint + iCell * numberQuadraturePointsPerCell] *
              cellJxWPtr[qPoint];
          }
      }

    if (!d_isExternalPotCorrHamiltonianComputed)
      computeCellHamiltonianMatrixExtPotContribution();
#if defined(DFTFE_WITH_DEVICE)
    d_VeffJxW.resize(d_VeffJxWHost.size());
    d_VeffJxW.copyFrom(d_VeffJxWHost);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
    d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost);
#endif
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::computeVEffExternalPotCorr(
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_lpspQuadratureID, false);
    const dftfe::uInt nCells        = d_basisOperationsPtrHost->nCells();
    const dftfe::Int  nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffExtPotJxWHost;
#else
    auto &d_VeffExtPotJxWHost = d_VeffExtPotJxW;
#endif
    d_VeffExtPotJxWHost.resize(nCells * nQuadsPerCell);

    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        const auto &temp =
          externalPotCorrValues.find(d_basisOperationsPtrHost->cellID(iCell))
            ->second;
        const double *cellJxWPtr =
          d_basisOperationsPtrHost->JxWBasisData().data() +
          iCell * nQuadsPerCell;
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          d_VeffExtPotJxWHost[iCell * nQuadsPerCell + iQuad] =
            temp[iQuad] * cellJxWPtr[iQuad];
      }

#if defined(DFTFE_WITH_DEVICE)
    d_VeffExtPotJxW.resize(d_VeffExtPotJxWHost.size());
    d_VeffExtPotJxW.copyFrom(d_VeffExtPotJxWHost);
#endif
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::setVEffExternalPotCorrToZero()
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_lpspQuadratureID, false);
    const dftfe::uInt nCells        = d_basisOperationsPtrHost->nCells();
    const dftfe::Int  nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffExtPotJxWHost;
#else
    auto &d_VeffExtPotJxWHost = d_VeffExtPotJxW;
#endif
    d_VeffExtPotJxWHost.resize(nCells * nQuadsPerCell);

    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          d_VeffExtPotJxWHost[iCell * nQuadsPerCell + iQuad] = 0.0;
      }

#if defined(DFTFE_WITH_DEVICE)
    d_VeffExtPotJxW.resize(d_VeffExtPotJxWHost.size());
    d_VeffExtPotJxW.copyFrom(d_VeffExtPotJxWHost);
#endif
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::reinitkPointSpinIndex(
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;
    d_HamiltonianIndex =
      d_dftParamsPtr->memOptMode ?
        0 :
        kPointIndex * (d_dftParamsPtr->spinPolarized + 1) + spinIndex;
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential)
        d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
          d_kPointIndex);

    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }

    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential &&
          d_dftParamsPtr->useSinglePrecCheby)
        d_pseudopotentialNonLocalOperatorSinglePrec
          ->initialiseOperatorActionOnX(d_kPointIndex);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::reinitNumberWavefunctions(
    const dftfe::uInt numWaveFunctions)
  {
    const dftfe::uInt nCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    if (d_cellWaveFunctionMatrixSrc.size() <
        (d_dftParamsPtr->memOptMode ? d_cellsBlockSizeHX : nCells) *
          nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrc.resize(
        (d_dftParamsPtr->memOptMode ? d_cellsBlockSizeHX : nCells) *
        nDofsPerCell * numWaveFunctions);
    if (d_dftParamsPtr->useSinglePrecCheby &&
        d_cellWaveFunctionMatrixSrcSinglePrec.size() <
          (d_dftParamsPtr->memOptMode ? d_cellsBlockSizeHX : nCells) *
            nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrcSinglePrec.resize(
        (d_dftParamsPtr->memOptMode ? d_cellsBlockSizeHX : nCells) *
        nDofsPerCell * numWaveFunctions);
    if (d_cellWaveFunctionMatrixDst.size() <
        d_nOMPThreads * d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixDst.resize(d_nOMPThreads * d_cellsBlockSizeHX *
                                         nDofsPerCell * numWaveFunctions);
    if (d_dftParamsPtr->useSinglePrecCheby &&
        d_cellWaveFunctionMatrixDstSinglePrec.size() <
          d_nOMPThreads * d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixDstSinglePrec.resize(
        d_nOMPThreads * d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions);

    if (d_useHubbard)
      {
        d_hubbardClassPtr->initialiseFlattenedDataStructure(numWaveFunctions);
      }

    if (d_dftParamsPtr->isPseudopotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              numWaveFunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlock);

            d_pseudopotentialNonLocalOperator
              ->initialiseCellWaveFunctionPointers(d_cellWaveFunctionMatrixSrc,
                                                   d_cellsBlockSizeHX);
          }
        else
          {
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              numWaveFunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlock);
          }
      }
    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseFlattenedDataStructure(
                numWaveFunctions,
                d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec);
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseCellWaveFunctionPointers(
                d_cellWaveFunctionMatrixSrcSinglePrec, d_cellsBlockSizeHX);
          }
        else
          d_pseudopotentialNonLocalOperatorSinglePrec
            ->initialiseFlattenedDataStructure(
              numWaveFunctions,
              d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec);

        if (d_dftParamsPtr->communPrecCheby == "BF16")
          d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
            .setCommunicationPrecision(
              dftfe::utils::mpi::communicationPrecision::half);

        else if (d_dftParamsPtr->communPrecCheby == "COMPRESSED")
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .setCompressBitsPerValue(d_dftParamsPtr->compressBitsPerValue);
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .setCommunicationPrecision(
                dftfe::utils::mpi::communicationPrecision::compress);
          }
      }

    d_basisOperationsPtr->reinit(numWaveFunctions,
                                 d_cellsBlockSizeHX,
                                 d_densityQuadratureID,
                                 false,
                                 false);

    // TODO extend to MGGA if required
    if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::DFTPlusU) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::HYBRID))
      {
        d_basisOperationsPtr->createMultiVector(numWaveFunctions,
                                                d_srcNonLocalTemp);
        d_basisOperationsPtr->createMultiVector(numWaveFunctions,
                                                d_dstNonLocalTemp);

        if (d_dftParamsPtr->useSinglePrecCheby)
          {
            d_basisOperationsPtr->createMultiVectorSinglePrec(
              numWaveFunctions, d_srcNonLocalTempSinglePrec);
            d_basisOperationsPtr->createMultiVectorSinglePrec(
              numWaveFunctions, d_dstNonLocalTempSinglePrec);
          }
      }


    dftfe::utils::MemoryStorage<dftfe::uInt, dftfe::utils::MemorySpace::HOST>
      nodeIds;

    dftfe::uInt relaventDofs = d_basisOperationsPtr->nRelaventDofs();
    nodeIds.resize(relaventDofs);
    for (dftfe::uInt i = 0; i < relaventDofs; i++)
      {
        nodeIds.data()[i] = i * numWaveFunctions;
      }
    d_mapNodeIdToProcId.resize(relaventDofs);
    d_mapNodeIdToProcId.copyFrom(nodeIds);

    d_numVectorsInternal = numWaveFunctions;
    // if (d_dftParamsPtr->useCorrectionEquation)
    //   d_basisOperationsPtr->createMultiVector(numWaveFunctions,
    //                                           d_tempBlockVectorOverlapInvX);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  KohnShamDFTBaseOperator<memorySpace>::getMPICommunicatorDomain()
  {
    return d_mpiCommDomain;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST> *
  KohnShamDFTBaseOperator<memorySpace>::getOverloadedConstraintMatrixHost()
    const
  {
    return &(d_basisOperationsPtrHost
               ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamDFTBaseOperator<memorySpace>::getInverseSqrtMassVector()
  {
    return d_basisOperationsPtr->inverseSqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamDFTBaseOperator<memorySpace>::getSqrtMassVector()
  {
    return d_basisOperationsPtr->sqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &
  KohnShamDFTBaseOperator<memorySpace>::getScratchFEMultivector(
    const dftfe::uInt numVectors,
    const dftfe::uInt index)
  {
    return d_basisOperationsPtr->getMultiVector(
      numVectors *
        ((d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC) ? 2 : 1),
      index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
  KohnShamDFTBaseOperator<memorySpace>::getScratchFEMultivectorSinglePrec(
    const dftfe::uInt numVectors,
    const dftfe::uInt index)
  {
    return d_basisOperationsPtr->getMultiVectorSinglePrec(
      numVectors *
        ((d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC) ? 2 : 1),
      index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<
    memorySpace>::computeCellHamiltonianMatrixExtPotContribution()
  {
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_lpspQuadratureID,
                                 false,
                                 true);
    const dftfe::uInt nCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    d_cellHamiltonianMatrixExtPot.resize(nCells * nDofsPerCell * nDofsPerCell);
    d_cellHamiltonianMatrixExtPot.setValue(0.0);
    d_basisOperationsPtr->computeWeightedCellMassMatrix(
      std::pair<dftfe::uInt, dftfe::uInt>(0, nCells),
      d_VeffExtPotJxW,
      d_cellHamiltonianMatrixExtPot);
    d_isExternalPotCorrHamiltonianComputed = true;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::computeCellHamiltonianMatrix(
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    if ((d_dftParamsPtr->isPseudopotential ||
         d_dftParamsPtr->smearedNuclearCharges) &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      if (!d_isExternalPotCorrHamiltonianComputed)
        computeCellHamiltonianMatrixExtPotContribution();
    const dftfe::uInt nCells           = d_basisOperationsPtr->nCells();
    const dftfe::uInt nQuadsPerCell    = d_basisOperationsPtr->nQuadsPerCell();
    const dftfe::uInt nDofsPerCell     = d_basisOperationsPtr->nDofsPerCell();
    const double      scalarCoeffAlpha = 1.0;
    const double      scalarCoeffHalf  = 0.5;
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_densityQuadratureID,
                                 false,
                                 true);
    for (dftfe::uInt iCell = 0; iCell < nCells;
         iCell += d_cellsBlockSizeHamiltonianConstruction)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell,
          std::min(iCell + d_cellsBlockSizeHamiltonianConstruction, nCells));
        tempHamMatrixRealBlock.setValue(0.0);
        if ((d_dftParamsPtr->isPseudopotential ||
             d_dftParamsPtr->smearedNuclearCharges) &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_BLASWrapperPtr->xcopy(nDofsPerCell * nDofsPerCell *
                                      (cellRange.second - cellRange.first),
                                    d_cellHamiltonianMatrixExtPot.data() +
                                      cellRange.first * nDofsPerCell *
                                        nDofsPerCell,
                                    1,
                                    tempHamMatrixRealBlock.data(),
                                    1);
          }
        d_basisOperationsPtr->computeWeightedCellMassMatrix(
          cellRange, d_VeffJxW, tempHamMatrixRealBlock);

        if (d_dftParamsPtr->noncolin)
          {
            tempHamMatrixBZBlockNonCollin.setValue(0.0);
            tempHamMatrixBYBlockNonCollin.setValue(0.0);
            tempHamMatrixBXBlockNonCollin.setValue(0.0);
            d_basisOperationsPtr->computeWeightedCellMassMatrix(
              cellRange, d_BeffzJxW, tempHamMatrixBZBlockNonCollin);
            d_basisOperationsPtr->computeWeightedCellMassMatrix(
              cellRange, d_BeffyJxW, tempHamMatrixBYBlockNonCollin);
            d_basisOperationsPtr->computeWeightedCellMassMatrix(
              cellRange, d_BeffxJxW, tempHamMatrixBXBlockNonCollin);
          }
        else
          {
            bool isGradDensityDataDependent =
              (d_excManagerPtr->getExcSSDFunctionalObj()
                 ->getDensityBasedFamilyType() == densityFamilyType::GGA);
            if (isGradDensityDataDependent)
              d_basisOperationsPtr
                ->computeWeightedCellNjGradNiPlusNiGradNjMatrix(
                  cellRange,
                  d_invJacderExcWithSigmaTimesGradRhoJxW,
                  tempHamMatrixRealBlock);
          }
        const bool isTauMGGA =
          (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
           ExcFamilyType::TauMGGA);
        if (isTauMGGA)
          {
            d_basisOperationsPtr->computeWeightedCellStiffnessMatrix(
              cellRange,
              d_invJacinvJacderExcWithTauJxW,
              tempHamMatrixRealBlock);
          }

        if (!onlyHPrimePartForFirstOrderDensityMatResponse)
          d_BLASWrapperPtr->xaxpy(
            nDofsPerCell * nDofsPerCell * (cellRange.second - cellRange.first),
            &scalarCoeffHalf,
            d_basisOperationsPtr->cellStiffnessMatrixBasisData().data() +
              cellRange.first * nDofsPerCell * nDofsPerCell,
            1,
            tempHamMatrixRealBlock.data(),
            1);

        if constexpr (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
          {
            tempHamMatrixImagBlock.setValue(0.0);
            if (!onlyHPrimePartForFirstOrderDensityMatResponse)
              {
                const double *kPointCoords =
                  d_kPointCoordinates.data() + 3 * d_kPointIndex;
                const double kSquareTimesHalf =
                  0.5 * (kPointCoords[0] * kPointCoords[0] +
                         kPointCoords[1] * kPointCoords[1] +
                         kPointCoords[2] * kPointCoords[2]);
                d_BLASWrapperPtr->xaxpy(
                  nDofsPerCell * nDofsPerCell *
                    (cellRange.second - cellRange.first),
                  &kSquareTimesHalf,
                  d_basisOperationsPtr->cellMassMatrixBasisData().data() +
                    cellRange.first * nDofsPerCell * nDofsPerCell,
                  1,
                  tempHamMatrixRealBlock.data(),
                  1);
                d_basisOperationsPtr->computeWeightedCellNjGradNiMatrix(
                  cellRange,
                  d_invJacKPointTimesJxW[d_kPointIndex],
                  tempHamMatrixImagBlock);

                if (isTauMGGA)
                  {
                    d_basisOperationsPtr->computeWeightedCellMassMatrix(
                      cellRange,
                      d_halfKSquareTimesDerExcwithTauJxW[d_kPointIndex],
                      tempHamMatrixRealBlock);

                    d_basisOperationsPtr
                      ->computeWeightedCellNjGradNiMinusNiGradNjMatrix(
                        cellRange,
                        d_derExcwithTauTimesinvJacKpointTimesJxW[d_kPointIndex],
                        tempHamMatrixImagBlock);
                  }
              }
            if (!d_dftParamsPtr->noncolin)
              d_BLASWrapperPtr->copyRealArrsToComplexArr(
                nDofsPerCell * nDofsPerCell *
                  (cellRange.second - cellRange.first),
                tempHamMatrixRealBlock.data(),
                tempHamMatrixImagBlock.data(),
                d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                  cellRange.first * nDofsPerCell * nDofsPerCell);
            else
              {
                internal::computeCellHamiltonianMatrixNonCollinearFromBlocks(
                  cellRange,
                  nDofsPerCell,
                  tempHamMatrixRealBlock,
                  tempHamMatrixImagBlock,
                  tempHamMatrixBZBlockNonCollin,
                  tempHamMatrixBYBlockNonCollin,
                  tempHamMatrixBXBlockNonCollin,
                  d_cellHamiltonianMatrix[d_HamiltonianIndex]);
              }
          }
        else
          {
            d_BLASWrapperPtr->xcopy(
              nDofsPerCell * nDofsPerCell *
                (cellRange.second - cellRange.first),
              tempHamMatrixRealBlock.data(),
              1,
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * nDofsPerCell * nDofsPerCell,
              1);
          }
      }
    if (d_dftParamsPtr->useSinglePrecCheby)
      d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
        d_cellHamiltonianMatrix[d_HamiltonianIndex].size(),
        d_cellHamiltonianMatrix[d_HamiltonianIndex].data(),
        d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data());
    if (d_dftParamsPtr->memOptMode)
      if ((d_dftParamsPtr->isPseudopotential ||
           d_dftParamsPtr->smearedNuclearCharges) &&
          !onlyHPrimePartForFirstOrderDensityMatResponse)
        {
          d_cellHamiltonianMatrixExtPot.clear();
          d_isExternalPotCorrHamiltonianComputed = false;
        }
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::HX(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarHX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt spinorFactor   = d_dftParamsPtr->noncolin ? 2 : 1;
    const dftfe::uInt numberWavefunctions = src.numVectors() / spinorFactor;
    if (d_numVectorsInternal != numberWavefunctions * spinorFactor)
      reinitNumberWavefunctions(numberWavefunctions * spinorFactor);
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_dftParamsPtr->tensorOpType == "TF32")
          d_BLASWrapperPtr->setTensorOpDataType(
            dftfe::linearAlgebra::tensorOpDataType::tf32);
        if (d_dftParamsPtr->tensorOpType == "BF16")
          d_BLASWrapperPtr->setTensorOpDataType(
            dftfe::linearAlgebra::tensorOpDataType::bf16);
      }
#endif
    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions * spinorFactor)
      d_basisOperationsPtr->reinit(numberWavefunctions * spinorFactor,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    src.updateGhostValues();
    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute(src);
    const dataTypes::numberFP32 scalarCoeffAlpha = dataTypes::numberFP32(1.0),
                                scalarCoeffBeta  = dataTypes::numberFP32(0.0);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      {
        if (d_dftParamsPtr->isPseudopotential)
          {
            d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
              d_kPointIndex);
          }

        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedCopyToBlock(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          src.data(),
          d_cellWaveFunctionMatrixSrcSinglePrec.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            numberWavefunctions * spinorFactor),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
#pragma omp critical(hx_Cconj)
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperatorSinglePrec->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions *
                     spinorFactor),
              cellRange);
          }
      }
    if (d_dftParamsPtr->isPseudopotential &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec.setValue(
          0);
        d_pseudopotentialNonLocalOperatorSinglePrec
          ->applyAllReduceOnCconjtransX(
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec);
        d_pseudopotentialNonLocalOperatorSinglePrec->applyVOnCconjtransX(
          d_dftParamsPtr->hasSOC ? CouplingStructure::blockDiagonal :
                                   CouplingStructure::diagonal,
          d_pseudopotentialClassPtr->getCouplingMatrixSinglePrec(),
          d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
          true);
      }
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (d_dftParamsPtr->memOptMode)
          {
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions * spinorFactor,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          numberWavefunctions,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor,
          &scalarCoeffAlpha,
          d_cellWaveFunctionMatrixSrcSinglePrec.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            numberWavefunctions * spinorFactor),
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data() +
            cellRange.first * numDoFsPerCell * spinorFactor * numDoFsPerCell *
              spinorFactor,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor * numDoFsPerCell * spinorFactor,
          &scalarCoeffBeta,
          d_cellWaveFunctionMatrixDstSinglePrec.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              numberWavefunctions * spinorFactor,
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          cellRange.second - cellRange.first);
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperatorSinglePrec->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDstSinglePrec.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions * spinorFactor,
              cellRange);
          }
#pragma omp critical(hx_assembly)
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarHX,
          d_cellWaveFunctionMatrixDstSinglePrec.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              numberWavefunctions * spinorFactor,
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::DFTPlusU) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::HYBRID) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::MGGA))
      {
        dftfe::uInt relaventDofs = d_basisOperationsPtr->nRelaventDofs();
        d_BLASWrapperPtr->xcopy(src.locallyOwnedSize() * numberWavefunctions,
                                src.data(),
                                1,
                                d_srcNonLocalTempSinglePrec.data(),
                                1);

        d_srcNonLocalTempSinglePrec.updateGhostValues();
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute(d_srcNonLocalTempSinglePrec);

        d_dstNonLocalTempSinglePrec.setValue(0.0);
        d_excManagerPtr->getExcSSDFunctionalObj()
          ->applyWaveFunctionDependentFuncDerWrtPsi(d_srcNonLocalTempSinglePrec,
                                                    d_dstNonLocalTempSinglePrec,
                                                    numberWavefunctions,
                                                    d_kPointIndex,
                                                    d_spinIndex);

        d_BLASWrapperPtr->axpby(dst.localSize() * numberWavefunctions,
                                scalarHX,
                                d_dstNonLocalTempSinglePrec.data(),
                                1.0,
                                dst.data());
      }

    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute_slave_to_master(dst);


    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      d_BLASWrapperPtr->setTensorOpDataType(
        dftfe::linearAlgebra::tensorOpDataType::fp32);
#endif
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::HX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt spinorFactor   = d_dftParamsPtr->noncolin ? 2 : 1;
    const dftfe::uInt numberWavefunctions = src.numVectors() / spinorFactor;
    if (d_numVectorsInternal != numberWavefunctions * spinorFactor)
      reinitNumberWavefunctions(numberWavefunctions * spinorFactor);
    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions * spinorFactor)
      d_basisOperationsPtr->reinit(numberWavefunctions * spinorFactor,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    src.updateGhostValues();
    d_basisOperationsPtr->distribute(src);
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      {
        if (d_dftParamsPtr->isPseudopotential)
          d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
            d_kPointIndex);

        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedCopyToBlock(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            numberWavefunctions * spinorFactor),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
#pragma omp critical(hx_Cconj)
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions *
                     spinorFactor),
              cellRange);
          }
      }
    if (d_dftParamsPtr->isPseudopotential &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
        d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
          d_pseudopotentialNonLocalProjectorTimesVectorBlock);
        d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
          d_dftParamsPtr->hasSOC ? CouplingStructure::blockDiagonal :
                                   CouplingStructure::diagonal,
          d_pseudopotentialClassPtr->getCouplingMatrix(),
          d_pseudopotentialNonLocalProjectorTimesVectorBlock,
          true);
      }
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (d_dftParamsPtr->memOptMode)
          {
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions * spinorFactor,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrc.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          numberWavefunctions,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor,
          &scalarCoeffAlpha,
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            spinorFactor * numberWavefunctions),
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
            cellRange.first * numDoFsPerCell * spinorFactor * numDoFsPerCell *
              spinorFactor,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor * numDoFsPerCell * spinorFactor,
          &scalarCoeffBeta,
          d_cellWaveFunctionMatrixDst.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              spinorFactor * numberWavefunctions,
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          cellRange.second - cellRange.first);
        if (hasNonlocalComponents)
          d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
            d_cellWaveFunctionMatrixDst.data() +
              omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                spinorFactor * numberWavefunctions,
            cellRange);
#pragma omp critical(hx_assembly)
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarHX,
          d_cellWaveFunctionMatrixDst.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              numberWavefunctions * spinorFactor,
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::DFTPlusU) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::HYBRID) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::MGGA))
      {
        dftfe::uInt relaventDofs = d_basisOperationsPtr->nRelaventDofs();
        d_BLASWrapperPtr->xcopy(src.locallyOwnedSize() * numberWavefunctions,
                                src.data(),
                                1,
                                d_srcNonLocalTemp.data(),
                                1);

        d_srcNonLocalTemp.updateGhostValues();
        d_basisOperationsPtr->distribute(d_srcNonLocalTemp);

        d_dstNonLocalTemp.setValue(0.0);
        d_excManagerPtr->getExcSSDFunctionalObj()
          ->applyWaveFunctionDependentFuncDerWrtPsi(d_srcNonLocalTemp,
                                                    d_dstNonLocalTemp,
                                                    numberWavefunctions,
                                                    d_kPointIndex,
                                                    d_spinIndex);

        d_BLASWrapperPtr->axpby(dst.localSize() * numberWavefunctions,
                                scalarHX,
                                d_dstNonLocalTemp.data(),
                                1.0,
                                dst.data());
      }

    d_basisOperationsPtr->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
      .distribute_slave_to_master(dst);


    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTBaseOperator<memorySpace>::HXWithLowdinOrthonormalisedInput(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt spinorFactor   = d_dftParamsPtr->noncolin ? 2 : 1;
    const dftfe::uInt numberWavefunctions = src.numVectors() / spinorFactor;
    if (d_numVectorsInternal != numberWavefunctions * spinorFactor)
      reinitNumberWavefunctions(numberWavefunctions * spinorFactor);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions * spinorFactor)
      d_basisOperationsPtr->reinit(numberWavefunctions * spinorFactor,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);

    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());

    src.updateGhostValues();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      {
        if (d_dftParamsPtr->isPseudopotential)
          {
            d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
              d_kPointIndex);
          }

        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          1.0,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            numberWavefunctions * spinorFactor),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
#pragma omp critical(hx_Cconj)
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrc.data() +
                (d_dftParamsPtr->memOptMode ?
                   0 :
                   cellRange.first * numDoFsPerCell * numberWavefunctions *
                     spinorFactor),
              cellRange);
          }
      }
    if (d_dftParamsPtr->isPseudopotential &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
        d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
          d_pseudopotentialNonLocalProjectorTimesVectorBlock);
        d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
          d_dftParamsPtr->hasSOC ? CouplingStructure::blockDiagonal :
                                   CouplingStructure::diagonal,
          d_pseudopotentialClassPtr->getCouplingMatrix(),
          d_pseudopotentialNonLocalProjectorTimesVectorBlock,
          true);
      }
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        if (d_dftParamsPtr->memOptMode)
          {
            d_BLASWrapperPtr->stridedBlockScaleCopy(
              numberWavefunctions * spinorFactor,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              1.0,
              d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData()
                  .data() +
                cellRange.first * numDoFsPerCell,
              src.data(),
              d_cellWaveFunctionMatrixSrc.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          numberWavefunctions,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor,
          &scalarCoeffAlpha,
          d_cellWaveFunctionMatrixSrc.data() +
            (d_dftParamsPtr->memOptMode ? 0 :
                                          cellRange.first * numDoFsPerCell *
                                            spinorFactor * numberWavefunctions),
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
            cellRange.first * numDoFsPerCell * spinorFactor * numDoFsPerCell *
              spinorFactor,
          numDoFsPerCell * spinorFactor,
          numDoFsPerCell * spinorFactor * numDoFsPerCell * spinorFactor,
          &scalarCoeffBeta,
          d_cellWaveFunctionMatrixDst.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              spinorFactor * numberWavefunctions,
          numberWavefunctions,
          numDoFsPerCell * spinorFactor * numberWavefunctions,
          cellRange.second - cellRange.first);
        if (hasNonlocalComponents)
          {
            d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions * spinorFactor,
              cellRange);
          }
#pragma omp critical(hx_assembly)
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions * spinorFactor,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarHX,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          d_cellWaveFunctionMatrixDst.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              numberWavefunctions * spinorFactor,
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::DFTPlusU) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::HYBRID) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::MGGA))
      {
        dftfe::uInt relaventDofs = d_basisOperationsPtr->nRelaventDofs();
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions,
          relaventDofs,
          1.0,
          getInverseSqrtMassVector().data(),
          src.data(),
          d_srcNonLocalTemp.data(),
          d_mapNodeIdToProcId.data());

        d_srcNonLocalTemp.updateGhostValues();
        d_basisOperationsPtr->distribute(d_srcNonLocalTemp);

        // TODO d_srcNonLocalTemp and d_dstNonLocalTemp can be removed
        d_dstNonLocalTemp.setValue(0.0);
        d_excManagerPtr->getExcSSDFunctionalObj()
          ->applyWaveFunctionDependentFuncDerWrtPsi(d_srcNonLocalTemp,
                                                    d_dstNonLocalTemp,
                                                    numberWavefunctions,
                                                    d_kPointIndex,
                                                    d_spinIndex);


        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(d_dstNonLocalTemp);


        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions,
          relaventDofs,
          scalarHX,
          getInverseSqrtMassVector().data(),
          d_dstNonLocalTemp.data(),
          dst.data(),
          d_mapNodeIdToProcId.data());
      }

    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr
      ->distribute_slave_to_master(dst);
    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
  }

  template class KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
