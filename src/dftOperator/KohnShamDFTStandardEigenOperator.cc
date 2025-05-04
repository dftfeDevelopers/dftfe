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

#include <KohnShamDFTStandardEigenOperator.h>
#include <ExcDFTPlusU.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#endif

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamDFTStandardEigenOperator<memorySpace>::
    KohnShamDFTStandardEigenOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
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
    : KohnShamDFTBaseOperator<memorySpace>(BLASWrapperPtr,
                                           basisOperationsPtr,
                                           basisOperationsPtrHost,
                                           pseudopotentialClassPtr,
                                           excManagerPtr,
                                           dftParamsPtr,
                                           densityQuadratureID,
                                           lpspQuadratureID,
                                           feOrderPlusOneQuadratureID,
                                           mpi_comm_parent,
                                           mpi_comm_domain)
  {}



  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTStandardEigenOperator<memorySpace>::overlapMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarOX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool useApproximateMatrixEntries)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    const double      one(1.0);
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);

    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    if (useApproximateMatrixEntries)
      {
        const dftfe::uInt blockSize = src.numVectors();
        d_BLASWrapperPtr->stridedBlockAxpy(
          blockSize,
          src.locallyOwnedSize(),
          src.data(),
          d_basisOperationsPtr->massVectorBasisData().data(),
          scalarOX,
          dst.data());
      }
    else
      {
        src.updateGhostValues();
        d_basisOperationsPtr->distribute(src);
        const dataTypes::number scalarCoeffAlpha = scalarOX,
                                scalarCoeffBeta  = dataTypes::number(0.0);
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->cellMassMatrix().data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
#pragma omp critical(hx_assembly)
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarOX,
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(dst);
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTStandardEigenOperator<memorySpace>::overlapInverseMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double scalarOinvX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst)
  {
    const dftfe::uInt blockSize = src.numVectors();
    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * blockSize,
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    d_BLASWrapperPtr->stridedBlockAxpy(
      blockSize,
      src.locallyOwnedSize(),
      src.data(),
      d_basisOperationsPtr->inverseMassVectorBasisData().data(),
      scalarOinvX,
      dst.data());
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTStandardEigenOperator<memorySpace>::overlapInverseMatrixTimesX(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarOinvX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst)
  {
    const dftfe::uInt blockSize = src.numVectors();
    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * blockSize,
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());
    d_BLASWrapperPtr->stridedBlockAxpy(
      blockSize,
      src.locallyOwnedSize(),
      src.data(),
      d_basisOperationsPtr->inverseMassVectorBasisData().data(),
      scalarOinvX,
      dst.data());
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTStandardEigenOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse,
    const bool skip1,
    const bool skip2,
    const bool skip3)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperator
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            if (d_dftParamsPtr->isPseudopotential)
              d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
                d_kPointIndex);

            d_excManagerPtr->getExcSSDFunctionalObj()
              ->reinitKPointDependentVariables(d_kPointIndex);
          }
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedBlockScaleCopy(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              1.0,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              src.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
#pragma omp critical(hxc_Cconj)
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperator->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrc.data() +
                  cellRange.first * numDoFsPerCell * numberWavefunctions,
                cellRange);
          }
      }
    if (!skip2)
      {
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlock.setValue(0);
            d_pseudopotentialNonLocalOperator->applyAllReduceOnCconjtransX(
              d_pseudopotentialNonLocalProjectorTimesVectorBlock, true);
            d_pseudopotentialNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedEnd();
            d_pseudopotentialNonLocalProjectorTimesVectorBlock
              .updateGhostValuesBegin();
          }
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlock
              .updateGhostValuesEnd();
            d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_pseudopotentialClassPtr->getCouplingMatrix(),
              d_pseudopotentialNonLocalProjectorTimesVectorBlock,
              true);
          }
      }
    if (!skip3)
      {
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperator->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDst.data() +
                  omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                    numberWavefunctions,
                cellRange);
#pragma omp critical(hxc_assembly)
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
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
            d_BLASWrapperPtr->stridedBlockAxpBy(
              numberWavefunctions,
              src.locallyOwnedSize(),
              src.data(),
              d_basisOperationsPtr->inverseMassVectorBasisData().data(),
              1.0,
              0.0,
              d_srcNonLocalTemp.data());

            d_srcNonLocalTemp.updateGhostValues();
            d_basisOperationsPtr->distribute(d_srcNonLocalTemp);

            d_dstNonLocalTemp.setValue(0.0);
            d_excManagerPtr->getExcSSDFunctionalObj()
              ->applyWaveFunctionDependentFuncDerWrtPsi(d_srcNonLocalTemp,
                                                        d_dstNonLocalTemp,
                                                        numberWavefunctions,
                                                        d_kPointIndex,
                                                        d_spinIndex);

            d_BLASWrapperPtr->axpby(relaventDofs * numberWavefunctions,
                                    scalarHX,
                                    d_dstNonLocalTemp.data(),
                                    1.0,
                                    dst.data());
          }

        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(dst);
      }
    if (!skip1 && !skip2 && !skip3)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamDFTStandardEigenOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarHX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse,
    const bool skip1,
    const bool skip2,
    const bool skip3)
  {
    const dftfe::uInt numCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numberWavefunctions = src.numVectors();
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_dftParamsPtr->tensorOpType == "TF32")
          d_BLASWrapperPtr->setTensorOpDataType(
            dftfe::linearAlgebra::tensorOpDataType::tf32);
      }
#endif
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_pseudopotentialNonLocalOperatorSinglePrec
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
    const dataTypes::numberFP32 scalarCoeffAlpha = dataTypes::numberFP32(1.0),
                                scalarCoeffBeta  = dataTypes::numberFP32(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->initialiseOperatorActionOnX(d_kPointIndex);
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedBlockScaleCopy(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              1.0,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              src.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
#pragma omp critical(hxc_Cconj)
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperatorSinglePrec->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                  cellRange.first * numDoFsPerCell * numberWavefunctions,
                cellRange);
          }
      }
    if (!skip2)
      {
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .setValue(0);
            d_pseudopotentialNonLocalOperatorSinglePrec
              ->applyAllReduceOnCconjtransX(
                d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
                true);
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedEnd();
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .updateGhostValuesBegin();
          }
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec
              .updateGhostValuesEnd();
            d_pseudopotentialNonLocalOperatorSinglePrec->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_pseudopotentialClassPtr->getCouplingMatrixSinglePrec(),
              d_pseudopotentialNonLocalProjectorTimesVectorBlockSinglePrec,
              true);
          }
      }
    if (!skip3)
      {
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDstSinglePrec.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            if (hasNonlocalComponents)
              d_pseudopotentialNonLocalOperatorSinglePrec->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDstSinglePrec.data() +
                  omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                    numberWavefunctions,
                cellRange);
#pragma omp critical(hxc_assembly)
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_cellWaveFunctionMatrixDstSinglePrec.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
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
            d_BLASWrapperPtr->stridedBlockAxpBy(
              numberWavefunctions,
              src.locallyOwnedSize(),
              src.data(),
              d_basisOperationsPtr->inverseMassVectorBasisData().data(),
              1.0,
              0.0,
              d_srcNonLocalTempSinglePrec.data());

            d_srcNonLocalTempSinglePrec.updateGhostValues();

            d_basisOperationsPtr
              ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
              .distribute(d_srcNonLocalTempSinglePrec);

            d_dstNonLocalTempSinglePrec.setValue(0.0);
            d_excManagerPtr->getExcSSDFunctionalObj()
              ->applyWaveFunctionDependentFuncDerWrtPsi(
                d_srcNonLocalTempSinglePrec,
                d_dstNonLocalTempSinglePrec,
                numberWavefunctions,
                d_kPointIndex,
                d_spinIndex);

            d_BLASWrapperPtr->axpby(relaventDofs * numberWavefunctions,
                                    scalarHX,
                                    d_dstNonLocalTempSinglePrec.data(),
                                    1.0,
                                    dst.data());
          }

        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(dst);
      }
    if (!skip1 && !skip2 && !skip3)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
#if defined(DFTFE_WITH_DEVICE)
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      d_BLASWrapperPtr->setTensorOpDataType(
        dftfe::linearAlgebra::tensorOpDataType::fp32);
#endif
  }

  template class KohnShamDFTStandardEigenOperator<
    dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamDFTStandardEigenOperator<
    dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
