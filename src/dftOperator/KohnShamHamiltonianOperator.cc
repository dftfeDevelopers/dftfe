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
// @author Nikhil Kodali
//

#include <KohnShamHamiltonianOperator.h>
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
  //
  // constructor
  //
  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamHamiltonianOperator<memorySpace>::KohnShamHamiltonianOperator(
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
    std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                             oncvClassPtr,
    std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    dftParameters *                          dftParamsPtr,
    const unsigned int                       densityQuadratureID,
    const unsigned int                       lpspQuadratureID,
    const unsigned int                       feOrderPlusOneQuadratureID,
    const MPI_Comm &                         mpi_comm_parent,
    const MPI_Comm &                         mpi_comm_domain)
    : d_kPointIndex(0)
    , d_spinIndex(0)
    , d_HamiltonianIndex(0)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_basisOperationsPtr(basisOperationsPtr)
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_oncvClassPtr(oncvClassPtr)
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
    d_nOMPThreads =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 1 : d_nOMPThreads;
    if (d_dftParamsPtr->isPseudopotential)
      d_ONCVnonLocalOperator = oncvClassPtr->getNonLocalOperator();

    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      d_ONCVnonLocalOperatorSinglePrec =
        oncvClassPtr->getNonLocalOperatorSinglePrec();
    d_cellsBlockSizeHamiltonianConstruction =
      memorySpace == dftfe::utils::MemorySpace::HOST ? 1 : 50;
    d_cellsBlockSizeHX = memorySpace == dftfe::utils::MemorySpace::HOST ?
                           1 :
                           d_basisOperationsPtr->nCells();
    d_numVectorsInternal = 0;

    d_useHubbard = false;
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
  // initialize KohnShamHamiltonianOperator object
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::init(
    const std::vector<double> &kPointCoordinates,
    const std::vector<double> &kPointWeights)
  {
    computing_timer.enter_subsection("KohnShamHamiltonianOperator setup");
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
    d_cellHamiltonianMatrix.resize(
      d_dftParamsPtr->memOptMode ?
        1 :
        (d_kPointWeights.size() * (d_dftParamsPtr->spinPolarized + 1)));
    d_cellHamiltonianMatrixSinglePrec.resize(
      d_dftParamsPtr->useSinglePrecCheby ? d_cellHamiltonianMatrix.size() : 0);

    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    tempHamMatrixRealBlock.resize(nDofsPerCell * nDofsPerCell *
                                  d_cellsBlockSizeHamiltonianConstruction);
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      tempHamMatrixImagBlock.resize(nDofsPerCell * nDofsPerCell *
                                    d_cellsBlockSizeHamiltonianConstruction);
    for (unsigned int iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrix.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrix[iHamiltonian].resize(nDofsPerCell * nDofsPerCell *
                                                   nCells);
    for (unsigned int iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrixSinglePrec.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrixSinglePrec[iHamiltonian].resize(
        nDofsPerCell * nDofsPerCell * nCells);

    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID, false);
    const unsigned int numberQuadraturePoints =
      d_basisOperationsPtrHost->nQuadsPerCell();
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
#if defined(DFTFE_WITH_DEVICE)
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            d_invJacKPointTimesJxWHost;
#else
          auto &d_invJacKPointTimesJxWHost =
            d_invJacKPointTimesJxW[kPointIndex];
#endif
          d_invJacKPointTimesJxWHost.resize(nCells * numberQuadraturePoints * 3,
                                            0.0);
          for (unsigned int iCell = 0; iCell < nCells; ++iCell)
            {
              auto cellJxWPtr =
                d_basisOperationsPtrHost->JxWBasisData().data() +
                iCell * numberQuadraturePoints;
              const double *kPointCoordinatesPtr =
                kPointCoordinates.data() + 3 * kPointIndex;

              if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                {
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                           iCell * 9);
                      for (unsigned jDim = 0; jDim < 3; ++jDim)
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacKPointTimesJxWHost[iCell *
                                                       numberQuadraturePoints *
                                                       3 +
                                                     iQuad * 3 + iDim] +=
                            -inverseJacobiansQuadPtr[3 * jDim + iDim] *
                            kPointCoordinatesPtr[jDim] * cellJxWPtr[iQuad];
                    }
                }
              else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                {
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        d_invJacKPointTimesJxWHost[iCell *
                                                     numberQuadraturePoints *
                                                     3 +
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
    computing_timer.leave_subsection("KohnShamHamiltonianOperator setup");
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::resetExtPotHamFlag()
  {
    d_isExternalPotCorrHamiltonianComputed = false;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::resetKohnShamOp()
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

        d_hubbardClassPtr->initialiseCellWaveFunctionPointers(
          d_numVectorsInternal);
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEff(
    std::shared_ptr<AuxDensityMatrix<memorySpace>> auxDensityXCRepresentation,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                phiValues,
    const unsigned int spinIndex)
  {
    bool isIntegrationByPartsGradDensityDependenceVxc =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    const bool isGGA = isIntegrationByPartsGradDensityDependenceVxc;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const unsigned int totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int numberQuadraturePointsPerCell =
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
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.clear();
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(
      isGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 : 0,
      0.0);


    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      xDataOut;
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      cDataOut;


    std::vector<double> &pdexDensitySpinUp =
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdexDensitySpinDown =
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    std::vector<double> &pdecDensitySpinUp =
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdecDensitySpinDown =
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    if (isGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
        cDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
      }

    auto quadPointsAll = d_basisOperationsPtrHost->quadPoints();

    auto quadWeightsAll = d_basisOperationsPtrHost->JxW();

    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      {
        std::vector<double> quadPointsInCell(numberQuadraturePointsPerCell * 3);
        std::vector<double> quadWeightsInCell(numberQuadraturePointsPerCell);
        for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsPerCell;
             ++iQuad)
          {
            for (unsigned int idim = 0; idim < 3; ++idim)
              quadPointsInCell[3 * iQuad + idim] =
                quadPointsAll[iCell * numberQuadraturePointsPerCell * 3 +
                              3 * iQuad + idim];
            quadWeightsInCell[iQuad] = std::real(
              quadWeightsAll[iCell * numberQuadraturePointsPerCell + iQuad]);
          }

        d_excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCRepresentation, quadPointsInCell, xDataOut, cDataOut);

        const std::vector<double> &pdexDensitySpinIndex =
          spinIndex == 0 ? pdexDensitySpinUp : pdexDensitySpinDown;
        const std::vector<double> &pdecDensitySpinIndex =
          spinIndex == 0 ? pdecDensitySpinUp : pdecDensitySpinDown;

        std::vector<double> pdexSigma;
        std::vector<double> pdecSigma;
        if (isGGA)
          {
            pdexSigma = xDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            pdecSigma = cDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          }

        std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
                             densityData;
        std::vector<double> &densitySpinUp =
          densityData[DensityDescriptorDataAttributes::valuesSpinUp];
        std::vector<double> &densitySpinDown =
          densityData[DensityDescriptorDataAttributes::valuesSpinDown];
        std::vector<double> &gradDensitySpinUp =
          densityData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        std::vector<double> &gradDensitySpinDown =
          densityData[DensityDescriptorDataAttributes::gradValuesSpinDown];

        if (isGGA)
          auxDensityXCRepresentation->applyLocalOperations(quadPointsInCell,
                                                           densityData);

        const std::vector<double> &gradDensityXCSpinIndex =
          spinIndex == 0 ? gradDensitySpinUp : gradDensitySpinDown;
        const std::vector<double> &gradDensityXCOtherSpinIndex =
          spinIndex == 0 ? gradDensitySpinDown : gradDensitySpinUp;


        const double *tempPhi =
          phiValues.data() + iCell * numberQuadraturePointsPerCell;

        auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                          iCell * numberQuadraturePointsPerCell;
        for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsPerCell;
             ++iQuad)
          {
            d_VeffJxWHost[iCell * numberQuadraturePointsPerCell + iQuad] =
              (tempPhi[iQuad] + pdexDensitySpinIndex[iQuad] +
               pdecDensitySpinIndex[iQuad]) *
              cellJxWPtr[iQuad];
          }

        if (isGGA)
          {
            if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
              {
                for (unsigned int iQuad = 0;
                     iQuad < numberQuadraturePointsPerCell;
                     ++iQuad)
                  {
                    const double *inverseJacobiansQuadPtr =
                      d_basisOperationsPtrHost->inverseJacobiansBasisData()
                        .data() +
                      (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                         iCell * numberQuadraturePointsPerCell * 9 + iQuad * 9 :
                         iCell * 9);
                    const double *gradDensityQuadPtr =
                      gradDensityXCSpinIndex.data() + iQuad * 3;
                    const double *gradDensityOtherQuadPtr =
                      gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                    const double term = (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                                         pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                                        cellJxWPtr[iQuad];
                    const double termoff =
                      (pdexSigma[iQuad * 3 + 1] + pdecSigma[iQuad * 3 + 1]) *
                      cellJxWPtr[iQuad];
                    for (unsigned jDim = 0; jDim < 3; ++jDim)
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        d_invJacderExcWithSigmaTimesGradRhoJxWHost
                          [iCell * numberQuadraturePointsPerCell * 3 +
                           iQuad * 3 + iDim] +=
                          inverseJacobiansQuadPtr[3 * jDim + iDim] *
                          (2.0 * gradDensityQuadPtr[jDim] * term +
                           gradDensityOtherQuadPtr[jDim] * termoff);
                  }
              }
            else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
              {
                for (unsigned int iQuad = 0;
                     iQuad < numberQuadraturePointsPerCell;
                     ++iQuad)
                  {
                    const double *inverseJacobiansQuadPtr =
                      d_basisOperationsPtrHost->inverseJacobiansBasisData()
                        .data() +
                      iCell * 3;
                    const double *gradDensityQuadPtr =
                      gradDensityXCSpinIndex.data() + iQuad * 3;
                    const double *gradDensityOtherQuadPtr =
                      gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                    const double term = (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                                         pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                                        cellJxWPtr[iQuad];
                    const double termoff =
                      (pdexSigma[iQuad * 3 + 1] + pdecSigma[iQuad * 3 + 1]) *
                      cellJxWPtr[iQuad];
                    for (unsigned iDim = 0; iDim < 3; ++iDim)
                      d_invJacderExcWithSigmaTimesGradRhoJxWHost
                        [iCell * numberQuadraturePointsPerCell * 3 + iQuad * 3 +
                         iDim] = inverseJacobiansQuadPtr[iDim] *
                                 (2.0 * gradDensityQuadPtr[iDim] * term +
                                  gradDensityOtherQuadPtr[iDim] * termoff);
                  }
              }
          } // GGA
      }     // cell loop
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
  KohnShamHamiltonianOperator<memorySpace>::setVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                vKS_quadValues,
    const unsigned int spinIndex)
  {
    const unsigned int spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const unsigned int totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int numberQuadraturePoints =
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
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(0, 0.0);

    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      {
        auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                          iCell * numberQuadraturePoints;
        for (unsigned int qPoint = 0; qPoint < numberQuadraturePoints; ++qPoint)
          {
            // TODO extend to spin polarised case
            d_VeffJxWHost[qPoint + iCell * numberQuadraturePoints] =
              vKS_quadValues[0][qPoint + iCell * numberQuadraturePoints] *
              cellJxWPtr[qPoint];
          }
      }

    resetExtPotHamFlag();
    setVEffExternalPotCorrToZero();
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
  KohnShamHamiltonianOperator<memorySpace>::computeVEffExternalPotCorr(
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_lpspQuadratureID, false);
    const unsigned int nCells = d_basisOperationsPtrHost->nCells();
    const int nQuadsPerCell   = d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffExtPotJxWHost;
#else
    auto &d_VeffExtPotJxWHost = d_VeffExtPotJxW;
#endif
    d_VeffExtPotJxWHost.resize(nCells * nQuadsPerCell);

    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        const auto &temp =
          externalPotCorrValues.find(d_basisOperationsPtrHost->cellID(iCell))
            ->second;
        const double *cellJxWPtr =
          d_basisOperationsPtrHost->JxWBasisData().data() +
          iCell * nQuadsPerCell;
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
  KohnShamHamiltonianOperator<memorySpace>::setVEffExternalPotCorrToZero()
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_lpspQuadratureID, false);
    const unsigned int nCells = d_basisOperationsPtrHost->nCells();
    const int nQuadsPerCell   = d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffExtPotJxWHost;
#else
    auto &d_VeffExtPotJxWHost = d_VeffExtPotJxW;
#endif
    d_VeffExtPotJxWHost.resize(nCells * nQuadsPerCell);

    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          d_VeffExtPotJxWHost[iCell * nQuadsPerCell + iQuad] = 0.0;
      }

#if defined(DFTFE_WITH_DEVICE)
    d_VeffExtPotJxW.resize(d_VeffExtPotJxWHost.size());
    d_VeffExtPotJxW.copyFrom(d_VeffExtPotJxWHost);
#endif
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::reinitkPointSpinIndex(
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;
    d_HamiltonianIndex =
      d_dftParamsPtr->memOptMode ?
        0 :
        kPointIndex * (d_dftParamsPtr->spinPolarized + 1) + spinIndex;
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential)
        d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);

    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }

    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential &&
          d_dftParamsPtr->useSinglePrecCheby)
        d_ONCVnonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
          d_kPointIndex);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::reinitNumberWavefunctions(
    const unsigned int numWaveFunctions)
  {
    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    if (d_cellWaveFunctionMatrixSrc.size() <
        nCells * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrc.resize(nCells * nDofsPerCell *
                                         numWaveFunctions);
    if (d_dftParamsPtr->useSinglePrecCheby &&
        d_cellWaveFunctionMatrixSrcSinglePrec.size() <
          nCells * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrcSinglePrec.resize(nCells * nDofsPerCell *
                                                   numWaveFunctions);
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

        d_hubbardClassPtr->initialiseCellWaveFunctionPointers(numWaveFunctions);
      }

    if (d_dftParamsPtr->isPseudopotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_ONCVnonLocalOperator->initialiseFlattenedDataStructure(
              numWaveFunctions, d_ONCVNonLocalProjectorTimesVectorBlock);
            d_ONCVnonLocalOperator->initialiseCellWaveFunctionPointers(
              d_cellWaveFunctionMatrixSrc);
          }
        else
          {
            d_ONCVnonLocalOperator->initialiseFlattenedDataStructure(
              numWaveFunctions, d_ONCVNonLocalProjectorTimesVectorBlock);
          }
      }
    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_ONCVnonLocalOperatorSinglePrec->initialiseFlattenedDataStructure(
              numWaveFunctions,
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec);
            d_ONCVnonLocalOperatorSinglePrec
              ->initialiseCellWaveFunctionPointers(
                d_cellWaveFunctionMatrixSrcSinglePrec);
          }
        else
          d_ONCVnonLocalOperatorSinglePrec->initialiseFlattenedDataStructure(
            numWaveFunctions,
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec);
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
      }


    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
      nodeIds;

    unsigned int relaventDofs = d_basisOperationsPtr->nRelaventDofs();
    nodeIds.resize(relaventDofs);
    for (dftfe::size_type i = 0; i < relaventDofs; i++)
      {
        nodeIds.data()[i] = i * numWaveFunctions;
      }
    d_mapNodeIdToProcId.resize(relaventDofs);
    d_mapNodeIdToProcId.copyFrom(nodeIds);

    d_numVectorsInternal = numWaveFunctions;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  KohnShamHamiltonianOperator<memorySpace>::getMPICommunicatorDomain()
  {
    return d_mpiCommDomain;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST> *
  KohnShamHamiltonianOperator<memorySpace>::getOverloadedConstraintMatrixHost()
    const
  {
    return &(d_basisOperationsPtrHost
               ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getInverseSqrtMassVector()
  {
    return d_basisOperationsPtr->inverseSqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getSqrtMassVector()
  {
    return d_basisOperationsPtr->sqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getScratchFEMultivector(
    const unsigned int numVectors,
    const unsigned int index)
  {
    return d_basisOperationsPtr->getMultiVector(numVectors, index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getScratchFEMultivectorSinglePrec(
    const unsigned int numVectors,
    const unsigned int index)
  {
    return d_basisOperationsPtr->getMultiVectorSinglePrec(numVectors, index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<
    memorySpace>::computeCellHamiltonianMatrixExtPotContribution()
  {
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_lpspQuadratureID,
                                 false,
                                 true);
    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    d_cellHamiltonianMatrixExtPot.resize(nCells * nDofsPerCell * nDofsPerCell);
    d_cellHamiltonianMatrixExtPot.setValue(0.0);
    d_basisOperationsPtr->computeWeightedCellMassMatrix(
      std::pair<unsigned int, unsigned int>(0, nCells),
      d_VeffExtPotJxW,
      d_cellHamiltonianMatrixExtPot);
    d_isExternalPotCorrHamiltonianComputed = true;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeCellHamiltonianMatrix(
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
    const unsigned int nCells           = d_basisOperationsPtr->nCells();
    const unsigned int nQuadsPerCell    = d_basisOperationsPtr->nQuadsPerCell();
    const unsigned int nDofsPerCell     = d_basisOperationsPtr->nDofsPerCell();
    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffHalf  = 0.5;
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_densityQuadratureID,
                                 false,
                                 true);
    for (unsigned int iCell = 0; iCell < nCells;
         iCell += d_cellsBlockSizeHamiltonianConstruction)
      {
        std::pair<unsigned int, unsigned int> cellRange(
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

        bool isGradDensityDataDependent =
          (d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);
        if (isGradDensityDataDependent)
          d_basisOperationsPtr->computeWeightedCellNjGradNiPlusNiGradNjMatrix(
            cellRange,
            d_invJacderExcWithSigmaTimesGradRhoJxW,
            tempHamMatrixRealBlock);
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
                const double *kPointCoors =
                  d_kPointCoordinates.data() + 3 * d_kPointIndex;
                const double kSquareTimesHalf =
                  0.5 * (kPointCoors[0] * kPointCoors[0] +
                         kPointCoors[1] * kPointCoors[1] +
                         kPointCoors[2] * kPointCoors[2]);
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
              }
            d_BLASWrapperPtr->copyRealArrsToComplexArr(
              nDofsPerCell * nDofsPerCell *
                (cellRange.second - cellRange.first),
              tempHamMatrixRealBlock.data(),
              tempHamMatrixImagBlock.data(),
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * nDofsPerCell * nDofsPerCell);
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
  KohnShamHamiltonianOperator<memorySpace>::HX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
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

    src.updateGhostValues();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      {
        if (d_dftParamsPtr->isPseudopotential)
          {
            d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
          }

        d_excManagerPtr->getExcSSDFunctionalObj()
          ->reinitKPointDependentVariables(d_kPointIndex);
      }
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
       0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (unsigned int iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          1.0,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            cellRange.first * numDoFsPerCell * numberWavefunctions,
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
#pragma omp critical(hx_Cconj)
        if (hasNonlocalComponents)
          {
            d_ONCVnonLocalOperator->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              cellRange);
          }
      }
    if (d_dftParamsPtr->isPseudopotential &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        d_ONCVNonLocalProjectorTimesVectorBlock.setValue(0);
        d_ONCVnonLocalOperator->applyAllReduceOnCconjtransX(
          d_ONCVNonLocalProjectorTimesVectorBlock);
        d_ONCVnonLocalOperator->applyVOnCconjtransX(
          CouplingStructure::diagonal,
          d_oncvClassPtr->getCouplingMatrix(),
          d_ONCVNonLocalProjectorTimesVectorBlock,
          true);
      }
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (unsigned int iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<unsigned int, unsigned int> cellRange(
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
          {
            d_ONCVnonLocalOperator->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              cellRange);
          }
#pragma omp critical(hx_assembly)
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarHX,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          d_cellWaveFunctionMatrixDst.data() +
            omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
              numberWavefunctions,
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr
      ->distribute_slave_to_master(dst);

    if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::DFTPlusU) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::HYBRID) ||
        (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
         ExcFamilyType::MGGA))
      {
        unsigned int relaventDofs = d_basisOperationsPtr->nRelaventDofs();
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

    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::HXCheby(
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
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
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
      (d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
       0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;

    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        d_basisOperationsPtr->distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            if (d_dftParamsPtr->isPseudopotential)
              {
                d_ONCVnonLocalOperator->initialiseOperatorActionOnX(
                  d_kPointIndex);
              }
            d_excManagerPtr->getExcSSDFunctionalObj()
              ->reinitKPointDependentVariables(d_kPointIndex);
          }
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
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
#pragma omp critical(hxc_Cconj)
            if (hasNonlocalComponents)
              {
                d_ONCVnonLocalOperator->applyCconjtransOnX(
                  d_cellWaveFunctionMatrixSrc.data() +
                    cellRange.first * numDoFsPerCell * numberWavefunctions,
                  cellRange);
              }
          }
      }
    if (!skip2)
      {
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock.setValue(0);
            d_ONCVnonLocalOperator->applyAllReduceOnCconjtransX(
              d_ONCVNonLocalProjectorTimesVectorBlock, true);
            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedEnd();
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesBegin();
          }
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesEnd();
            d_ONCVnonLocalOperator->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_oncvClassPtr->getCouplingMatrix(),
              d_ONCVNonLocalProjectorTimesVectorBlock,
              true);
          }
      }


    if (!skip3)
      {
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
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
              {
                d_ONCVnonLocalOperator->applyCOnVCconjtransX(
                  d_cellWaveFunctionMatrixDst.data() +
                    omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                      numberWavefunctions,
                  cellRange);
              }
#pragma omp critical(hxc_assembly)
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              d_cellWaveFunctionMatrixDst.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }

        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(dst);

        if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::DFTPlusU) ||
            (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::HYBRID) ||
            (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::MGGA))
          {
            unsigned int relaventDofs = d_basisOperationsPtr->nRelaventDofs();

            d_BLASWrapperPtr->xcopy(relaventDofs * numberWavefunctions,
                                    src.data(),
                                    1,
                                    d_srcNonLocalTemp.data(),
                                    1);

            d_srcNonLocalTemp.updateGhostValues();
            d_basisOperationsPtr->distribute(d_srcNonLocalTemp);

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
              d_basisOperationsPtr->inverseMassVectorBasisData().data(),
              d_dstNonLocalTemp.data(),
              dst.data(),
              d_mapNodeIdToProcId.data());
          }
      }
    if (!skip1 && !skip2 && !skip3)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::HXCheby(
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
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
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
      (d_ONCVnonLocalOperatorSinglePrec
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
    const dataTypes::numberFP32 scalarCoeffAlpha = dataTypes::numberFP32(1.0),
                                scalarCoeffBeta  = dataTypes::numberFP32(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_ONCVnonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
              d_kPointIndex);
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
#pragma omp critical(hxc_Cconj)
            if (hasNonlocalComponents)
              d_ONCVnonLocalOperatorSinglePrec->applyCconjtransOnX(
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
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec.setValue(0);
            d_ONCVnonLocalOperatorSinglePrec->applyAllReduceOnCconjtransX(
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec, true);
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedEnd();
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
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
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .updateGhostValuesEnd();
            d_ONCVnonLocalOperatorSinglePrec->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_oncvClassPtr->getCouplingMatrixSinglePrec(),
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec,
              true);
          }
      }
    if (!skip3)
      {
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
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
              d_ONCVnonLocalOperatorSinglePrec->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDstSinglePrec.data() +
                  omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                    numberWavefunctions,
                cellRange);
#pragma omp critical(hxc_assembly)
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              d_cellWaveFunctionMatrixDstSinglePrec.data() +
                omp_get_thread_num() * d_cellsBlockSizeHX * numDoFsPerCell *
                  numberWavefunctions,
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }

        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(dst);
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


  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
