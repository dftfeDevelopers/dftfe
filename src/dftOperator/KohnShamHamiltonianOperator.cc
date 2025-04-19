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
// @author Nikhil Kodali, Srinibas Nandi
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
    d_cellsBlockSizeHX   = memorySpace == dftfe::utils::MemorySpace::HOST ?
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
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      tempHamMatrixImagBlock.resize(nDofsPerCell * nDofsPerCell *
                                    d_cellsBlockSizeHamiltonianConstruction);
    for (dftfe::uInt iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrix.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrix[iHamiltonian].resize(nDofsPerCell * nDofsPerCell *
                                                   nCells);
    for (dftfe::uInt iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrixSinglePrec.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrixSinglePrec[iHamiltonian].resize(
        nDofsPerCell * nDofsPerCell * nCells);

    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID, false);
    const dftfe::uInt numberQuadraturePoints =
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
          d_invJacKPointTimesJxWHost.resize(nCells * numberQuadraturePoints * 3,
                                            0.0);
          for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
            {
              auto cellJxWPtr =
                d_basisOperationsPtrHost->JxWBasisData().data() +
                iCell * numberQuadraturePoints;
              const double *kPointCoordinatesPtr =
                kPointCoordinates.data() + 3 * kPointIndex;

              if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                {
                  for (dftfe::uInt iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                           iCell * 9);
                      for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
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
                  for (dftfe::uInt iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
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
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEff(
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
    const dftfe::uInt numberQuadraturePointsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesGradRhoJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacinvJacderExcWithTauJxWHost;
#else
    auto &d_VeffJxWHost = d_VeffJxW;
    auto &d_invJacderExcWithSigmaTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesGradRhoJxW;
    auto &d_invJacinvJacderExcWithTauJxWHost = d_invJacinvJacderExcWithTauJxW;
#endif
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePointsPerCell,
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

    pdexDensitySpinUp.resize(numberQuadraturePointsPerCell, 0.0);
    pdecDensitySpinUp.resize(numberQuadraturePointsPerCell, 0.0);
    pdexDensitySpinDown.resize(numberQuadraturePointsPerCell, 0.0);
    pdecDensitySpinDown.resize(numberQuadraturePointsPerCell, 0.0);

    if (isGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>(3 * numberQuadraturePointsPerCell, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>(3 * numberQuadraturePointsPerCell, 0.0);
      }
    if (isTauMGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          std::vector<double>(numberQuadraturePointsPerCell, 0.0);
        xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          std::vector<double>(numberQuadraturePointsPerCell, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          std::vector<double>(numberQuadraturePointsPerCell, 0.0);
        cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          std::vector<double>(numberQuadraturePointsPerCell, 0.0);
      }

    // The Hamiltonian operator for the MGGA case is dependent on the k point.
    // All the GGA calculations are done inside the condition (kPointIndex == 0)
    for (dftfe::uInt kPointIndex = 0; kPointIndex < d_kPointWeights.size();
         kPointIndex++)
      {
#if defined(DFTFE_WITH_DEVICE)
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          d_halfKSquareTimesDerExcwithTauJxWHost;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          d_derExcwithTauTimesinvJacKpointTimesJxWHost;
#else
        auto &d_halfKSquareTimesDerExcwithTauJxWHost =
          d_halfKSquareTimesDerExcwithTauJxW[kPointIndex];

        auto &d_derExcwithTauTimesinvJacKpointTimesJxWHost =
          d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex];
#endif
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

        auto quadPointsAll = d_basisOperationsPtrHost->quadPoints();

        auto quadWeightsAll = d_basisOperationsPtrHost->JxW();

        for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
          {
            auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                              iCell * numberQuadraturePointsPerCell;
            d_excManagerPtr->getExcSSDFunctionalObj()
              ->computeRhoTauDependentXCData(
                *auxDensityXCRepresentation,
                std::make_pair<dftfe::uInt, dftfe::uInt>(
                  iCell * numberQuadraturePointsPerCell,
                  (iCell + 1) * numberQuadraturePointsPerCell),
                xDataOut,
                cDataOut);

            if (kPointIndex == 0)
              {
                const std::vector<double> &pdexDensitySpinIndex =
                  spinIndex == 0 ? pdexDensitySpinUp : pdexDensitySpinDown;
                const std::vector<double> &pdecDensitySpinIndex =
                  spinIndex == 0 ? pdecDensitySpinUp : pdecDensitySpinDown;

                std::vector<double> pdexSigma;
                std::vector<double> pdecSigma;
                if (isGGA)
                  {
                    pdexSigma =
                      xDataOut[xcRemainderOutputDataAttributes::pdeSigma];
                    pdecSigma =
                      cDataOut[xcRemainderOutputDataAttributes::pdeSigma];
                  }

                std::vector<double> pdexTauSpinIndex;
                std::vector<double> pdecTauSpinIndex;
                if (isTauMGGA)
                  {
                    pdexTauSpinIndex =
                      spinIndex == 0 ?
                        xDataOut
                          [xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                        xDataOut
                          [xcRemainderOutputDataAttributes::pdeTauSpinDown];

                    pdecTauSpinIndex =
                      spinIndex == 0 ?
                        cDataOut
                          [xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                        cDataOut
                          [xcRemainderOutputDataAttributes::pdeTauSpinDown];
                  }

                std::unordered_map<DensityDescriptorDataAttributes,
                                   std::vector<double>>
                                     densityData;
                std::vector<double> &densitySpinUp =
                  densityData[DensityDescriptorDataAttributes::valuesSpinUp];
                std::vector<double> &densitySpinDown =
                  densityData[DensityDescriptorDataAttributes::valuesSpinDown];
                std::vector<double> &gradDensitySpinUp = densityData
                  [DensityDescriptorDataAttributes::gradValuesSpinUp];
                std::vector<double> &gradDensitySpinDown = densityData
                  [DensityDescriptorDataAttributes::gradValuesSpinDown];


                // This applyLocalOperations is necessary because gradRho values
                // are required in the operator
                if (isGGA)
                  auxDensityXCRepresentation->applyLocalOperations(
                    std::make_pair<dftfe::uInt, dftfe::uInt>(
                      iCell * numberQuadraturePointsPerCell,
                      (iCell + 1) * numberQuadraturePointsPerCell),
                    densityData);

                const std::vector<double> &gradDensityXCSpinIndex =
                  spinIndex == 0 ? gradDensitySpinUp : gradDensitySpinDown;
                const std::vector<double> &gradDensityXCOtherSpinIndex =
                  spinIndex == 0 ? gradDensitySpinDown : gradDensitySpinUp;


                const double *tempPhi =
                  phiValues.data() + iCell * numberQuadraturePointsPerCell;

                for (dftfe::uInt iQuad = 0;
                     iQuad < numberQuadraturePointsPerCell;
                     ++iQuad)
                  {
                    d_VeffJxWHost[iCell * numberQuadraturePointsPerCell +
                                  iQuad] =
                      (tempPhi[iQuad] + pdexDensitySpinIndex[iQuad] +
                       pdecDensitySpinIndex[iQuad]) *
                      cellJxWPtr[iQuad];
                  }

                if (isGGA)
                  {
                    if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                      {
                        for (dftfe::uInt iQuad = 0;
                             iQuad < numberQuadraturePointsPerCell;
                             ++iQuad)
                          {
                            const double *inverseJacobiansQuadPtr =
                              d_basisOperationsPtrHost
                                ->inverseJacobiansBasisData()
                                .data() +
                              (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                                 iCell * numberQuadraturePointsPerCell * 9 +
                                   iQuad * 9 :
                                 iCell * 9);
                            const double *gradDensityQuadPtr =
                              gradDensityXCSpinIndex.data() + iQuad * 3;
                            const double *gradDensityOtherQuadPtr =
                              gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                            const double term =
                              (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                               pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                              cellJxWPtr[iQuad];
                            const double termoff = (pdexSigma[iQuad * 3 + 1] +
                                                    pdecSigma[iQuad * 3 + 1]) *
                                                   cellJxWPtr[iQuad];
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
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
                        for (dftfe::uInt iQuad = 0;
                             iQuad < numberQuadraturePointsPerCell;
                             ++iQuad)
                          {
                            const double *inverseJacobiansQuadPtr =
                              d_basisOperationsPtrHost
                                ->inverseJacobiansBasisData()
                                .data() +
                              iCell * 3;
                            const double *gradDensityQuadPtr =
                              gradDensityXCSpinIndex.data() + iQuad * 3;
                            const double *gradDensityOtherQuadPtr =
                              gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                            const double term =
                              (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                               pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                              cellJxWPtr[iQuad];
                            const double termoff = (pdexSigma[iQuad * 3 + 1] +
                                                    pdecSigma[iQuad * 3 + 1]) *
                                                   cellJxWPtr[iQuad];
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              d_invJacderExcWithSigmaTimesGradRhoJxWHost
                                [iCell * numberQuadraturePointsPerCell * 3 +
                                 iQuad * 3 + iDim] =
                                  inverseJacobiansQuadPtr[iDim] *
                                  (2.0 * gradDensityQuadPtr[iDim] * term +
                                   gradDensityOtherQuadPtr[iDim] * termoff);
                          }
                      }
                  } // GGA

                if (isTauMGGA)
                  {
                    if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                      {
                        for (dftfe::uInt iQuad = 0;
                             iQuad < numberQuadraturePointsPerCell;
                             ++iQuad)
                          {
                            const double *inverseJacobiansQuadPtr =
                              d_basisOperationsPtrHost
                                ->inverseJacobiansBasisData()
                                .data() +
                              (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                                 iCell * numberQuadraturePointsPerCell * 9 +
                                   iQuad * 9 :
                                 iCell * 9);
                            const auto jacobianFactorForTauPtr =
                              d_invJacinvJacderExcWithTauJxWHost.data() +
                              iCell * numberQuadraturePointsPerCell * 9 +
                              iQuad * 9;

                            const double termTau = 0.5 *
                                                   (pdexTauSpinIndex[iQuad] +
                                                    pdecTauSpinIndex[iQuad]) *
                                                   cellJxWPtr[iQuad];

                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              {
                                for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                                  {
                                    for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                                      {
                                        jacobianFactorForTauPtr[3 * jDim +
                                                                iDim] +=
                                          inverseJacobiansQuadPtr[3 * kDim +
                                                                  iDim] *
                                          inverseJacobiansQuadPtr[3 * kDim +
                                                                  jDim];
                                      }
                                    jacobianFactorForTauPtr[3 * jDim + iDim] *=
                                      termTau;
                                  }
                              }
                          }
                      }
                    else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                      {
                        for (dftfe::uInt iQuad = 0;
                             iQuad < numberQuadraturePointsPerCell;
                             ++iQuad)
                          {
                            const double *inverseJacobiansQuadPtr =
                              d_basisOperationsPtrHost
                                ->inverseJacobiansBasisData()
                                .data() +
                              iCell * 3;

                            const auto jacobianFactorForTauPtr =
                              d_invJacinvJacderExcWithTauJxWHost.data() +
                              iCell * numberQuadraturePointsPerCell * 9 +
                              iQuad * 9;
                            const double termTau = 0.5 *
                                                   (pdexTauSpinIndex[iQuad] +
                                                    pdecTauSpinIndex[iQuad]) *
                                                   cellJxWPtr[iQuad];

                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              {
                                jacobianFactorForTauPtr[3 * iDim + iDim] +=
                                  inverseJacobiansQuadPtr[iDim] *
                                  inverseJacobiansQuadPtr[iDim] * termTau;
                              }
                          }
                      }
                  } // TauMGGA
              }     // kpointIndex=0
            if (isTauMGGA &&
                std::is_same<dataTypes::number, std::complex<double>>::value)
              {
                std::vector<double> pdexTauSpinIndex =
                  spinIndex == 0 ?
                    xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                    xDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown];
                std::vector<double> pdecTauSpinIndex =
                  spinIndex == 0 ?
                    cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] :
                    cDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown];
                const double *kPointCoords =
                  d_kPointCoordinates.data() + 3 * kPointIndex;

                for (dftfe::uInt iQuad = 0;
                     iQuad < numberQuadraturePointsPerCell;
                     ++iQuad)
                  {
                    const double kSquareTimesHalf =
                      0.5 * (kPointCoords[0] * kPointCoords[0] +
                             kPointCoords[1] * kPointCoords[1] +
                             kPointCoords[2] * kPointCoords[2]);
                    d_halfKSquareTimesDerExcwithTauJxWHost
                      [iCell * numberQuadraturePointsPerCell + iQuad] =
                        kSquareTimesHalf *
                        (pdexTauSpinIndex[iQuad] + pdecTauSpinIndex[iQuad]) *
                        cellJxWPtr[iQuad];
                  }

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
                          {
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              {
                                d_derExcwithTauTimesinvJacKpointTimesJxWHost
                                  [iCell * numberQuadraturePointsPerCell * 3 +
                                   iQuad * 3 + iDim] +=
                                  -0.5 *
                                  inverseJacobiansQuadPtr[3 * jDim + iDim] *
                                  kPointCoords[jDim] * cellJxWPtr[iQuad] *
                                  (pdexTauSpinIndex[iQuad] +
                                   pdecTauSpinIndex[iQuad]);
                              }
                          }
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
                          {
                            d_derExcwithTauTimesinvJacKpointTimesJxWHost
                              [iCell * numberQuadraturePointsPerCell * 3 +
                               iQuad * 3 + iDim] =
                                -0.5 * inverseJacobiansQuadPtr[iDim] *
                                kPointCoords[iDim] * cellJxWPtr[iQuad] *
                                (pdexTauSpinIndex[iQuad] +
                                 pdecTauSpinIndex[iQuad]);
                          }
                      }
                  }
              } // TauMGGA
          }     // cell loop
#if defined(DFTFE_WITH_DEVICE)
        d_halfKSquareTimesDerExcwithTauJxW[kPointIndex].resize(
          d_halfKSquareTimesDerExcwithTauJxWHost.size());
        d_halfKSquareTimesDerExcwithTauJxW[kPointIndex].copyFrom(
          d_halfKSquareTimesDerExcwithTauJxWHost);
        d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex].resize(
          d_derExcwithTauTimesinvJacKpointTimesJxWHost.size());
        d_derExcwithTauTimesinvJacKpointTimesJxW[kPointIndex].copyFrom(
          d_derExcwithTauTimesinvJacKpointTimesJxWHost);
#endif
        if (d_dftParamsPtr->XCType.substr(0, 3) == "GGA")
          {
            break;
          }
      } // kpoint loop
#if defined(DFTFE_WITH_DEVICE)
    d_VeffJxW.resize(d_VeffJxWHost.size());
    d_VeffJxW.copyFrom(d_VeffJxWHost);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
    d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost);
    d_invJacinvJacderExcWithTauJxW.resize(
      d_invJacinvJacderExcWithTauJxWHost.size());
    d_invJacinvJacderExcWithTauJxW.copyFrom(d_invJacinvJacderExcWithTauJxWHost);
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::setVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                     &vKS_quadValues,
    const dftfe::uInt spinIndex)
  {
    const dftfe::uInt spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const dftfe::uInt totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const dftfe::uInt numberQuadraturePoints =
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

    for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      {
        auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                          iCell * numberQuadraturePoints;
        for (dftfe::uInt qPoint = 0; qPoint < numberQuadraturePoints; ++qPoint)
          {
            // TODO extend to spin polarised case
            d_VeffJxWHost[qPoint + iCell * numberQuadraturePoints] =
              vKS_quadValues[0][qPoint + iCell * numberQuadraturePoints] *
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
  KohnShamHamiltonianOperator<memorySpace>::computeVEffExternalPotCorr(
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
  KohnShamHamiltonianOperator<memorySpace>::setVEffExternalPotCorrToZero()
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
  KohnShamHamiltonianOperator<memorySpace>::reinitkPointSpinIndex(
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
    const dftfe::uInt numWaveFunctions)
  {
    const dftfe::uInt nCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
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
    const dftfe::uInt numVectors,
    const dftfe::uInt index)
  {
    return d_basisOperationsPtr->getMultiVector(numVectors, index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getScratchFEMultivectorSinglePrec(
    const dftfe::uInt numVectors,
    const dftfe::uInt index)
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

        bool isGradDensityDataDependent =
          (d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);
        const bool isTauMGGA =
          (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
           ExcFamilyType::TauMGGA);
        if (isGradDensityDataDependent)
          d_basisOperationsPtr->computeWeightedCellNjGradNiPlusNiGradNjMatrix(
            cellRange,
            d_invJacderExcWithSigmaTimesGradRhoJxW,
            tempHamMatrixRealBlock);
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
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
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
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
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
  KohnShamHamiltonianOperator<memorySpace>::HXWithLowdinOrthonormalisedInput(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
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
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<dftfe::uInt, dftfe::uInt> cellRange(
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
    for (dftfe::uInt iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
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


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::overlapMatrixTimesX(
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
  KohnShamHamiltonianOperator<memorySpace>::overlapInverseMatrixTimesX(
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
  KohnShamHamiltonianOperator<memorySpace>::overlapInverseMatrixTimesX(
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
      (d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
       0) &&
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
              d_ONCVnonLocalOperator->initialiseOperatorActionOnX(
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
              d_ONCVnonLocalOperator->applyCconjtransOnX(
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
              d_ONCVnonLocalOperator->applyCOnVCconjtransX(
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
      (d_ONCVnonLocalOperatorSinglePrec
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
            d_ONCVnonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
              d_kPointIndex);
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


  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
