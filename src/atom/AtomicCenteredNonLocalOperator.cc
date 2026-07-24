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
// @author Kartick Ramakrishnan, Sambit Das, Phani Motamarri, Vishal Subramanian
//
#include <dftfe/config.h>
#include <dftfe/AtomicCenteredNonLocalOperator.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <dftfe/AtomicCenteredNonLocalOperatorKernelsDevice.h>
#  include <dftfe/DeviceTypeConfig.h>
#  include <dftfe/DeviceKernelLauncherHelpers.h>
#  include <dftfe/DeviceAPICalls.h>
#  include <dftfe/DeviceDataTypeOverloads.h>
#endif
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    AtomicCenteredNonLocalOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperatorPtr,
      std::shared_ptr<AtomCenteredSphericalFunctionContainer>
                      atomCenteredSphericalFunctionContainer,
      const MPI_Comm &mpi_comm_parent,
      const bool      memOptMode,
      const bool      floatingNuclearCharges,
      const bool      useGlobalCMatrix,
      const bool      computeIonForces,
      const bool      computeCellStress)
    : d_mpi_communicator(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , d_n_mpi_processes(
        dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isMallocCalled(false)
  {
    d_BLASWrapperPtr   = BLASWrapperPtr;
    d_basisOperatorPtr = basisOperatorPtr;
    d_atomCenteredSphericalFunctionContainer =
      atomCenteredSphericalFunctionContainer;
    d_maxSingleAtomContribution = d_atomCenteredSphericalFunctionContainer
                                    ->getMaximumNumberOfSphericalFunctions();
    d_memoryOptMode          = memOptMode;
    d_floatingNuclearCharges = floatingNuclearCharges;
    d_useGlobalCMatrix       = useGlobalCMatrix;
#if defined(DFTFE_WITH_DEVICE)
    d_cellsBlockSize  = 0;
    d_numCellBatches  = 0;
    d_wfcStartPointer = NULL;
#endif
    d_computeIonForces  = computeIonForces && floatingNuclearCharges;
    d_computeCellStress = computeCellStress && floatingNuclearCharges;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::initKpoints(
    const std::vector<double> &kPointWeights,
    const std::vector<double> &kPointCoordinates)
  {
    d_kPointWeights     = kPointWeights;
    d_kPointCoordinates = kPointCoordinates;
    d_totalNonLocalEntries =
      d_atomCenteredSphericalFunctionContainer
        ->getTotalNumberOfSphericalFunctionsInCurrentProcessor();
    std::vector<dftfe::uInt> iElemNonLocalToElemIndexMap;
    d_atomCenteredSphericalFunctionContainer
      ->getTotalAtomsAndNonLocalElementsInCurrentProcessor(
        d_totalAtomsInCurrentProc,
        d_totalNonlocalElems,
        d_numberCellsForEachAtom,
        d_numberCellsAccumNonLocalAtoms,
        iElemNonLocalToElemIndexMap);
    d_iElemNonLocalToElemIndexMap.resize(d_totalNonlocalElems);
    d_iElemNonLocalToElemIndexMap.copyFrom(iElemNonLocalToElemIndexMap);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseOperatorActionOnX(
      dftfe::uInt                         kPointIndex,
      const nonLocalContractionVectorType NonLocalContractionVectorType)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_kPointIndex = kPointIndex;

        const std::vector<dftfe::uInt> atomIdsInProcessor =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            dftfe::uInt atomId = atomIdsInProcessor[iAtom];
            if (NonLocalContractionVectorType ==
                nonLocalContractionVectorType::CconjTransX)
              d_sphericalFnTimesWavefunMatrix[atomId].setValue(0.0);
            else if (NonLocalContractionVectorType ==
                     nonLocalContractionVectorType::CRconjTransX)
              d_sphericalFnTimesXTimesWavefunMatrix[atomId].setValue(0.0);
            else if (NonLocalContractionVectorType ==
                     nonLocalContractionVectorType::DconjTransX)
              d_sphericalFnTimesGradientWavefunMatrix[atomId].setValue(0.0);
            else if (NonLocalContractionVectorType ==
                     nonLocalContractionVectorType::DDyadicRconjTransX)
              d_sphericalFnTimesGradientWavefunDyadicXMatrix[atomId].setValue(
                0.0);
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_kPointIndex = kPointIndex;
        if (NonLocalContractionVectorType ==
            nonLocalContractionVectorType::CconjTransX)
          {
            if (!d_useGlobalCMatrix)
              {
                d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
                  .copyFrom(
                    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate,
                    d_totalNonlocalElems * d_numberNodesPerElement *
                      d_maxSingleAtomContribution,
                    d_kPointIndex * d_totalNonlocalElems *
                      d_numberNodesPerElement * d_maxSingleAtomContribution,
                    0);
                d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
                  .copyFrom(
                    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose,
                    d_totalNonlocalElems * d_numberNodesPerElement *
                      d_maxSingleAtomContribution,
                    d_kPointIndex * d_totalNonlocalElems *
                      d_numberNodesPerElement * d_maxSingleAtomContribution,
                    0);
              }
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::CRconjTransX)
          {
            d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugateDevice
              .copyFrom(
                d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugate,
                d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 3,
                d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 3,
                0);
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DconjTransX)
          {
            d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
              .copyFrom(
                d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate,
                d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 3,
                d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 3,
                0);
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DDyadicRconjTransX)
          {
            d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugateDevice
              .copyFrom(
                d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugate,
                d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 9,
                d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
                  d_maxSingleAtomContribution * 9,
                0);
          }
      }
#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::computeCMatrixEntries(
    std::shared_ptr<dftfe::basis::FEBasisOperations<
      dataTypes::number,
      double,
      dftfe::utils::MemorySpace::HOST>> basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                      BLASWrapperHostPtr,
    const dftfe::uInt quadratureIndex)
  {
    d_locallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const dftfe::uInt numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const dftfe::uInt numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    d_numberNodesPerElement    = basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numCells = d_locallyOwnedCells;

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = basisOperationsPtr->quadPoints();
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST>
                                    JxwVector = basisOperationsPtr->JxW();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomCenteredSphericalFunctionContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomCenteredSphericalFunctionContainer
        ->getPeriodicImageCoordinatesList();
    const dftfe::uInt maxkPoints = d_kPointWeights.size();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomCenteredSphericalFunctionContainer->getSphericalFunctions();
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    if (d_computeIonForces)
      {
        d_DMatrixEntriesConjugate.clear();
        d_DMatrixEntriesConjugate.resize(numberAtomsOfInterest);
      }
    if (d_computeCellStress)
      {
        d_CRMatrixEntriesConjugate.clear();
        d_CRMatrixEntriesConjugate.resize(numberAtomsOfInterest);
        d_DDyadicRMatrixEntriesConjugate.clear();
        d_DDyadicRMatrixEntriesConjugate.resize(numberAtomsOfInterest);
      }

    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);
    d_atomCenteredKpointIndexedSphericalFnQuadValues.clear();
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.clear();
    d_cellIdToAtomIdsLocalCompactSupportMap.clear();
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    d_nonTrivialSphericalFnPerCell.clear();
    d_nonTrivialSphericalFnPerCell.resize(numCells, 0);

    d_nonTrivialSphericalFnsCellStartIndex.clear();
    d_nonTrivialSphericalFnsCellStartIndex.resize(numCells, 0);

    d_atomIdToNonTrivialSphericalFnCellStartIndex.clear();
    std::map<dftfe::uInt, std::vector<dftfe::uInt>>
                             globalAtomIdToNonTrivialSphericalFnsCellStartIndex;
    std::vector<dftfe::uInt> accumTemp(numCells, 0);
    // Loop over atoms to determine sizes of various vectors for forces
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        dftfe::uInt       atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom] =
          std::vector<dftfe::uInt>(numCells, 0);
        globalAtomIdToNonTrivialSphericalFnsCellStartIndex[atomId] =
          std::vector<dftfe::uInt>(numCells, 0);
        for (dftfe::Int iElemComp = 0;
             iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const dftfe::uInt elementId =
              elementIndexesInAtomCompactSupport[iElemComp];

            d_cellIdToAtomIdsLocalCompactSupportMap[elementId].push_back(iAtom);

            d_nonTrivialSphericalFnPerCell[elementId] += numSphericalFunctions;
            d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom][elementId] =
              accumTemp[elementId];
            globalAtomIdToNonTrivialSphericalFnsCellStartIndex
              [atomId][elementId] = accumTemp[elementId];
            accumTemp[elementId] += numSphericalFunctions;
          }
      }

    d_sumNonTrivialSphericalFnOverAllCells =
      std::accumulate(d_nonTrivialSphericalFnPerCell.begin(),
                      d_nonTrivialSphericalFnPerCell.end(),
                      0);

    dftfe::uInt accumNonTrivialSphericalFnCells = 0;
    for (dftfe::Int iElem = 0; iElem < numCells; ++iElem)
      {
        d_nonTrivialSphericalFnsCellStartIndex[iElem] =
          accumNonTrivialSphericalFnCells;
        accumNonTrivialSphericalFnCells +=
          d_nonTrivialSphericalFnPerCell[iElem];
      }
    if (!d_floatingNuclearCharges)
      {
        d_atomCenteredKpointIndexedSphericalFnQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints,
          ValueType(0));
        d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints * 3,
          ValueType(0));
      }

    std::vector<std::vector<dftfe::uInt>> sphericalFnKetTimesVectorLocalIds;
    sphericalFnKetTimesVectorLocalIds.clear();
    sphericalFnKetTimesVectorLocalIds.resize(d_totalAtomsInCurrentProc);
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


        for (dftfe::uInt alpha = 0; alpha < numSphericalFunctions; ++alpha)
          {
            dftfe::uInt globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;

            dftfe::uInt localId = d_SphericalFunctionKetTimesVectorPar[0]
                                    .get_partitioner()
                                    ->global_to_local(globalId);
            sphericalFnKetTimesVectorLocalIds[iAtom].push_back(localId);
          }
      }

    d_sphericalFnTimesVectorFlattenedVectorLocalIds.clear();
    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.clear();
    for (dftfe::uInt ielem = 0; ielem < numCells; ++ielem)
      {
        for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            bool isNonTrivial = false;
            for (dftfe::uInt i = 0;
                 i < d_cellIdToAtomIdsLocalCompactSupportMap[ielem].size();
                 i++)
              if (d_cellIdToAtomIdsLocalCompactSupportMap[ielem][i] == iAtom)
                {
                  isNonTrivial = true;
                  break;
                }
            if (isNonTrivial)
              {
                dftfe::uInt       atomId = atomIdsInProc[iAtom];
                const dftfe::uInt Znum   = atomicNumber[atomId];
                const dftfe::uInt numSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (dftfe::uInt iAlpha = 0; iAlpha < numSphericalFunctions;
                     ++iAlpha)
                  {
                    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.push_back(
                      ielem);
                    d_sphericalFnTimesVectorFlattenedVectorLocalIds.push_back(
                      sphericalFnKetTimesVectorLocalIds[iAtom][iAlpha]);
                  }
              }
          }
      }
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        dftfe::uInt      ChargeId = atomIdsInProc[iAtom];
        dealii::Point<3> nuclearCoordinates(atomCoordinates[3 * ChargeId + 0],
                                            atomCoordinates[3 * ChargeId + 1],
                                            atomCoordinates[3 * ChargeId + 2]);

        const dftfe::uInt   atomId = ChargeId;
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const dftfe::uInt Znum = atomicNumber[ChargeId];
        const dftfe::uInt NumRadialSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt NumTotalSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[ChargeId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();

        dftfe::uInt imageIdsSize = imageCoordinates.size() / 3;

        if (numberElementsInAtomCompactSupport > 0)
          {
            d_CMatrixEntriesConjugate[ChargeId].resize(
              numberElementsInAtomCompactSupport);
            d_CMatrixEntriesTranspose[ChargeId].resize(
              numberElementsInAtomCompactSupport);
            if (d_computeIonForces)
              {
                d_DMatrixEntriesConjugate[ChargeId].resize(
                  numberElementsInAtomCompactSupport);
              }
            if (d_computeCellStress)
              {
                d_CRMatrixEntriesConjugate[ChargeId].resize(
                  numberElementsInAtomCompactSupport);
                d_DDyadicRMatrixEntriesConjugate[ChargeId].resize(
                  numberElementsInAtomCompactSupport);
              }
          }
        const dftfe::uInt nCellsPerBatch = 4;
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>
          sphericalFunctionBasisTimesJxWHost(nCellsPerBatch *
                                               NumTotalSphericalFunctions *
                                               numberQuadraturePoints *
                                               maxkPoints,
                                             0.0);
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>
          sphericalFunctionBasisWithDistanceTimesJxWHost(
            nCellsPerBatch * NumTotalSphericalFunctions * 3 *
              numberQuadraturePoints * maxkPoints,
            0.0);
        std::vector<dataTypes::number>
          inverseJacobianTimesGradientShapeFnForChargeId;
        if (d_computeIonForces || d_computeCellStress)
          {
            inverseJacobianTimesGradientShapeFnForChargeId.clear();
            inverseJacobianTimesGradientShapeFnForChargeId.resize(
              numberElementsInAtomCompactSupport * 3 * numberQuadraturePoints *
              d_numberNodesPerElement);
          }

        const char              transA = 'N', transB = 'N';
        const dataTypes::number scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
        const dftfe::uInt       inc = 1;
        const dftfe::uInt       n =
          nCellsPerBatch * maxkPoints * NumTotalSphericalFunctions;
        const dftfe::uInt m = d_numberNodesPerElement;
        const dftfe::uInt k = numberQuadraturePoints;
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          projectorTimesXMatrices;
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          gradientProjectorMatrices;
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          gradientProjectorDyadicXMatrices;

        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          projectorMatrices;
        dftfe::uInt kptBatch               = 1;
        dftfe::uInt projectorMatrixSizeOld = 0;
#if defined(DFTFE_WITH_DEVICE)
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          sphericalFunctionBasisTimesJxW,
          sphericalFunctionBasisWithDistanceTimesJxW;
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>
          projectorTimesXMatricesHost, gradientProjectorDyadicXMatricesHost,
          projectorMatricesHost, gradientProjectorMatricesHost;
        projectorMatricesHost.resize(m * n, 0.0);

        if (d_computeCellStress)
          {
            projectorTimesXMatricesHost.resize(3 * m * n, 0.0);
            gradientProjectorDyadicXMatricesHost.resize(9 * m * n, 0.0);
          }
        if (d_computeIonForces)
          gradientProjectorMatricesHost.resize(3 * m * n, 0.0);

        // sphericalFunctionBasisTimesJxW(WithDistance) are separate
        // device-resident staging buffers here, refilled per k-point batch
        // via copyFrom from the full *Host buffers below, so they are sized
        // to a single k-point batch's worth of data.
        sphericalFunctionBasisTimesJxW.resize(nCellsPerBatch *
                                                numberQuadraturePoints *
                                                kptBatch *
                                                NumTotalSphericalFunctions,
                                              0.0);
        if (d_computeCellStress)
          sphericalFunctionBasisWithDistanceTimesJxW.resize(
            3 * nCellsPerBatch * numberQuadraturePoints * kptBatch *
              NumTotalSphericalFunctions,
            0.0);
#else
        auto &sphericalFunctionBasisTimesJxW =
          sphericalFunctionBasisTimesJxWHost;
        auto &sphericalFunctionBasisWithDistanceTimesJxW =
          sphericalFunctionBasisWithDistanceTimesJxWHost;
        auto &projectorMatricesHost         = projectorMatrices;
        auto &projectorTimesXMatricesHost   = projectorTimesXMatrices;
        auto &gradientProjectorMatricesHost = gradientProjectorMatrices;
        auto &gradientProjectorDyadicXMatricesHost =
          gradientProjectorDyadicXMatrices;
        // sphericalFunctionBasisTimesJxW(WithDistance) alias the *Host
        // buffers here, which are already sized to hold all maxkPoints'
        // worth of data (see their construction above) — do NOT resize them
        // down to a single k-point batch like the device path above, or the
        // later copyFrom calls will index past the aliased buffer's actual
        // size for any iKpt > 0.
#endif
        for (dftfe::Int iElemComp = 0;
             iElemComp < numberElementsInAtomCompactSupport;
             iElemComp += nCellsPerBatch)
          {
            std::vector<dftfe::uInt> cellIndexes;
            cellIndexes.clear();
            sphericalFunctionBasisTimesJxWHost.setValue(0.0);
            if (d_computeCellStress)
              sphericalFunctionBasisWithDistanceTimesJxWHost.setValue(0.0);
            for (dftfe::Int iCell = 0; iCell < nCellsPerBatch; iCell++)
              {
                if ((iElemComp + iCell) >= numberElementsInAtomCompactSupport)
                  break;
                const dftfe::uInt elementIndex =
                  elementIndexesInAtomCompactSupport[iElemComp + iCell];
                cellIndexes.push_back(elementIndex);
                const double *quadPointsInElement =
                  quadraturePointsVector.data() +
                  elementIndex * numberQuadraturePoints * 3;
                const dataTypes::number *JxwInElement =
                  JxwVector.data() + elementIndex * numberQuadraturePoints;
                for (dftfe::uInt alpha = 0; alpha < NumRadialSphericalFunctions;
                     ++alpha)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                      sphericalFunction.find(std::make_pair(Znum, alpha))
                        ->second;
                    dftfe::uInt lQuantumNumber = sphFn->getQuantumNumberl();
                    const dftfe::uInt startIndex =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalSphericalFunctionIndexStart(Znum, alpha);
                    dftfe::uInt endIndex = startIndex + 2 * lQuantumNumber + 1;
                    std::vector<dataTypes::number> sphericalFunctionBasis(
                      maxkPoints * numberQuadraturePoints *
                        (2 * lQuantumNumber + 1),
                      ValueType(0.0));
                    std::vector<dataTypes::number>
                      sphericalFunctionBasisTimesImageDist(
                        maxkPoints * numberQuadraturePoints *
                          (2 * lQuantumNumber + 1) * 3,
                        ValueType(0.0));
                    for (dftfe::Int iImageAtomCount = 0;
                         iImageAtomCount < imageIdsSize;
                         ++iImageAtomCount)
                      {
                        dealii::Point<3> chargePoint(0.0, 0.0, 0.0);
                        if (iImageAtomCount == 0)
                          {
                            chargePoint = nuclearCoordinates;
                          }
                        else
                          {
                            chargePoint[0] =
                              imageCoordinates[3 * iImageAtomCount + 0];
                            chargePoint[1] =
                              imageCoordinates[3 * iImageAtomCount + 1];
                            chargePoint[2] =
                              imageCoordinates[3 * iImageAtomCount + 2];
                          }
                        double x[3], pointMinusLatticeVector[3];
                        double sphericalHarmonicVal, radialVal,
                          sphericalFunctionValue;
                        double r, theta, phi, angle;

                        for (dftfe::uInt iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          {
                            x[0] = quadPointsInElement[3 * iQuadPoint + 0] -
                                   chargePoint[0];
                            x[1] = quadPointsInElement[3 * iQuadPoint + 1] -
                                   chargePoint[1];
                            x[2] = quadPointsInElement[3 * iQuadPoint + 2] -
                                   chargePoint[2];
                            sphericalHarmonicUtils::convertCartesianToSpherical(
                              x, r, theta, phi);
                            if (r <= sphFn->getRadialCutOff())
                              {
                                radialVal = sphFn->getRadialValue(r);
                                dftfe::uInt tempIndex = 0;
                                for (dftfe::Int mQuantumNumber =
                                       dftfe::Int(-lQuantumNumber);
                                     mQuantumNumber <=
                                     dftfe::Int(lQuantumNumber);
                                     mQuantumNumber++)
                                  {
                                    sphericalHarmonicUtils::
                                      getSphericalHarmonicVal(
                                        theta,
                                        phi,
                                        lQuantumNumber,
                                        mQuantumNumber,
                                        sphericalHarmonicVal);

                                    sphericalFunctionValue =
                                      radialVal * sphericalHarmonicVal;



                                    //
                                    // kpoint loop
                                    //
#ifdef USE_COMPLEX
                                    pointMinusLatticeVector[0] =
                                      x[0] + nuclearCoordinates[0];
                                    pointMinusLatticeVector[1] =
                                      x[1] + nuclearCoordinates[1];
                                    pointMinusLatticeVector[2] =
                                      x[2] + nuclearCoordinates[2];
                                    for (dftfe::Int kPoint = 0;
                                         kPoint < maxkPoints;
                                         ++kPoint)
                                      {
                                        angle =
                                          d_kPointCoordinates[3 * kPoint + 0] *
                                            pointMinusLatticeVector[0] +
                                          d_kPointCoordinates[3 * kPoint + 1] *
                                            pointMinusLatticeVector[1] +
                                          d_kPointCoordinates[3 * kPoint + 2] *
                                            pointMinusLatticeVector[2];
                                        dataTypes::number tempValue =
                                          dataTypes::number(
                                            cos(angle) * sphericalFunctionValue,
                                            -sin(angle) *
                                              sphericalFunctionValue);

                                        sphericalFunctionBasis
                                          [kPoint * numberQuadraturePoints *
                                             (2 * lQuantumNumber + 1) +
                                           tempIndex * numberQuadraturePoints +
                                           iQuadPoint] += tempValue;
                                        sphericalFunctionBasisTimesJxWHost
                                          [iCell * NumTotalSphericalFunctions *
                                             numberQuadraturePoints +
                                           kPoint * NumTotalSphericalFunctions *
                                             nCellsPerBatch *
                                             numberQuadraturePoints +
                                           (startIndex + tempIndex) *
                                             numberQuadraturePoints +
                                           iQuadPoint] +=
                                          tempValue *
                                          std::real(JxwInElement[iQuadPoint]);
                                        for (dftfe::uInt iDim = 0; iDim < 3;
                                             ++iDim)
                                          sphericalFunctionBasisTimesImageDist
                                            [kPoint * numberQuadraturePoints *
                                               (2 * lQuantumNumber + 1) * 3 +
                                             tempIndex *
                                               numberQuadraturePoints * 3 +
                                             iQuadPoint * 3 + iDim] +=
                                            tempValue * x[iDim];
                                        for (dftfe::uInt iDim = 0; iDim < 3;
                                             ++iDim)
                                          sphericalFunctionBasisWithDistanceTimesJxWHost
                                            [kPoint *
                                               NumTotalSphericalFunctions *
                                               nCellsPerBatch *
                                               numberQuadraturePoints * 3 +
                                             iCell *
                                               NumTotalSphericalFunctions *
                                               numberQuadraturePoints * 3 +
                                             (startIndex + tempIndex) *
                                               numberQuadraturePoints * 3 +
                                             iDim * numberQuadraturePoints +
                                             iQuadPoint] +=
                                            tempValue * x[iDim] *
                                            std::real(JxwInElement[iQuadPoint]);
                                      } // k-Point Loop
#else
                                    sphericalFunctionBasis
                                      [tempIndex * numberQuadraturePoints +
                                       iQuadPoint] += sphericalFunctionValue;
                                    sphericalFunctionBasisTimesJxWHost
                                      [iCell * NumTotalSphericalFunctions *
                                         numberQuadraturePoints +
                                       (startIndex + tempIndex) *
                                         numberQuadraturePoints +
                                       iQuadPoint] += sphericalFunctionValue *
                                                      JxwInElement[iQuadPoint];
                                    for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                                      sphericalFunctionBasisTimesImageDist
                                        [tempIndex * numberQuadraturePoints *
                                           3 +
                                         iQuadPoint * 3 + iDim] +=
                                        sphericalFunctionValue * x[iDim];
                                    for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                                      sphericalFunctionBasisWithDistanceTimesJxWHost
                                        [iCell * NumTotalSphericalFunctions *
                                           numberQuadraturePoints * 3 +
                                         (startIndex + tempIndex) *
                                           numberQuadraturePoints * 3 +
                                         iDim * numberQuadraturePoints +
                                         iQuadPoint] +=
                                        sphericalFunctionValue * x[iDim] *
                                        (JxwInElement[iQuadPoint]);

#endif
                                    tempIndex++;
                                  } // Angular momentum m loop
                              }     // inside r <= Rmax

                          } // quad loop

                      } // image atom loop
                    const dftfe::uInt startIndex1 =
                      d_nonTrivialSphericalFnsCellStartIndex
                        [elementIndex]; // extract the location of first
                                        // projector in the elementIndex
                    const dftfe::uInt startIndex2 =
                      globalAtomIdToNonTrivialSphericalFnsCellStartIndex
                        [ChargeId][elementIndex]; // extract the location of the
                                                  // ChargeId's first projector
                                                  // in the cell
                    if (!d_floatingNuclearCharges)
                      {
                        for (dftfe::Int kPoint = 0; kPoint < maxkPoints;
                             ++kPoint)
                          {
                            for (dftfe::uInt tempIndex = startIndex;
                                 tempIndex < endIndex;
                                 tempIndex++)
                              {
                                for (dftfe::Int iQuadPoint = 0;
                                     iQuadPoint < numberQuadraturePoints;
                                     ++iQuadPoint)
                                  d_atomCenteredKpointIndexedSphericalFnQuadValues
                                    [kPoint *
                                       d_sumNonTrivialSphericalFnOverAllCells *
                                       numberQuadraturePoints +
                                     startIndex1 * numberQuadraturePoints +
                                     (startIndex2 + tempIndex) *
                                       numberQuadraturePoints +
                                     iQuadPoint] = sphericalFunctionBasis
                                      [kPoint * numberQuadraturePoints *
                                         (2 * lQuantumNumber + 1) +
                                       (tempIndex - startIndex) *
                                         numberQuadraturePoints +
                                       iQuadPoint];

                                for (dftfe::Int iQuadPoint = 0;
                                     iQuadPoint < numberQuadraturePoints;
                                     ++iQuadPoint)
                                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                                    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                                      [kPoint *
                                         d_sumNonTrivialSphericalFnOverAllCells *
                                         numberQuadraturePoints * 3 +
                                       startIndex1 * numberQuadraturePoints *
                                         3 +
                                       (startIndex2 + tempIndex) *
                                         numberQuadraturePoints * 3 +
                                       iQuadPoint * 3 + iDim] =
                                        sphericalFunctionBasisTimesImageDist
                                          [kPoint * numberQuadraturePoints *
                                             (2 * lQuantumNumber + 1) * 3 +
                                           (tempIndex - startIndex) *
                                             numberQuadraturePoints * 3 +
                                           iQuadPoint * 3 + iDim];
                              } // tempIndex
                          }
                      }
                  } // alpha loop
              }     // iCell



            for (dftfe::uInt iKpt = 0; iKpt < maxkPoints; iKpt += kptBatch)
              {
                for (dftfe::uInt iCell = 0; iCell < cellIndexes.size();
                     iCell += nCellsPerBatch)
                  {
                    dftfe::uInt cellBatchSize =
                      std::min(dftfe::uInt(nCellsPerBatch),
                               dftfe::uInt(cellIndexes.size() - iCell));
                    dftfe::uInt projectorMatrixSize =
                      cellBatchSize * kptBatch * NumTotalSphericalFunctions *
                      d_numberNodesPerElement;
                    if (projectorMatrixSize != projectorMatrixSizeOld)
                      {
                        projectorMatrices.resize(projectorMatrixSize);
                        gradientProjectorMatrices.resize(3 *
                                                         projectorMatrixSize);
                        projectorTimesXMatrices.resize(3 * projectorMatrixSize);
                        gradientProjectorDyadicXMatrices.resize(
                          9 * projectorMatrixSize);
                        projectorMatrixSizeOld = projectorMatrixSize;
                      }
                    dftfe::uInt srcOffset = iKpt * nCellsPerBatch *
                                            NumTotalSphericalFunctions *
                                            numberQuadraturePoints;
                    sphericalFunctionBasisTimesJxW.copyFrom(
                      sphericalFunctionBasisTimesJxWHost,
                      cellBatchSize * numberQuadraturePoints * kptBatch *
                        NumTotalSphericalFunctions,
                      srcOffset,
                      0);
                    if (d_computeCellStress)
                      {
                        sphericalFunctionBasisWithDistanceTimesJxW.copyFrom(
                          sphericalFunctionBasisWithDistanceTimesJxWHost,
                          cellBatchSize * numberQuadraturePoints * kptBatch *
                            NumTotalSphericalFunctions * 3,
                          3 * srcOffset,
                          0);
                      }

                    d_basisOperatorPtr->reinit(NumTotalSphericalFunctions,
                                               cellIndexes.size(),
                                               quadratureIndex,
                                               false,
                                               false,
                                               true);
                    d_basisOperatorPtr
                      ->computeScalarFieldTimesShapeFunctionIntegral(
                        cellIndexes,
                        kptBatch,
                        NumTotalSphericalFunctions,
                        cellIndexes.size(),
                        0,
                        sphericalFunctionBasisTimesJxW,
                        projectorMatrices);
                    if (d_computeIonForces)
                      d_basisOperatorPtr
                        ->computeScalarFieldTimesGradientShapeFunctionIntegral(
                          cellIndexes,
                          kptBatch,
                          NumTotalSphericalFunctions,
                          cellIndexes.size(),
                          0,
                          sphericalFunctionBasisTimesJxW,
                          gradientProjectorMatrices);
                    if (d_computeCellStress)
                      {
                        d_basisOperatorPtr
                          ->computeVectorFieldDyadicGradientShapeFunctionIntegral(
                            cellIndexes,
                            kptBatch,
                            NumTotalSphericalFunctions,
                            cellIndexes.size(),
                            0,
                            sphericalFunctionBasisWithDistanceTimesJxW,
                            gradientProjectorDyadicXMatrices);
                        d_basisOperatorPtr
                          ->computeScalarFieldTimesShapeFunctionIntegral(
                            cellIndexes,
                            kptBatch,
                            NumTotalSphericalFunctions * 3,
                            cellIndexes.size(),
                            0,
                            sphericalFunctionBasisWithDistanceTimesJxW,
                            projectorTimesXMatrices);
                      }
                    dftfe::uInt dstOffset = iKpt * nCellsPerBatch *
                                            NumTotalSphericalFunctions *
                                            d_numberNodesPerElement;
#if defined(DFTFE_WITH_DEVICE)
                    projectorMatricesHost.copyFrom(projectorMatrices,
                                                   projectorMatrixSize,
                                                   0,
                                                   dstOffset);

                    if (d_computeIonForces)
                      {
                        gradientProjectorMatricesHost.copyFrom(
                          gradientProjectorMatrices,
                          3 * projectorMatrixSize,
                          0,
                          3 * dstOffset);
                      }
                    if (d_computeCellStress)
                      {
                        projectorTimesXMatricesHost.copyFrom(
                          projectorTimesXMatrices,
                          3 * projectorMatrixSize,
                          0,
                          3 * dstOffset);
                        gradientProjectorDyadicXMatricesHost.copyFrom(
                          gradientProjectorDyadicXMatrices,
                          9 * projectorMatrixSize,
                          0,
                          9 * dstOffset);
                      }
#endif
                  }
              }



            for (dftfe::Int iCell = 0; iCell < nCellsPerBatch; iCell++)
              {
                if (iElemComp + iCell >= numberElementsInAtomCompactSupport)
                  break;
                d_CMatrixEntriesConjugate[ChargeId][iElemComp + iCell].resize(
                  d_numberNodesPerElement * NumTotalSphericalFunctions *
                    maxkPoints,
                  ValueType(0.0));
                d_CMatrixEntriesTranspose[ChargeId][iElemComp + iCell].resize(
                  d_numberNodesPerElement * NumTotalSphericalFunctions *
                    maxkPoints,
                  ValueType(0.0));

                if (d_computeIonForces)
                  {
                    d_DMatrixEntriesConjugate[ChargeId][iElemComp + iCell]
                      .resize(d_numberNodesPerElement *
                                NumTotalSphericalFunctions * maxkPoints * 3,
                              ValueType(0.0));
                  }
                if (d_computeCellStress)
                  {
                    d_DDyadicRMatrixEntriesConjugate[ChargeId][iElemComp +
                                                               iCell]
                      .resize(d_numberNodesPerElement *
                                NumTotalSphericalFunctions * maxkPoints * 9,
                              ValueType(0.0));
                    d_CRMatrixEntriesConjugate[ChargeId][iElemComp + iCell]
                      .resize(d_numberNodesPerElement *
                                NumTotalSphericalFunctions * maxkPoints * 3,
                              ValueType(0.0));
                  }
                std::vector<ValueType> &CMatrixEntriesConjugateAtomElem =
                  d_CMatrixEntriesConjugate[ChargeId][iElemComp + iCell];

                std::vector<ValueType> *DMatrixEntriesConjugateAtomElem =
                  d_computeIonForces ?
                    &d_DMatrixEntriesConjugate[ChargeId][iElemComp + iCell] :
                    nullptr;

                std::vector<ValueType> *DDyadicRMatrixEntriesConjugateAtomElem =
                  d_computeCellStress ?
                    &d_DDyadicRMatrixEntriesConjugate[ChargeId]
                                                     [iElemComp + iCell] :
                    nullptr;

                std::vector<ValueType> *CRMatrixEntriesConjugateAtomElem =
                  d_computeCellStress ?
                    &d_CRMatrixEntriesConjugate[ChargeId][iElemComp + iCell] :
                    nullptr;

                std::vector<ValueType> &CMatrixEntriesTransposeAtomElem =
                  d_CMatrixEntriesTranspose[ChargeId][iElemComp + iCell];



                for (dftfe::Int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  {
                    for (dftfe::Int beta = 0; beta < NumTotalSphericalFunctions;
                         ++beta)
                      for (dftfe::Int iNode = 0;
                           iNode < d_numberNodesPerElement;
                           ++iNode)
                        {
                          const dftfe::uInt flattenedIndex =
                            kPoint * NumTotalSphericalFunctions *
                              d_numberNodesPerElement * nCellsPerBatch +
                            iCell * NumTotalSphericalFunctions *
                              d_numberNodesPerElement +
                            beta * d_numberNodesPerElement + iNode;

                          const dataTypes::number temp =
                            projectorMatricesHost[flattenedIndex];
#ifdef USE_COMPLEX
                          CMatrixEntriesConjugateAtomElem
                            [kPoint * d_numberNodesPerElement *
                               NumTotalSphericalFunctions +
                             d_numberNodesPerElement * beta + iNode] =
                              std::conj(temp);
                          CMatrixEntriesTransposeAtomElem
                            [kPoint * d_numberNodesPerElement *
                               NumTotalSphericalFunctions +
                             NumTotalSphericalFunctions * iNode + beta] = temp;
                          if (d_computeIonForces)
                            {
                              for (dftfe::Int iDim = 0; iDim < 3; ++iDim)
                                {
                                  const dftfe::uInt flattenedIndexForD =
                                    kPoint * NumTotalSphericalFunctions *
                                      d_numberNodesPerElement * 3 *
                                      nCellsPerBatch +
                                    iCell * NumTotalSphericalFunctions * 3 *
                                      d_numberNodesPerElement +
                                    beta * 3 * d_numberNodesPerElement +
                                    iDim * d_numberNodesPerElement + iNode;

                                  (*DMatrixEntriesConjugateAtomElem)
                                    [kPoint * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions * 3 +
                                     iDim * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions +
                                     beta * d_numberNodesPerElement + iNode] =
                                      std::conj(gradientProjectorMatricesHost
                                                  [flattenedIndexForD]);
                                }
                            }
                          if (d_computeCellStress)
                            {
                              for (dftfe::Int iDim = 0; iDim < 9; ++iDim)
                                {
                                  const dftfe::uInt flattenedIndexForD =
                                    kPoint * NumTotalSphericalFunctions *
                                      d_numberNodesPerElement * 9 *
                                      nCellsPerBatch +
                                    iCell * NumTotalSphericalFunctions * 9 *
                                      d_numberNodesPerElement +
                                    beta * 9 * d_numberNodesPerElement +
                                    iDim * d_numberNodesPerElement + iNode;
                                  (*DDyadicRMatrixEntriesConjugateAtomElem)
                                    [kPoint * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions * 9 +
                                     iDim * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions +
                                     beta * d_numberNodesPerElement + iNode] =
                                      std::conj(
                                        gradientProjectorDyadicXMatricesHost
                                          [flattenedIndexForD]);
                                }
                              for (dftfe::Int iDim = 0; iDim < 3; ++iDim)
                                {
                                  const dftfe::uInt flattenedIndexForD =
                                    kPoint * NumTotalSphericalFunctions *
                                      d_numberNodesPerElement * 3 *
                                      nCellsPerBatch +
                                    iCell * NumTotalSphericalFunctions * 3 *
                                      d_numberNodesPerElement +
                                    beta * 3 * d_numberNodesPerElement +
                                    iDim * d_numberNodesPerElement + iNode;
                                  (*CRMatrixEntriesConjugateAtomElem)
                                    [kPoint * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions * 3 +
                                     iDim * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions +
                                     beta * d_numberNodesPerElement + iNode] =
                                      std::conj(projectorTimesXMatricesHost
                                                  [flattenedIndexForD]);
                                }
                            }
#else
                          CMatrixEntriesConjugateAtomElem
                            [d_numberNodesPerElement * beta + iNode] = temp;
                          if (d_computeIonForces)
                            {
                              for (dftfe::Int iDim = 0; iDim < 3; ++iDim)
                                {
                                  const dftfe::uInt flattenedIndexForD =
                                    iCell * NumTotalSphericalFunctions * 3 *
                                      d_numberNodesPerElement +
                                    beta * 3 * d_numberNodesPerElement +
                                    iDim * d_numberNodesPerElement + iNode;
                                  (*DMatrixEntriesConjugateAtomElem)
                                    [iDim * d_numberNodesPerElement *
                                       NumTotalSphericalFunctions +
                                     beta * d_numberNodesPerElement + iNode] =
                                      (gradientProjectorMatricesHost
                                         [flattenedIndexForD]);
                                }
                            }
                          if (d_computeCellStress)
                            for (dftfe::Int iDim = 0; iDim < 9; ++iDim)
                              {
                                const dftfe::uInt flattenedIndexForD =
                                  iCell * NumTotalSphericalFunctions * 9 *
                                    d_numberNodesPerElement +
                                  beta * 9 * d_numberNodesPerElement +
                                  iDim * d_numberNodesPerElement + iNode;
                                (*DDyadicRMatrixEntriesConjugateAtomElem)
                                  [iDim * d_numberNodesPerElement *
                                     NumTotalSphericalFunctions +
                                   beta * d_numberNodesPerElement + iNode] =
                                    (gradientProjectorDyadicXMatricesHost
                                       [flattenedIndexForD]);
                              }


                          CMatrixEntriesTransposeAtomElem
                            [NumTotalSphericalFunctions * iNode + beta] = temp;
#endif
                        } // node loop
                  }       // k point loop

              } // non-trivial element loop
          }

      } // ChargeId loop
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (dftfe::uInt iCell = 0; iCell < d_locallyOwnedCells; iCell++)
          {
            if (atomSupportInElement(iCell))
              {
                d_nonlocalElemIdToCellIdVector.push_back(iCell);
                for (dftfe::Int iNode = 0; iNode < d_numberNodesPerElement;
                     iNode++)
                  {
                    dftfe::uInt localNodeId =
                      basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                        [iCell * d_numberNodesPerElement + iNode];
                    d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                      .push_back(localNodeId);
                  }
              }
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_elementIdToNonLocalElementIdMap.clear();
        d_elementIdToNonLocalElementIdMap.resize(d_locallyOwnedCells);
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
          .clear();
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
          .clear();
        d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugate
          .clear();
        d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
          .clear();
        d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugate
          .clear();
        if (!d_useGlobalCMatrix)
          {
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
              .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                        d_numberNodesPerElement * d_maxSingleAtomContribution,
                      ValueType(0.0));
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
              .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                        d_numberNodesPerElement * d_maxSingleAtomContribution,
                      ValueType(0.0));
          }
        if (d_computeIonForces)
          {
            d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
              .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                        d_numberNodesPerElement * d_maxSingleAtomContribution *
                        3,
                      ValueType(0.0));
          }
        if (d_computeCellStress)
          {
            d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugate
              .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                        d_numberNodesPerElement * d_maxSingleAtomContribution *
                        9,
                      ValueType(0.0));
            d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugate
              .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                        d_numberNodesPerElement * d_maxSingleAtomContribution *
                        3,
                      ValueType(0.0));
          }
        std::vector<dftfe::uInt> atomIdsInCurrentProcess =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();

        d_sphericalFnIdsParallelNumberingMap.clear();
        d_sphericalFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);
        d_sphericalFnIdsPaddedParallelNumberingMap.clear();
        d_sphericalFnIdsPaddedParallelNumberingMap.resize(
          atomIdsInCurrentProcess.size() * d_maxSingleAtomContribution, -1);

        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution, -1);

        d_nonlocalElemIdToLocalElemIdMap.clear();
        d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);

        d_mapSphericalFnTimesVectorAllCellsReduction.clear();
        d_mapSphericalFnTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution,
          d_totalNonLocalEntries + 1);



        dftfe::uInt countElemNode    = 0;
        dftfe::uInt countElem        = 0;
        dftfe::uInt countAlpha       = 0;
        dftfe::uInt numShapeFnsAccum = 0;

        dftfe::Int totalElements = 0;
        d_mapiAtomTosphFuncWaveStart.resize(d_totalAtomsInCurrentProc);

        std::map<dftfe::uInt, dftfe::uInt> atomIdToNumShapeFnsAccumulated;
        std::map<dftfe::uInt, dftfe::uInt> atomIdToMaxShapeFnsAccumulated;
        for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            const dftfe::uInt        atomId = atomIdsInCurrentProcess[iAtom];
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
              d_atomCenteredSphericalFunctionContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            dftfe::uInt totalAtomIdElementIterators =
              elementIndexesInAtomCompactSupport.size();
            totalElements += totalAtomIdElementIterators;
            const dftfe::uInt Znum = atomicNumber[atomId];
            const dftfe::uInt numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


            for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                dftfe::uInt globalId =
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    [std::make_pair(atomId, alpha)];

                const dftfe::uInt id = d_SphericalFunctionKetTimesVectorPar[0]
                                         .get_partitioner()
                                         ->global_to_local(globalId);

                if (alpha == 0)
                  {
                    d_mapiAtomTosphFuncWaveStart[iAtom] = countAlpha;
                  }
                d_sphericalFnIdsParallelNumberingMap[countAlpha] = id;
                d_sphericalFnIdsPaddedParallelNumberingMap
                  [iAtom * d_maxSingleAtomContribution + alpha] = id;

                countAlpha++;
              }

            atomIdToNumShapeFnsAccumulated[atomId] = numShapeFnsAccum;
            atomIdToMaxShapeFnsAccumulated[atomId] =
              iAtom * d_maxSingleAtomContribution;
            numShapeFnsAccum += numberSphericalFunctions;
          }

        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
        for (dftfe::uInt iElem = 0; iElem < d_locallyOwnedCells; iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> &atomIdsInCell =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCell.size();
                     iAtom++)
                  {
                    dftfe::uInt atomId = atomIdsInCell[iAtom];
                    dftfe::uInt Znum   = atomicNumber[atomId];
                    dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];
                    if (!d_useGlobalCMatrix)
                      {
                        for (dftfe::uInt ikpoint = 0;
                             ikpoint < d_kPointWeights.size();
                             ikpoint++)
                          for (dftfe::uInt iNode = 0;
                               iNode < d_numberNodesPerElement;
                               ++iNode)
                            {
                              for (dftfe::uInt alpha = 0;
                                   alpha < numberSphericalFunctions;
                                   ++alpha)
                                {
                                  d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                                    [ikpoint * d_totalNonlocalElems *
                                       d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     countElem * d_maxSingleAtomContribution *
                                       d_numberNodesPerElement +
                                     d_numberNodesPerElement * alpha +
                                     iNode] = d_CMatrixEntriesConjugate
                                      [atomId][nonZeroElementMatrixId]
                                      [ikpoint * d_numberNodesPerElement *
                                         numberSphericalFunctions +
                                       d_numberNodesPerElement * alpha + iNode];

                                  d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
                                    [ikpoint * d_totalNonlocalElems *
                                       d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     countElem * d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     d_maxSingleAtomContribution * iNode +
                                     alpha] = d_CMatrixEntriesTranspose
                                      [atomId][nonZeroElementMatrixId]
                                      [ikpoint * d_numberNodesPerElement *
                                         numberSphericalFunctions +
                                       numberSphericalFunctions * iNode +
                                       alpha];
                                }
                            }
                      }
                    // Fill the data for Dmatrix and DdyadicRmatrix @Kartick
                    if (d_computeIonForces)
                      {
                        for (dftfe::uInt ikpoint = 0;
                             ikpoint < d_kPointWeights.size();
                             ikpoint++)
                          {
                            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                              for (dftfe::uInt iNode = 0;
                                   iNode < d_numberNodesPerElement;
                                   ++iNode)
                                {
                                  for (dftfe::uInt alpha = 0;
                                       alpha < numberSphericalFunctions;
                                       ++alpha)
                                    {
                                      d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                                        [ikpoint * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution * 3 +
                                         iDim * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution +
                                         countElem *
                                           d_maxSingleAtomContribution *
                                           d_numberNodesPerElement +
                                         d_numberNodesPerElement * alpha +
                                         iNode] = d_DMatrixEntriesConjugate
                                          [atomId][nonZeroElementMatrixId]
                                          [ikpoint * d_numberNodesPerElement *
                                             numberSphericalFunctions * 3 +
                                           iDim * d_numberNodesPerElement *
                                             numberSphericalFunctions +
                                           d_numberNodesPerElement * alpha +
                                           iNode];
                                    }
                                }
                          }
                      }
                    if (d_computeCellStress)
                      {
                        for (dftfe::uInt ikpoint = 0;
                             ikpoint < d_kPointWeights.size();
                             ikpoint++)
                          {
                            for (dftfe::uInt iDim = 0; iDim < 9; iDim++)
                              for (dftfe::uInt iNode = 0;
                                   iNode < d_numberNodesPerElement;
                                   ++iNode)
                                {
                                  for (dftfe::uInt alpha = 0;
                                       alpha < numberSphericalFunctions;
                                       ++alpha)
                                    {
                                      d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugate
                                        [ikpoint * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution * 9 +
                                         iDim * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution +
                                         countElem *
                                           d_maxSingleAtomContribution *
                                           d_numberNodesPerElement +
                                         d_numberNodesPerElement * alpha +
                                         iNode] =
                                          d_DDyadicRMatrixEntriesConjugate
                                            [atomId][nonZeroElementMatrixId]
                                            [ikpoint * d_numberNodesPerElement *
                                               numberSphericalFunctions * 9 +
                                             iDim * d_numberNodesPerElement *
                                               numberSphericalFunctions +
                                             d_numberNodesPerElement * alpha +
                                             iNode];
                                    }
                                }
                            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                              for (dftfe::uInt iNode = 0;
                                   iNode < d_numberNodesPerElement;
                                   ++iNode)
                                {
                                  for (dftfe::uInt alpha = 0;
                                       alpha < numberSphericalFunctions;
                                       ++alpha)
                                    {
                                      d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugate
                                        [ikpoint * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution * 3 +
                                         iDim * d_totalNonlocalElems *
                                           d_numberNodesPerElement *
                                           d_maxSingleAtomContribution +
                                         countElem *
                                           d_maxSingleAtomContribution *
                                           d_numberNodesPerElement +
                                         d_numberNodesPerElement * alpha +
                                         iNode] = d_CRMatrixEntriesConjugate
                                          [atomId][nonZeroElementMatrixId]
                                          [ikpoint * d_numberNodesPerElement *
                                             numberSphericalFunctions * 3 +
                                           iDim * d_numberNodesPerElement *
                                             numberSphericalFunctions +
                                           d_numberNodesPerElement * alpha +
                                           iNode];
                                    }
                                }
                          }
                      }
                    d_nonlocalElemIdToLocalElemIdMap[countElem] = iElem;
                    for (dftfe::uInt alpha = 0;
                         alpha < numberSphericalFunctions;
                         ++alpha)
                      {
                        const dftfe::uInt index =
                          countElem * d_maxSingleAtomContribution + alpha;
                        d_mapSphericalFnTimesVectorAllCellsReduction[index] =
                          atomIdToNumShapeFnsAccumulated[atomId] + alpha;
                        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                          [countElem * d_maxSingleAtomContribution + alpha] =
                            atomIdToMaxShapeFnsAccumulated[atomId] + alpha;
                      }
                    d_elementIdToNonLocalElementIdMap[iElem].push_back(
                      std::make_pair(atomId, countElem));
                    countElem++;
                  }
              }
          }
        if (!d_useGlobalCMatrix)
          {
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
              .resize(
                d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                  .size() /
                d_kPointWeights.size());
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
              .resize(
                d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
                  .size() /
                d_kPointWeights.size());
          }
        if (d_computeIonForces)
          {
            d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
              .resize(
                d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                  .size() /
                d_kPointWeights.size());
          }
        if (d_computeCellStress)
          {
            d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugateDevice
              .resize(
                d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugate
                  .size() /
                d_kPointWeights.size());
            d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugateDevice
              .resize(
                d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugate
                  .size() /
                d_kPointWeights.size());
          }


        d_sphericalFnIdsParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.resize(
          d_sphericalFnIdsPaddedParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.resize(
          d_sphericalFnIdsParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsParallelNumberingMap);
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsPaddedParallelNumberingMap);
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.resize(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size());
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.copyFrom(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);


        d_mapSphericalFnTimesVectorAllCellsReductionDevice.clear();
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.resize(
          d_mapSphericalFnTimesVectorAllCellsReduction.size());
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.copyFrom(
          d_mapSphericalFnTimesVectorAllCellsReduction);


        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (dftfe::uInt i = 0; i < d_totalNonlocalElems; i++)
          {
            dftfe::uInt iCell = d_nonlocalElemIdToLocalElemIdMap[i];

            d_nonlocalElemIdToCellIdVector.push_back(iCell);
            for (dftfe::Int iNode = 0; iNode < d_numberNodesPerElement; iNode++)
              {
                dftfe::uInt localNodeId =
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    [iCell * d_numberNodesPerElement + iNode];
                d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                  .push_back(localNodeId);
              }
          }
      }



#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseFlattenedDataStructure(
      dftfe::uInt waveFunctionBlockSize,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      const nonLocalContractionVectorType NonLocalContractionVectorType)
  {
    std::vector<dftfe::uInt> tempNonLocalCellDofVector(
      d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.size());
    std::transform(
      d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.begin(),
      d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.end(),
      tempNonLocalCellDofVector.begin(),
      [&waveFunctionBlockSize](auto &c) { return c * waveFunctionBlockSize; });
    d_flattenedNonLocalCellDofIndexToProcessDofIndexMap.clear();
    d_flattenedNonLocalCellDofIndexToProcessDofIndexMap.resize(
      d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.size());
    d_flattenedNonLocalCellDofIndexToProcessDofIndexMap.copyFrom(
      tempNonLocalCellDofVector);

    if (d_useGlobalCMatrix)
      {
        for (dftfe::uInt iAtomicNum = 0;
             iAtomicNum < d_setOfAtomicNumber.size();
             iAtomicNum++)
          {
            dftfe::uInt Znum =
              *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
            dftfe::uInt numSphFunc =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numAtomsPerSpecies =
              d_listOfiAtomInSpecies[Znum].size();
            d_dotProductAtomicWaveInputWaveTemp[iAtomicNum].resize(
              numAtomsPerSpecies * numSphFunc * waveFunctionBlockSize);
          }
      }
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_numberWaveFunctions = waveFunctionBlockSize;

        if (NonLocalContractionVectorType ==
            nonLocalContractionVectorType::CconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesWavefunMatrix.clear();
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Znum   = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                d_sphericalFnTimesWavefunMatrix[atomId].resize(
                  numberSphericalFunctions * d_numberWaveFunctions,
                  ValueType(0.0));
              }
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::CRconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 3,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesXTimesWavefunMatrix.clear();
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Znum   = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                d_sphericalFnTimesXTimesWavefunMatrix[atomId].resize(
                  numberSphericalFunctions * d_numberWaveFunctions * 3,
                  ValueType(0.0));
              }
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 3,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesGradientWavefunMatrix.clear();
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Znum   = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                d_sphericalFnTimesGradientWavefunMatrix[atomId].resize(
                  numberSphericalFunctions * d_numberWaveFunctions * 3,
                  ValueType(0.0));
              }
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DDyadicRconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 9,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesGradientWavefunDyadicXMatrix.clear();
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                dftfe::uInt atomId = atomIdsInProcessor[iAtom];
                dftfe::uInt Znum   = atomicNumber[atomId];
                dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                d_sphericalFnTimesGradientWavefunDyadicXMatrix[atomId].resize(
                  numberSphericalFunctions * d_numberWaveFunctions * 9,
                  ValueType(0.0));
              }
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_numberWaveFunctions = waveFunctionBlockSize;

        if (NonLocalContractionVectorType ==
            nonLocalContractionVectorType::CconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesVectorAllCellsDevice.clear();
            d_sphericalFnTimesVectorAllCellsDevice.resize(
              d_totalNonlocalElems * d_numberWaveFunctions *
                d_maxSingleAtomContribution,
              ValueType(0.0));
            const std::vector<dftfe::uInt> atomIdsInProcessor =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            d_sphericalFnTimesVectorDevice.clear();
            d_sphericalFnTimesVectorDevice.resize(atomIdsInProcessor.size() *
                                                    d_numberWaveFunctions *
                                                    d_maxSingleAtomContribution,
                                                  ValueType(0.0));
            d_couplingMatrixTimesVectorDevice.clear();
            d_couplingMatrixTimesVectorDevice.resize(
              atomIdsInProcessor.size() * d_numberWaveFunctions *
                d_maxSingleAtomContribution,
              ValueType(0.0));

            if (!d_useGlobalCMatrix)
              {
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice.clear();
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice.resize(
                  d_numberWaveFunctions * d_totalNonlocalElems *
                    d_numberNodesPerElement,
                  ValueType(0.0));
              }
            d_sphericalFnTimesWavefunctionMatrix.clear();
            d_sphericalFnTimesWavefunctionMatrix.resize(d_numberWaveFunctions *
                                                        d_totalNonLocalEntries);
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::CRconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 3,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesXTimesVectorAllCellsDevice.clear();
            d_sphericalFnTimesXTimesVectorAllCellsDevice.resize(
              d_totalNonlocalElems * d_numberWaveFunctions * 3 *
                d_maxSingleAtomContribution,
              ValueType(0.0));
            d_sphericalFnTimesXTimesWavefunctionMatrix.clear();
            d_sphericalFnTimesXTimesWavefunctionMatrix.resize(
              d_numberWaveFunctions * d_totalNonLocalEntries * 3);
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 3,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesGradientVectorAllCellsDevice.clear();
            d_sphericalFnTimesGradientVectorAllCellsDevice.resize(
              d_totalNonlocalElems * d_numberWaveFunctions * 3 *
                d_maxSingleAtomContribution,
              ValueType(0.0));
            d_sphericalFnTimesGradientWavefunctionMatrix.clear();
            d_sphericalFnTimesGradientWavefunctionMatrix.resize(
              d_numberWaveFunctions * d_totalNonLocalEntries * 3);
          }
        else if (NonLocalContractionVectorType ==
                 nonLocalContractionVectorType::DDyadicRconjTransX)
          {
            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
              waveFunctionBlockSize * 9,
              sphericalFunctionKetTimesVectorParFlattened);
            d_sphericalFnTimesRDyadicGradientVectorAllCellsDevice.clear();
            d_sphericalFnTimesRDyadicGradientVectorAllCellsDevice.resize(
              d_totalNonlocalElems * d_numberWaveFunctions * 9 *
                d_maxSingleAtomContribution,
              ValueType(0.0));
            d_sphericalFnTimesGradientWavefunctionDyadicXMatrix.clear();
            d_sphericalFnTimesGradientWavefunctionDyadicXMatrix.resize(
              d_numberWaveFunctions * d_totalNonLocalEntries * 9);
          }
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::initialisePartitioner()
  {
    std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const dftfe::uInt numberAtoms =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    // //
    // // data structures for memory optimization of projectorKetTimesVector
    // //
    std::vector<dftfe::uInt> atomIdsAllProcessFlattened;
    MPI_Barrier(d_mpi_communicator);
    pseudoUtils::exchangeLocalList(atomIdsInCurrentProcess,
                                   atomIdsAllProcessFlattened,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<dftfe::uInt> atomIdsSizeCurrentProcess(1);
    atomIdsSizeCurrentProcess[0] = atomIdsInCurrentProcess.size();
    std::vector<dftfe::uInt> atomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(atomIdsSizeCurrentProcess,
                                   atomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<std::vector<dftfe::uInt>> atomIdsInAllProcess(
      d_n_mpi_processes);
    dftfe::uInt count = 0;
    for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        for (dftfe::uInt j = 0; j < atomIdsSizesAllProcess[iProc]; j++)
          {
            atomIdsInAllProcess[iProc].push_back(
              atomIdsAllProcessFlattened[count]);
            count++;
          }
      }
    atomIdsAllProcessFlattened.clear();

    std::vector<std::vector<dftfe::uInt>> atomIdsAndProcsList(numberAtoms);
    for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        for (dftfe::uInt iAtom = 0; iAtom < atomIdsInAllProcess[iProc].size();
             iAtom++)
          {
            dftfe::uInt atomId = atomIdsInAllProcess[iProc][iAtom];
            atomIdsAndProcsList[atomId].push_back(iProc);
          }
      }


    dealii::IndexSet ownedAtomIdsInCurrentProcess;
    ownedAtomIdsInCurrentProcess.set_size(numberAtoms); // Check this
    dealii::IndexSet ghostAtomIdsInCurrentProcess;
    ghostAtomIdsInCurrentProcess.set_size(numberAtoms);
    ghostAtomIdsInCurrentProcess.add_indices(atomIdsInCurrentProcess.begin(),
                                             atomIdsInCurrentProcess.end());


    std::vector<dftfe::uInt> ownedAtomSize(d_n_mpi_processes, 0);
    for (dftfe::uInt atomId = 0; atomId < numberAtoms; atomId++)
      {
        const std::vector<dftfe::uInt> procsList = atomIdsAndProcsList[atomId];
        dftfe::uInt                    lowestOwnedAtoms = 100000;
        dftfe::uInt                    lowestProcId     = 0;
        for (dftfe::Int iProc = 0; iProc < procsList.size(); iProc++)
          {
            dftfe::uInt procId = procsList[iProc];
            if (ownedAtomSize[procId] < lowestOwnedAtoms)
              {
                lowestOwnedAtoms = ownedAtomSize[procId];
                lowestProcId     = procId;
              }
          }

        ownedAtomSize[lowestProcId] += 1;

        if (lowestProcId == d_this_mpi_process)
          {
            ownedAtomIdsInCurrentProcess.add_index(atomId);
          }

      } // atomId


    // for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
    //   {
    //     if (iProc < d_this_mpi_process)
    //       {
    //         dealii::IndexSet temp;
    //         temp.set_size(numberAtoms);
    //         temp.add_indices(atomIdsInAllProcess[iProc].begin(),
    //                          atomIdsInAllProcess[iProc].end());
    //         ownedAtomIdsInCurrentProcess.subtract_set(temp);
    //       }
    //   }

    ghostAtomIdsInCurrentProcess.subtract_set(ownedAtomIdsInCurrentProcess);

    std::vector<dftfe::uInt> ownedAtomIdsSizeCurrentProcess(1);
    ownedAtomIdsSizeCurrentProcess[0] =
      ownedAtomIdsInCurrentProcess.n_elements();
    std::vector<dftfe::uInt> ownedAtomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(ownedAtomIdsSizeCurrentProcess,
                                   ownedAtomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);
    // // renumbering to make contiguous set of nonLocal atomIds
    std::map<dftfe::Int, dftfe::Int> oldToNewAtomIds;
    std::map<dftfe::Int, dftfe::Int> newToOldAtomIds;
    dftfe::uInt                      startingCount = 0;
    for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            startingCount += ownedAtomIdsSizesAllProcess[iProc];
          }
      }

    dealii::IndexSet ownedAtomIdsInCurrentProcessRenum,
      ghostAtomIdsInCurrentProcessRenum;
    ownedAtomIdsInCurrentProcessRenum.set_size(numberAtoms);
    ghostAtomIdsInCurrentProcessRenum.set_size(numberAtoms);
    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcess.begin();
         it != ownedAtomIdsInCurrentProcess.end();
         it++)
      {
        oldToNewAtomIds[*it]           = startingCount;
        newToOldAtomIds[startingCount] = *it;
        ownedAtomIdsInCurrentProcessRenum.add_index(startingCount);
        startingCount++;
      }

    pseudoUtils::exchangeNumberingMap(oldToNewAtomIds,
                                      d_n_mpi_processes,
                                      d_mpi_communicator);
    pseudoUtils::exchangeNumberingMap(newToOldAtomIds,
                                      d_n_mpi_processes,
                                      d_mpi_communicator);

    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcess.begin();
         it != ghostAtomIdsInCurrentProcess.end();
         it++)
      {
        dftfe::uInt newAtomId = oldToNewAtomIds[*it];
        ghostAtomIdsInCurrentProcessRenum.add_index(newAtomId);
      }

    if (d_this_mpi_process == 0 && false)
      {
        for (std::map<dftfe::Int, dftfe::Int>::const_iterator it =
               oldToNewAtomIds.begin();
             it != oldToNewAtomIds.end();
             it++)
          std::cout << " old nonlocal atom id: " << it->first
                    << " new nonlocal atomid: " << it->second << std::endl;

        std::cout
          << "number of local owned non local atom ids in all processors"
          << '\n';
        for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
          std::cout << ownedAtomIdsSizesAllProcess[iProc] << ",";
        std::cout << std::endl;
      }
    if (false)
      {
        std::stringstream ss1;
        ownedAtomIdsInCurrentProcess.print(ss1);
        std::stringstream ss2;
        ghostAtomIdsInCurrentProcess.print(ss2);
        std::string s1(ss1.str());
        s1.pop_back();
        std::string s2(ss2.str());
        s2.pop_back();
        std::cout << "procId: " << d_this_mpi_process << " old owned: " << s1
                  << " old ghost: " << s2 << std::endl;
        std::stringstream ss3;
        ownedAtomIdsInCurrentProcessRenum.print(ss3);
        std::stringstream ss4;
        ghostAtomIdsInCurrentProcessRenum.print(ss4);
        std::string s3(ss3.str());
        s3.pop_back();
        std::string s4(ss4.str());
        s4.pop_back();
        std::cout << "procId: " << d_this_mpi_process << " new owned: " << s3
                  << " new ghost: " << s4 << std::endl;
      }
    AssertThrow(
      ownedAtomIdsInCurrentProcessRenum.is_ascending_and_one_to_one(
        d_mpi_communicator),
      dealii::ExcMessage(
        "Incorrect renumbering and/or partitioning of non local atom ids"));

    dftfe::Int               numberLocallyOwnedSphericalFunctions = 0;
    dftfe::Int               numberGhostSphericalFunctions        = 0;
    std::vector<dftfe::uInt> coarseNodeIdsCurrentProcess;
    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcessRenum.begin();
         it != ownedAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        coarseNodeIdsCurrentProcess.push_back(
          numberLocallyOwnedSphericalFunctions);
        numberLocallyOwnedSphericalFunctions +=
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(
              atomicNumber[newToOldAtomIds[*it]]);
      }

    std::vector<dftfe::uInt> ghostAtomIdNumberSphericalFunctions;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        const dftfe::uInt temp = d_atomCenteredSphericalFunctionContainer
                                   ->getTotalNumberOfSphericalFunctionsPerAtom(
                                     atomicNumber[newToOldAtomIds[*it]]);
        numberGhostSphericalFunctions += temp;
        ghostAtomIdNumberSphericalFunctions.push_back(temp);
      }

    std::vector<dftfe::uInt> numberLocallyOwnedSphericalFunctionsCurrentProcess(
      1);
    numberLocallyOwnedSphericalFunctionsCurrentProcess[0] =
      numberLocallyOwnedSphericalFunctions;
    std::vector<dftfe::uInt> numberLocallyOwnedSphericalFunctionsAllProcess;
    pseudoUtils::exchangeLocalList(
      numberLocallyOwnedSphericalFunctionsCurrentProcess,
      numberLocallyOwnedSphericalFunctionsAllProcess,
      d_n_mpi_processes,
      d_mpi_communicator);

    startingCount = 0;
    for (dftfe::uInt iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            startingCount +=
              numberLocallyOwnedSphericalFunctionsAllProcess[iProc];
          }
      }

    d_locallyOwnedSphericalFunctionIdsCurrentProcess.clear();
    d_locallyOwnedSphericalFunctionIdsCurrentProcess.set_size(
      std::accumulate(numberLocallyOwnedSphericalFunctionsAllProcess.begin(),
                      numberLocallyOwnedSphericalFunctionsAllProcess.end(),
                      0));
    std::vector<dftfe::uInt> v(numberLocallyOwnedSphericalFunctions);
    std::iota(std::begin(v), std::end(v), startingCount);
    d_locallyOwnedSphericalFunctionIdsCurrentProcess.add_indices(v.begin(),
                                                                 v.end());

    std::vector<dftfe::uInt> coarseNodeIdsAllProcess;
    for (dftfe::uInt i = 0; i < coarseNodeIdsCurrentProcess.size(); ++i)
      coarseNodeIdsCurrentProcess[i] += startingCount;
    pseudoUtils::exchangeLocalList(coarseNodeIdsCurrentProcess,
                                   coarseNodeIdsAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    d_ghostSphericalFunctionIdsCurrentProcess.clear();
    d_ghostSphericalFunctionIdsCurrentProcess.set_size(
      std::accumulate(numberLocallyOwnedSphericalFunctionsAllProcess.begin(),
                      numberLocallyOwnedSphericalFunctionsAllProcess.end(),
                      0));
    dftfe::uInt localGhostCount = 0;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        std::vector<dftfe::uInt> g(
          ghostAtomIdNumberSphericalFunctions[localGhostCount]);
        std::iota(std::begin(g), std::end(g), coarseNodeIdsAllProcess[*it]);
        d_ghostSphericalFunctionIdsCurrentProcess.add_indices(g.begin(),
                                                              g.end());
        localGhostCount++;
      }
    if (false)
      {
        std::stringstream ss1;
        d_locallyOwnedSphericalFunctionIdsCurrentProcess.print(ss1);
        std::stringstream ss2;
        d_ghostSphericalFunctionIdsCurrentProcess.print(ss2);
        std::string s1(ss1.str());
        s1.pop_back();
        std::string s2(ss2.str());
        s2.pop_back();
        std::cout << "procId: " << d_this_mpi_process
                  << " projectors owned: " << s1 << " projectors ghost: " << s2
                  << std::endl;
      }
    AssertThrow(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess
        .is_ascending_and_one_to_one(d_mpi_communicator),
      dealii::ExcMessage(
        "Incorrect numbering and/or partitioning of non local projectors"));

    d_sphericalFunctionIdsNumberingMapCurrentProcess.clear();
    d_OwnedAtomIdsInCurrentProcessor.clear();
    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcess.begin();
         it != ownedAtomIdsInCurrentProcess.end();
         it++)
      {
        const dftfe::Int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);
        d_OwnedAtomIdsInCurrentProcessor.push_back(*it);

        for (dftfe::uInt i = 0; i < numberSphericalFunctions; ++i)
          {
            d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
              *it, i)] = coarseNodeIdsAllProcess[oldToNewAtomIds[*it]] + i;
          }
      }

    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcess.begin();
         it != ghostAtomIdsInCurrentProcess.end();
         it++)
      {
        const dftfe::Int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);

        for (dftfe::uInt i = 0; i < numberSphericalFunctions; ++i)
          {
            d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
              *it, i)] = coarseNodeIdsAllProcess[oldToNewAtomIds[*it]] + i;
          }
      }

    if (false)
      {
        for (std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                      dftfe::uInt>::const_iterator it =
               d_sphericalFunctionIdsNumberingMapCurrentProcess.begin();
             it != d_sphericalFunctionIdsNumberingMapCurrentProcess.end();
             ++it)
          {
            std::cout << "procId: " << d_this_mpi_process << " ["
                      << it->first.first << "," << it->first.second << "] "
                      << it->second << std::endl;
          }
      }
      // d_mpiPatternP2P =
      //   std::make_shared<const
      //   utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>(
      //     d_locallyOwnedSphericalFunctionIdsCurrentProcess,
      //     d_ghostSphericalFunctionIdsCurrentProcess,
      //     d_mpi_communicator);
      // ValueType zero = 0.0;
      // d_SphericalFunctionKetTimesVectorFlattened =
      //   dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>(
      //     d_mpiPatternP2P, d_numberOfVectors, zero);
#ifdef USE_COMPLEX
    distributedCPUVec<std::complex<double>> vec(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess,
      d_ghostSphericalFunctionIdsCurrentProcess,
      d_mpi_communicator);
#else
    distributedCPUVec<double> vec(
      d_locallyOwnedSphericalFunctionIdsCurrentProcess,
      d_ghostSphericalFunctionIdsCurrentProcess,
      d_mpi_communicator);
#endif
    vec.update_ghost_values();
    d_SphericalFunctionKetTimesVectorPar.resize(1);
    d_SphericalFunctionKetTimesVectorPar[0].reinit(vec);
    std::vector<std::pair<dftfe::uInt, dftfe::uInt>> localIds;
    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt       globalId =
          d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
            atomId, 0)];

        const dftfe::uInt id = d_SphericalFunctionKetTimesVectorPar[0]
                                 .get_partitioner()
                                 ->global_to_local(globalId);
        localIds.push_back(std::pair<dftfe::uInt, dftfe::uInt>(id, iAtom));
      }
    std::sort(localIds.begin(), localIds.end());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::getTotalAtomInCurrentProcessor()
    const
  {
    return (d_totalAtomsInCurrentProc);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonLocalElementsInCurrentProcessor() const
  {
    return (d_totalNonlocalElems);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonLocalEntriesCurrentProcessor() const
  {
    return (d_totalNonLocalEntries);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::getMaxSingleAtomEntries() const
  {
    return (d_maxSingleAtomContribution);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::atomSupportInElement(
    dftfe::uInt iElem) const
  {
    return (
      d_atomCenteredSphericalFunctionContainer->atomSupportInElement(iElem));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getGlobalDofAtomIdSphericalFnPair(const dftfe::uInt atomId,
                                      const dftfe::uInt alpha) const
  {
    return d_sphericalFunctionIdsNumberingMapCurrentProcess
      .find(std::make_pair(atomId, alpha))
      ->second;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getLocalIdOfDistributedVec(const dftfe::uInt globalId) const
  {
    return (d_SphericalFunctionKetTimesVectorPar[0]
              .get_partitioner()
              ->global_to_local(globalId));
  }


  // To be removed
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getAtomCenteredKpointIndexedSphericalFnQuadValues() const
  {
    return d_atomCenteredKpointIndexedSphericalFnQuadValues;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues() const
  {
    return d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues;
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getSphericalFnTimesVectorFlattenedVectorLocalIds() const

  {
    return d_sphericalFnTimesVectorFlattenedVectorLocalIds;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::vector<dftfe::uInt>> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getAtomIdToNonTrivialSphericalFnCellStartIndex() const
  {
    return d_atomIdToNonTrivialSphericalFnCellStartIndex;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonTrivialSphericalFnsOverAllCells() const
  {
    return d_sumNonTrivialSphericalFnOverAllCells;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<dftfe::uInt, std::vector<dftfe::uInt>> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCellIdToAtomIdsLocalCompactSupportMap() const
  {
    return d_cellIdToAtomIdsLocalCompactSupportMap;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonTrivialSphericalFnsPerCell() const
  {
    return d_nonTrivialSphericalFnPerCell;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonTrivialSphericalFnsCellStartIndex() const
  {
    return d_nonTrivialSphericalFnsCellStartIndex;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonTrivialAllCellsSphericalFnAlphaToElemIdMap() const
  {
    return d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap;
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    copyBackFromDistributedVectorToLocalDataStructure(
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      const dftfe::utils::MemoryStorage<double, memorySpace> &scalingVector)
  {
    if (d_totalNonLocalEntries > 0)
      {
        AssertThrow(
          scalingVector.size() >= d_numberWaveFunctions,
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent size of scaling vector. Not same as number of WaveFunctions"));

        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            const std::vector<dftfe::uInt> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            dftfe::uInt       startIndex = 0;
            const dftfe::uInt inc        = 1;

            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                const dftfe::uInt atomId = atomIdsInProc[iAtom];
                const dftfe::uInt Znum   = atomicNumber[atomId];
                const dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


                for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                     alpha++)
                  {
                    const dftfe::uInt localId =
                      sphericalFunctionKetTimesVectorParFlattened
                        .getMPIPatternP2P()
                        ->globalToLocal(
                          d_sphericalFunctionIdsNumberingMapCurrentProcess
                            .find(std::make_pair(atomId, alpha))
                            ->second);

                    std::transform(
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions,
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions + d_numberWaveFunctions,
                      scalingVector.begin(),
                      d_sphericalFnTimesWavefunMatrix[atomId].begin() +
                        d_numberWaveFunctions * alpha,
                      [&](auto &a, auto &b) {
                        return b * dataTypes::number(a);
                      });
                  }
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            copyDistributedVectorToPaddedMemoryStorageVectorDevice(
              sphericalFunctionKetTimesVectorParFlattened,
              d_sphericalFnTimesVectorDevice);

            // scaling kernel
            // TODO this function does not takes sqrt of the alpha
            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              sqrtAlphaScalingWaveFunctionEntries(
                d_maxSingleAtomContribution,
                d_numberWaveFunctions,
                d_totalAtomsInCurrentProc,
                scalingVector.data(),
                d_sphericalFnTimesVectorDevice.data());
            // storing in d_sphericalFnTimesWavefunctionMatrix

            dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::HOST>
              sphericalFnTimesVectorHostTemp;
            sphericalFnTimesVectorHostTemp.resize(
              d_sphericalFnTimesVectorDevice.size());
            sphericalFnTimesVectorHostTemp.copyFrom(
              d_sphericalFnTimesVectorDevice);

            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            const std::vector<dftfe::uInt> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            for (dftfe::uInt iAtom = 0; iAtom < atomIdsInProc.size(); iAtom++)
              {
                const dftfe::uInt atomId = atomIdsInProc[iAtom];
                const dftfe::uInt Znum   = atomicNumber[atomId];
                const dftfe::uInt numberOfSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                d_sphericalFnTimesWavefunMatrix[atomId].clear();
                d_sphericalFnTimesWavefunMatrix[atomId].resize(
                  numberOfSphericalFunctions * d_numberWaveFunctions, 0.0);
                const dftfe::uInt offset = iAtom * d_maxSingleAtomContribution;
                d_sphericalFnTimesWavefunMatrix[atomId].copyFrom(
                  sphericalFnTimesVectorHostTemp,
                  numberOfSphericalFunctions * d_numberWaveFunctions,
                  offset * d_numberWaveFunctions,
                  0);
              }
          }
#endif
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const ValueType *
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCconjtansXLocalDataStructure(const dftfe::uInt iAtom) const
  {
    const dftfe::uInt atomId = d_atomCenteredSphericalFunctionContainer
                                 ->getAtomIdsInCurrentProcess()[iAtom];
    return (d_sphericalFnTimesWavefunMatrix.find(atomId)->second).begin();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyVOnCconjtransX(
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
              &sphericalFunctionKetTimesVectorParFlattened,
    const bool flagCopyResultsToMatrix)
  {
    if (d_totalNonLocalEntries > 0)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            const std::vector<dftfe::uInt> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            if (couplingtype == CouplingStructure::diagonal)
              {
                dftfe::uInt       startIndex = 0;
                const dftfe::uInt inc        = 1;
                for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                     iAtom++)
                  {
                    const dftfe::uInt atomId = atomIdsInProc[iAtom];
                    const dftfe::uInt Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


                    for (dftfe::uInt alpha = 0;
                         alpha < numberSphericalFunctions;
                         alpha++)
                      {
                        ValueType nonlocalConstantV =
                          couplingMatrix[startIndex++];
                        const dftfe::uInt localId =
                          sphericalFunctionKetTimesVectorParFlattened
                            .getMPIPatternP2P()
                            ->globalToLocal(
                              d_sphericalFunctionIdsNumberingMapCurrentProcess
                                .find(std::make_pair(atomId, alpha))
                                ->second);
                        if (flagCopyResultsToMatrix)
                          {
                            std::transform(
                              sphericalFunctionKetTimesVectorParFlattened
                                  .begin() +
                                localId * d_numberWaveFunctions,
                              sphericalFunctionKetTimesVectorParFlattened
                                  .begin() +
                                localId * d_numberWaveFunctions +
                                d_numberWaveFunctions,
                              d_sphericalFnTimesWavefunMatrix[atomId].begin() +
                                d_numberWaveFunctions * alpha,
                              [&nonlocalConstantV](auto &a) {
                                return nonlocalConstantV * a;
                              });
                          }
                        else
                          {
                            d_BLASWrapperPtr->xscal(
                              sphericalFunctionKetTimesVectorParFlattened
                                  .begin() +
                                localId * d_numberWaveFunctions,
                              nonlocalConstantV,
                              d_numberWaveFunctions);
                          }
                      }
                  }
              }
            else if (couplingtype == CouplingStructure::blockDiagonal)
              {
                const ValueType one   = 1.0;
                const ValueType zero  = 0.0;
                dftfe::uInt     alpha = 0;
                for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
                  {
                    const dftfe::uInt atomId = atomIdsInProc[iAtom];
                    const dftfe::uInt Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::uInt localId =
                      sphericalFunctionKetTimesVectorParFlattened
                        .getMPIPatternP2P()
                        ->globalToLocal(
                          d_sphericalFunctionIdsNumberingMapCurrentProcess
                            .find(std::make_pair(atomId, 0))
                            ->second);
                    d_BLASWrapperPtr->xgemm(
                      'N',
                      'T',
                      d_numberWaveFunctions / 2,
                      numberSphericalFunctions * 2,
                      numberSphericalFunctions * 2,
                      &one,
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions,
                      d_numberWaveFunctions / 2,
                      couplingMatrix.begin() + alpha,
                      numberSphericalFunctions * 2,
                      &zero,
                      &d_sphericalFnTimesWavefunMatrix[atomId][0],
                      d_numberWaveFunctions / 2);
                    if (!flagCopyResultsToMatrix)
                      {
                        d_BLASWrapperPtr->xcopy(
                          d_numberWaveFunctions * numberSphericalFunctions,
                          &d_sphericalFnTimesWavefunMatrix[atomId][0],
                          1,
                          sphericalFunctionKetTimesVectorParFlattened.begin() +
                            localId * d_numberWaveFunctions,
                          1);
                      }
                    alpha +=
                      numberSphericalFunctions * numberSphericalFunctions * 4;
                  }
              }
            else if (couplingtype == CouplingStructure::dense)
              {
                dftfe::uInt       startIndex = 0;
                const dftfe::uInt inc        = 1;
                const ValueType   alpha      = 1;
                const ValueType   beta       = 0;
                for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                     iAtom++)
                  {
                    const dftfe::uInt atomId = atomIdsInProc[iAtom];
                    d_sphericalFnTimesWavefunMatrix[atomId].clear();

                    const dftfe::uInt Znum = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    d_sphericalFnTimesWavefunMatrix[atomId].resize(
                      numberSphericalFunctions * d_numberWaveFunctions, 0.0);
                    std::vector<ValueType> nonlocalConstantVmatrix(
                      numberSphericalFunctions * numberSphericalFunctions, 0.0);
                    d_BLASWrapperPtr->xcopy(numberSphericalFunctions *
                                              numberSphericalFunctions,
                                            &couplingMatrix[startIndex],
                                            1,
                                            &nonlocalConstantVmatrix[0],
                                            1);

                    const dftfe::uInt localId =
                      sphericalFunctionKetTimesVectorParFlattened
                        .getMPIPatternP2P()
                        ->globalToLocal(
                          d_sphericalFunctionIdsNumberingMapCurrentProcess
                            .find(std::make_pair(atomId, 0))
                            ->second);

                    d_BLASWrapperPtr->xgemm(
                      'N',
                      'N',
                      d_numberWaveFunctions,
                      numberSphericalFunctions,
                      numberSphericalFunctions,
                      &alpha,
                      sphericalFunctionKetTimesVectorParFlattened.begin() +
                        localId * d_numberWaveFunctions,
                      d_numberWaveFunctions,
                      &nonlocalConstantVmatrix[0],
                      numberSphericalFunctions,
                      &beta,
                      &d_sphericalFnTimesWavefunMatrix[atomId][0],
                      d_numberWaveFunctions);
                    if (!flagCopyResultsToMatrix)
                      {
                        d_BLASWrapperPtr->xcopy(
                          d_numberWaveFunctions * numberSphericalFunctions,
                          &d_sphericalFnTimesWavefunMatrix[atomId][0],
                          1,
                          sphericalFunctionKetTimesVectorParFlattened.begin() +
                            localId * d_numberWaveFunctions,
                          1);
                      }
                    startIndex +=
                      numberSphericalFunctions * numberSphericalFunctions;
                  }
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            if (couplingtype == CouplingStructure::diagonal)
              {
                copyDistributedVectorToPaddedMemoryStorageVectorDevice(
                  sphericalFunctionKetTimesVectorParFlattened,
                  d_sphericalFnTimesVectorDevice);

                d_BLASWrapperPtr->stridedBlockScale(
                  d_numberWaveFunctions,
                  couplingMatrix.size(),
                  ValueType(1.0),
                  couplingMatrix.begin(),
                  d_sphericalFnTimesVectorDevice.begin());


                if (flagCopyResultsToMatrix)
                  dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                    copyFromParallelNonLocalVecToAllCellsVec(
                      d_numberWaveFunctions,
                      d_totalNonlocalElems,
                      d_maxSingleAtomContribution,
                      d_sphericalFnTimesVectorDevice.begin(),
                      d_sphericalFnTimesVectorAllCellsDevice.begin(),
                      d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice
                        .begin());
                else
                  copyPaddedMemoryStorageVectorToDistributeVectorDevice(
                    d_sphericalFnTimesVectorDevice,
                    sphericalFunctionKetTimesVectorParFlattened);
              }
            else if (couplingtype == CouplingStructure::blockDiagonal)
              {
                copyDistributedVectorToPaddedMemoryStorageVectorDevice(
                  sphericalFunctionKetTimesVectorParFlattened,
                  d_sphericalFnTimesVectorDevice);
                const ValueType one  = 1.0;
                const ValueType zero = 0.0;

                d_BLASWrapperPtr->xgemmStridedBatched(
                  'N',
                  'T',
                  d_numberWaveFunctions / 2,
                  d_maxSingleAtomContribution * 2,
                  d_maxSingleAtomContribution * 2,
                  &one,
                  d_sphericalFnTimesVectorDevice.begin(),
                  d_numberWaveFunctions / 2,
                  d_maxSingleAtomContribution * d_numberWaveFunctions,
                  couplingMatrix.begin(),
                  d_maxSingleAtomContribution * 2,
                  d_maxSingleAtomContribution * d_maxSingleAtomContribution * 4,
                  &zero,
                  d_couplingMatrixTimesVectorDevice.begin(),
                  d_numberWaveFunctions / 2,
                  d_maxSingleAtomContribution * d_numberWaveFunctions,
                  d_totalAtomsInCurrentProc);
                if (flagCopyResultsToMatrix)
                  dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                    copyFromParallelNonLocalVecToAllCellsVec(
                      d_numberWaveFunctions,
                      d_totalNonlocalElems,
                      d_maxSingleAtomContribution,
                      d_couplingMatrixTimesVectorDevice.begin(),
                      d_sphericalFnTimesVectorAllCellsDevice.begin(),
                      d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice
                        .begin());
                else
                  copyPaddedMemoryStorageVectorToDistributeVectorDevice(
                    d_couplingMatrixTimesVectorDevice,
                    sphericalFunctionKetTimesVectorParFlattened);
              }
            else if (couplingtype == CouplingStructure::dense)
              {
                copyDistributedVectorToPaddedMemoryStorageVectorDevice(
                  sphericalFunctionKetTimesVectorParFlattened,
                  d_sphericalFnTimesVectorDevice);
                const ValueType one  = 1.0;
                const ValueType zero = 0.0;

                d_BLASWrapperPtr->xgemmStridedBatched(
                  'N',
                  'T',
                  d_numberWaveFunctions,
                  d_maxSingleAtomContribution,
                  d_maxSingleAtomContribution,
                  &one,
                  d_sphericalFnTimesVectorDevice.begin(),
                  d_numberWaveFunctions,
                  d_maxSingleAtomContribution * d_numberWaveFunctions,
                  couplingMatrix.begin(),
                  d_maxSingleAtomContribution,
                  d_maxSingleAtomContribution * d_maxSingleAtomContribution,
                  &zero,
                  d_couplingMatrixTimesVectorDevice.begin(),
                  d_numberWaveFunctions,
                  d_maxSingleAtomContribution * d_numberWaveFunctions,
                  d_totalAtomsInCurrentProc);
                if (flagCopyResultsToMatrix)

                  {
                    dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                      copyFromParallelNonLocalVecToAllCellsVec(
                        d_numberWaveFunctions,
                        d_totalNonlocalElems,
                        d_maxSingleAtomContribution,
                        d_couplingMatrixTimesVectorDevice.begin(),
                        d_sphericalFnTimesVectorAllCellsDevice.begin(),
                        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice
                          .begin());
                  }
                else
                  {
                    copyPaddedMemoryStorageVectorToDistributeVectorDevice(
                      d_couplingMatrixTimesVectorDevice,
                      sphericalFunctionKetTimesVectorParFlattened);
                  }
              }
          }
#endif
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    applyAllReduceOnCconjtransX(
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
                &sphericalFunctionKetTimesVectorParFlattened,
      const bool skipComm,
      const nonLocalContractionVectorType NonLocalContractionVectorType)
  {
    if (d_totalNonLocalEntries > 0)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            const std::vector<dftfe::uInt> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<dftfe::uInt> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc;
                 iAtom++)
              {
                const dftfe::uInt atomId = atomIdsInProc[iAtom];
                dftfe::uInt       Znum   = atomicNumber[atomId];
                const dftfe::uInt numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                const dftfe::uInt id =
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    .find(std::make_pair(atomId, 0))
                    ->second;
                if (NonLocalContractionVectorType ==
                    nonLocalContractionVectorType::CconjTransX)
                  {
                    std::memcpy(
                      sphericalFunctionKetTimesVectorParFlattened.data() +
                        sphericalFunctionKetTimesVectorParFlattened
                            .getMPIPatternP2P()
                            ->globalToLocal(id) *
                          d_numberWaveFunctions,
                      d_sphericalFnTimesWavefunMatrix[atomId].begin(),
                      d_numberWaveFunctions * numberSphericalFunctions *
                        sizeof(ValueType));
                  }
                if (NonLocalContractionVectorType ==
                    nonLocalContractionVectorType::CRconjTransX)
                  {
                    for (dftfe::Int dim = 0; dim < 3; dim++)
                      d_BLASWrapperPtr->xcopy(
                        d_numberWaveFunctions * numberSphericalFunctions,
                        d_sphericalFnTimesXTimesWavefunMatrix[atomId].begin() +
                          numberSphericalFunctions * d_numberWaveFunctions *
                            dim,
                        1,
                        sphericalFunctionKetTimesVectorParFlattened.data() +
                          sphericalFunctionKetTimesVectorParFlattened
                              .getMPIPatternP2P()
                              ->globalToLocal(id) *
                            d_numberWaveFunctions * 3 +
                          dim,
                        3);
                  }
                else if (NonLocalContractionVectorType ==
                         nonLocalContractionVectorType::DconjTransX)
                  {
                    for (dftfe::Int dim = 0; dim < 3; dim++)
                      d_BLASWrapperPtr->xcopy(
                        d_numberWaveFunctions * numberSphericalFunctions,
                        d_sphericalFnTimesGradientWavefunMatrix[atomId]
                            .begin() +
                          numberSphericalFunctions * d_numberWaveFunctions *
                            dim,
                        1,
                        sphericalFunctionKetTimesVectorParFlattened.data() +
                          sphericalFunctionKetTimesVectorParFlattened
                              .getMPIPatternP2P()
                              ->globalToLocal(id) *
                            d_numberWaveFunctions * 3 +
                          dim,
                        3);
                  }
                else if (NonLocalContractionVectorType ==
                         nonLocalContractionVectorType::DDyadicRconjTransX)
                  {
                    for (dftfe::Int dim = 0; dim < 9; dim++)
                      d_BLASWrapperPtr->xcopy(
                        d_numberWaveFunctions * numberSphericalFunctions,
                        d_sphericalFnTimesGradientWavefunDyadicXMatrix[atomId]
                            .begin() +
                          numberSphericalFunctions * d_numberWaveFunctions *
                            dim,
                        1,
                        sphericalFunctionKetTimesVectorParFlattened.data() +
                          sphericalFunctionKetTimesVectorParFlattened
                              .getMPIPatternP2P()
                              ->globalToLocal(id) *
                            d_numberWaveFunctions * 9 +
                          dim,
                        9);
                  }
              }
            if (!skipComm)
              {
                sphericalFunctionKetTimesVectorParFlattened
                  .accumulateAddLocallyOwned(1);
                sphericalFunctionKetTimesVectorParFlattened.updateGhostValues(
                  1);
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            if (NonLocalContractionVectorType ==
                nonLocalContractionVectorType::CconjTransX)
              {
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  copyToDealiiParallelNonLocalVec(
                    d_numberWaveFunctions,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesWavefunctionMatrix.begin(),
                    sphericalFunctionKetTimesVectorParFlattened.begin(),
                    d_sphericalFnIdsParallelNumberingMapDevice.begin(),
                    1);
              }
            if (NonLocalContractionVectorType ==
                nonLocalContractionVectorType::CRconjTransX)
              {
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  copyToDealiiParallelNonLocalVec(
                    d_numberWaveFunctions,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesXTimesWavefunctionMatrix.begin(),
                    sphericalFunctionKetTimesVectorParFlattened.begin(),
                    d_sphericalFnIdsParallelNumberingMapDevice.begin(),
                    3);
              }
            else if (NonLocalContractionVectorType ==
                     nonLocalContractionVectorType::DconjTransX)
              {
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  copyToDealiiParallelNonLocalVec(
                    d_numberWaveFunctions,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesGradientWavefunctionMatrix.begin(),
                    sphericalFunctionKetTimesVectorParFlattened.begin(),
                    d_sphericalFnIdsParallelNumberingMapDevice.begin(),
                    3);
              }
            else if (NonLocalContractionVectorType ==
                     nonLocalContractionVectorType::DDyadicRconjTransX)
              {
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  copyToDealiiParallelNonLocalVec(
                    d_numberWaveFunctions,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesGradientWavefunctionDyadicXMatrix.begin(),
                    sphericalFunctionKetTimesVectorParFlattened.begin(),
                    d_sphericalFnIdsParallelNumberingMapDevice.begin(),
                    9);
              }


            if (!skipComm)
              {
                sphericalFunctionKetTimesVectorParFlattened
                  .accumulateAddLocallyOwned(1);
                sphericalFunctionKetTimesVectorParFlattened.updateGhostValues(
                  1);
              }
          }
#endif
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCRconjtransOnX(
    const ValueType                          *X,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange)
  {
    AssertThrow(
      d_computeCellStress,
      dealii::ExcMessage(
        "DFT-FE Error: Cell Stress not enabled. This operation is not allowed. Ensure the AtomicCenteredNonLocalOperator is configured to compute cell stress."));


    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType   zero(0.0), one(1.0);
        const dftfe::uInt inc                            = 1;
        d_AllReduceCompleted                             = false;
        dftfe::Int                      numberOfElements = d_locallyOwnedCells;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();

        double integValue = 0.0;
        for (dftfe::Int iElem = cellRange.first; iElem < cellRange.second;
             iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    dftfe::uInt       atomId = atomIdsInElement[iAtom];
                    dftfe::uInt       Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];

                    for (dftfe::Int dim = 0; dim < 3; dim++)
                      d_BLASWrapperPtr->xgemm(
                        'N',
                        'N',
                        d_numberWaveFunctions,
                        numberSphericalFunctions,
                        d_numberNodesPerElement,
                        &one,
                        &X[(iElem - cellRange.first) * d_numberNodesPerElement *
                           d_numberWaveFunctions],
                        d_numberWaveFunctions,
                        &d_CRMatrixEntriesConjugate
                          [atomId][nonZeroElementMatrixId]
                          [3 * d_kPointIndex * d_numberNodesPerElement *
                             numberSphericalFunctions +
                           dim * d_numberNodesPerElement *
                             numberSphericalFunctions],
                        d_numberNodesPerElement,
                        &one,
                        &d_sphericalFnTimesXTimesWavefunMatrix
                          [atomId][dim * d_numberWaveFunctions *
                                   numberSphericalFunctions],
                        d_numberWaveFunctions);
                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first !=
        // d_nonlocalElements
        dftfe::uInt iCellBatch = cellRange.first / d_cellsBlockSize;
        // Xpointer not same assert check
        AssertThrow(
          X == d_wfcStartPointerInCellRange[iCellBatch],
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent X called. Make sure the input X is correct."));
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);
        if (d_nonLocalElementsInCellRange[iCellBatch] > 0)
          for (dftfe::Int iDim = 0; iDim < 3; iDim++)
            d_BLASWrapperPtr->xgemmBatched(
              'N',
              'N',
              d_numberWaveFunctions,
              d_maxSingleAtomContribution,
              d_numberNodesPerElement,
              &scalarCoeffAlpha,
              (const ValueType **)deviceWfcPointersInCellRange[iCellBatch],
              d_numberWaveFunctions,
              (const ValueType **)
                devicePointerCRDaggerInCellRange[iDim][iCellBatch],
              d_numberNodesPerElement,
              &scalarCoeffBeta,
              devicePointerCRDaggerOutTempInCellRange[iDim][iCellBatch],
              d_numberWaveFunctions,
              d_nonLocalElementsInCellRange[iCellBatch]);
        if (iCellBatch == d_numCellBatches - 1)
          {
            d_sphericalFnTimesXTimesWavefunctionMatrix.setValue(ValueType(0.0));
            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
              {
                dftfe::uInt offsetSrc = iDim * d_numberWaveFunctions *
                                        d_maxSingleAtomContribution *
                                        d_totalNonlocalElems;
                dftfe::uInt offsetDst =
                  iDim * d_numberWaveFunctions * d_totalNonLocalEntries;
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  assembleAtomLevelContributionsFromCellLevel(
                    d_numberWaveFunctions,
                    d_totalNonlocalElems,
                    d_maxSingleAtomContribution,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesXTimesVectorAllCellsDevice,
                    d_mapSphericalFnTimesVectorAllCellsReductionDevice,
                    d_sphericalFnTimesXTimesWavefunctionMatrix,
                    offsetSrc,
                    offsetDst);
              }
          }
      }
#endif
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyDconjtransOnX(
    const ValueType                          *X,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange)
  {
    AssertThrow(
      d_computeIonForces,
      dealii::ExcMessage(
        "DFT-FE Error: IonForces not enabled. This operation is not allowed. Ensure the AtomicCenteredNonLocalOperator is configured to compute ion forces."));


    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType   zero(0.0), one(1.0);
        const dftfe::uInt inc                            = 1;
        d_AllReduceCompleted                             = false;
        dftfe::Int                      numberOfElements = d_locallyOwnedCells;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();

        double integValue = 0.0;
        for (dftfe::Int iElem = cellRange.first; iElem < cellRange.second;
             iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    dftfe::uInt       atomId = atomIdsInElement[iAtom];
                    dftfe::uInt       Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];

                    for (dftfe::Int dim = 0; dim < 3; dim++)
                      d_BLASWrapperPtr->xgemm(
                        'N',
                        'N',
                        d_numberWaveFunctions,
                        numberSphericalFunctions,
                        d_numberNodesPerElement,
                        &one,
                        &X[(iElem - cellRange.first) * d_numberNodesPerElement *
                           d_numberWaveFunctions],
                        d_numberWaveFunctions,
                        &d_DMatrixEntriesConjugate
                          [atomId][nonZeroElementMatrixId]
                          [3 * d_kPointIndex * d_numberNodesPerElement *
                             numberSphericalFunctions +
                           dim * d_numberNodesPerElement *
                             numberSphericalFunctions],
                        d_numberNodesPerElement,
                        &one,
                        &d_sphericalFnTimesGradientWavefunMatrix
                          [atomId][dim * d_numberWaveFunctions *
                                   numberSphericalFunctions],
                        d_numberWaveFunctions);
                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first !=
        // d_nonlocalElements
        dftfe::uInt iCellBatch = cellRange.first / d_cellsBlockSize;
        // Xpointer not same assert check
        AssertThrow(
          X == d_wfcStartPointerInCellRange[iCellBatch],
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent X called. Make sure the input X is correct."));
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);
        if (d_nonLocalElementsInCellRange[iCellBatch] > 0)
          for (dftfe::Int iDim = 0; iDim < 3; iDim++)
            d_BLASWrapperPtr->xgemmBatched(
              'N',
              'N',
              d_numberWaveFunctions,
              d_maxSingleAtomContribution,
              d_numberNodesPerElement,
              &scalarCoeffAlpha,
              (const ValueType **)deviceWfcPointersInCellRange[iCellBatch],
              d_numberWaveFunctions,
              (const ValueType **)
                devicePointerDDaggerInCellRange[iDim][iCellBatch],
              d_numberNodesPerElement,
              &scalarCoeffBeta,
              devicePointerDDaggerOutTempInCellRange[iDim][iCellBatch],
              d_numberWaveFunctions,
              d_nonLocalElementsInCellRange[iCellBatch]);
        if (iCellBatch == d_numCellBatches - 1)
          {
            d_sphericalFnTimesGradientWavefunctionMatrix.setValue(
              ValueType(0.0));
            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
              {
                dftfe::uInt offsetSrc = iDim * d_numberWaveFunctions *
                                        d_maxSingleAtomContribution *
                                        d_totalNonlocalElems;
                dftfe::uInt offsetDst =
                  iDim * d_numberWaveFunctions * d_totalNonLocalEntries;
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  assembleAtomLevelContributionsFromCellLevel(
                    d_numberWaveFunctions,
                    d_totalNonlocalElems,
                    d_maxSingleAtomContribution,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesGradientVectorAllCellsDevice,
                    d_mapSphericalFnTimesVectorAllCellsReductionDevice,
                    d_sphericalFnTimesGradientWavefunctionMatrix,
                    offsetSrc,
                    offsetDst);
              }
          }
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    applyDDyadicRconjtransOnX(
      const ValueType                          *X,
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange)
  {
    AssertThrow(
      d_computeCellStress,
      dealii::ExcMessage(
        "DFT-FE Error: IonForces not enabled. This operation is not allowed. Ensure the AtomicCenteredNonLocalOperator is configured to compute ion forces."));


    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType   zero(0.0), one(1.0);
        const dftfe::uInt inc                            = 1;
        d_AllReduceCompleted                             = false;
        dftfe::Int                      numberOfElements = d_locallyOwnedCells;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();

        double integValue = 0.0;
        for (dftfe::Int iElem = cellRange.first; iElem < cellRange.second;
             iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    dftfe::uInt       atomId = atomIdsInElement[iAtom];
                    dftfe::uInt       Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];

                    for (dftfe::Int dim = 0; dim < 9; dim++)
                      d_BLASWrapperPtr->xgemm(
                        'N',
                        'N',
                        d_numberWaveFunctions,
                        numberSphericalFunctions,
                        d_numberNodesPerElement,
                        &one,
                        &X[(iElem - cellRange.first) * d_numberNodesPerElement *
                           d_numberWaveFunctions],
                        d_numberWaveFunctions,
                        &d_DDyadicRMatrixEntriesConjugate
                          [atomId][nonZeroElementMatrixId]
                          [9 * d_kPointIndex * d_numberNodesPerElement *
                             numberSphericalFunctions +
                           dim * d_numberNodesPerElement *
                             numberSphericalFunctions],
                        d_numberNodesPerElement,
                        &one,
                        &d_sphericalFnTimesGradientWavefunDyadicXMatrix
                          [atomId][dim * d_numberWaveFunctions *
                                   numberSphericalFunctions],
                        d_numberWaveFunctions);
                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first !=
        // d_nonlocalElements
        dftfe::uInt iCellBatch = cellRange.first / d_cellsBlockSize;
        // Xpointer not same assert check
        AssertThrow(
          X == d_wfcStartPointerInCellRange[iCellBatch],
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent X called. Make sure the input X is correct."));
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);
        if (d_nonLocalElementsInCellRange[iCellBatch] > 0)
          for (dftfe::Int iDim = 0; iDim < 9; iDim++)
            d_BLASWrapperPtr->xgemmBatched(
              'N',
              'N',
              d_numberWaveFunctions,
              d_maxSingleAtomContribution,
              d_numberNodesPerElement,
              &scalarCoeffAlpha,
              (const ValueType **)deviceWfcPointersInCellRange[iCellBatch],
              d_numberWaveFunctions,
              (const ValueType **)
                devicePointerDdyadicRDaggerInCellRange[iDim][iCellBatch],
              d_numberNodesPerElement,
              &scalarCoeffBeta,
              devicePointerDdyadicRDaggerOutTempInCellRange[iDim][iCellBatch],
              d_numberWaveFunctions,
              d_nonLocalElementsInCellRange[iCellBatch]);
        if (iCellBatch == d_numCellBatches - 1)
          {
            d_sphericalFnTimesGradientWavefunctionDyadicXMatrix.setValue(
              ValueType(0.0));
            for (dftfe::Int iDim = 0; iDim < 9; iDim++)
              {
                dftfe::uInt offsetSrc = iDim * d_numberWaveFunctions *
                                        d_maxSingleAtomContribution *
                                        d_totalNonlocalElems;
                dftfe::uInt offsetDst =
                  iDim * d_numberWaveFunctions * d_totalNonLocalEntries;
                dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
                  assembleAtomLevelContributionsFromCellLevel(
                    d_numberWaveFunctions,
                    d_totalNonlocalElems,
                    d_maxSingleAtomContribution,
                    d_totalNonLocalEntries,
                    d_sphericalFnTimesRDyadicGradientVectorAllCellsDevice,
                    d_mapSphericalFnTimesVectorAllCellsReductionDevice,
                    d_sphericalFnTimesGradientWavefunctionDyadicXMatrix,
                    offsetSrc,
                    offsetDst);
              }
          }
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCconjtransOnX(
    const ValueType                          *X,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange)
  {
    Assert(
      !d_useGlobalCMatrix,
      dealii::ExcMessage(
        "DFT-FE Error: applyCconjtransOnX() is called for cell level C matrix route without it being initialised "));
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType   zero(0.0), one(1.0);
        const dftfe::uInt inc                            = 1;
        d_AllReduceCompleted                             = false;
        dftfe::Int                      numberOfElements = d_locallyOwnedCells;
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();

        double integValue = 0.0;
        for (dftfe::Int iElem = cellRange.first; iElem < cellRange.second;
             iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    dftfe::uInt       atomId = atomIdsInElement[iAtom];
                    dftfe::uInt       Znum   = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];
                    d_BLASWrapperPtr->xgemm(
                      'N',
                      'N',
                      d_numberWaveFunctions,
                      numberSphericalFunctions,
                      d_numberNodesPerElement,
                      &one,
                      &X[(iElem - cellRange.first) * d_numberNodesPerElement *
                         d_numberWaveFunctions],
                      d_numberWaveFunctions,
                      &d_CMatrixEntriesConjugate[atomId][nonZeroElementMatrixId]
                                                [d_kPointIndex *
                                                 d_numberNodesPerElement *
                                                 numberSphericalFunctions],
                      d_numberNodesPerElement,
                      &one,
                      &d_sphericalFnTimesWavefunMatrix[atomId][0],
                      d_numberWaveFunctions);
                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first !=
        // d_nonlocalElements
        dftfe::uInt iCellBatch = cellRange.first / d_cellsBlockSize;
        // Xpointer not same assert check
        AssertThrow(
          X == d_wfcStartPointerInCellRange[iCellBatch],
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent X called. Make sure the input X is correct."));
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);
        if (d_nonLocalElementsInCellRange[iCellBatch] > 0)
          d_BLASWrapperPtr->xgemmBatched(
            'N',
            'N',
            d_numberWaveFunctions,
            d_maxSingleAtomContribution,
            d_numberNodesPerElement,
            &scalarCoeffAlpha,
            (const ValueType **)deviceWfcPointersInCellRange[iCellBatch],
            d_numberWaveFunctions,
            (const ValueType **)devicePointerCDaggerInCellRange[iCellBatch],
            d_numberNodesPerElement,
            &scalarCoeffBeta,
            devicePointerCDaggerOutTempInCellRange[iCellBatch],
            d_numberWaveFunctions,
            d_nonLocalElementsInCellRange[iCellBatch]);
        if (iCellBatch == d_numCellBatches - 1)
          {
            d_sphericalFnTimesWavefunctionMatrix.setValue(ValueType(0.0));
            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              assembleAtomLevelContributionsFromCellLevel(
                d_numberWaveFunctions,
                d_totalNonlocalElems,
                d_maxSingleAtomContribution,
                d_totalNonLocalEntries,
                d_sphericalFnTimesVectorAllCellsDevice,
                d_mapSphericalFnTimesVectorAllCellsReductionDevice,
                d_sphericalFnTimesWavefunctionMatrix);
          }
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCconjtransOnX(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &X)
  {
    Assert(
      d_useGlobalCMatrix,
      dealii::ExcMessage(
        "DFT-FE Error: applyCconjtransOnX() is called for global C matrix route without it being initialised "));

    const ValueType scalarCoeffAlpha = ValueType(1.0),
                    scalarCoeffBeta  = ValueType(0.0);
    const char        transA = 'N', transB = 'N';
    const char        doTransMatrix = 'C';
    const dftfe::uInt inc           = 1;
    for (dftfe::uInt iAtomicNum = 0; iAtomicNum < d_setOfAtomicNumber.size();
         iAtomicNum++)
      {
        dftfe::uInt Znum = *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
        dftfe::uInt numSphFunc =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        dftfe::uInt numAtomsPerSpecies = d_listOfiAtomInSpecies[Znum].size();

        dftfe::uInt totalAtomicWaveFunctions = numSphFunc * numAtomsPerSpecies;
        d_dotProductAtomicWaveInputWaveTemp[iAtomicNum].setValue(0.0);
        if (totalAtomicWaveFunctions > 0)
          {
            d_BLASWrapperPtr->xgemm(
              transA,
              doTransMatrix,
              d_numberWaveFunctions,
              totalAtomicWaveFunctions,
              d_totalLocallyOwnedNodes,
              &scalarCoeffAlpha,
              X.data(), // assumes the constraint.distribute() has been called
              d_numberWaveFunctions,
              d_CMatrixGlobal[d_kPointIndex][iAtomicNum].data(),
              totalAtomicWaveFunctions,
              &scalarCoeffBeta,
              d_dotProductAtomicWaveInputWaveTemp[iAtomicNum].data(),
              d_numberWaveFunctions);
          }
        std::vector<dftfe::uInt> atomIdsInCurrentProcess =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();

        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            for (dftfe::uInt atomIndex = 0; atomIndex < numAtomsPerSpecies;
                 atomIndex++)
              {
                dftfe::uInt iAtom  = d_listOfiAtomInSpecies[Znum][atomIndex];
                dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
                d_BLASWrapperPtr->xcopy(
                  numSphFunc * d_numberWaveFunctions,
                  &d_dotProductAtomicWaveInputWaveTemp[iAtomicNum]
                                                      [atomIndex * numSphFunc *
                                                       d_numberWaveFunctions],
                  1,
                  &d_sphericalFnTimesWavefunMatrix[atomId][0],
                  1);
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            for (dftfe::uInt atomIndex = 0; atomIndex < numAtomsPerSpecies;
                 atomIndex++)
              {
                dftfe::uInt iAtom  = d_listOfiAtomInSpecies[Znum][atomIndex];
                dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];

                d_BLASWrapperPtr->xcopy(
                  numSphFunc * d_numberWaveFunctions,
                  &d_dotProductAtomicWaveInputWaveTemp[iAtomicNum]
                                                      [atomIndex * numSphFunc *
                                                       d_numberWaveFunctions],
                  1,
                  d_sphericalFnTimesWavefunctionMatrix.begin() +
                    d_mapiAtomTosphFuncWaveStart[iAtom] * d_numberWaveFunctions,
                  1);
              }
          }
#endif
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCVCconjtransOnX(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    const dftfe::uInt                                          kPointIndex,
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      &sphericalFunctionKetTimesVectorParFlattened,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &dst)
  {
    const dftfe::uInt inc = 1;
    applyVCconjtransOnX(src,
                        kPointIndex,
                        couplingtype,
                        couplingMatrix,
                        sphericalFunctionKetTimesVectorParFlattened,
                        true);

    if (!d_useGlobalCMatrix)
      {
        dftfe::utils::MemoryStorage<ValueType, memorySpace> Xtemp;
        Xtemp.resize(d_locallyOwnedCells * d_numberNodesPerElement *
                       d_numberWaveFunctions,
                     0.0);
        applyCOnVCconjtransX(Xtemp.data(),
                             std::make_pair(0, d_locallyOwnedCells));
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            for (dftfe::uInt iCell = 0; iCell < d_locallyOwnedCells; ++iCell)
              {
                for (dftfe::uInt iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    dealii::types::global_dof_index localNodeId =
                      (d_basisOperatorPtr->d_cellDofIndexToProcessDofIndexMap
                         [iCell * d_numberNodesPerElement + iNode]) *
                      d_numberWaveFunctions;
                    d_BLASWrapperPtr->xcopy(
                      d_numberWaveFunctions,
                      &Xtemp[iCell * d_numberNodesPerElement *
                               d_numberWaveFunctions +
                             iNode * d_numberWaveFunctions],
                      inc,
                      dst.data() + localNodeId,
                      inc);
                  }
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            Assert(
              d_basisOperatorPtr->nVectors() == d_numberWaveFunctions,
              dealii::ExcMessage(
                "DFT-FE Error: d_BasisOperatorMemPtr in Atomic non local operator is not set with correct input size."));


            d_BLASWrapperPtr->stridedCopyFromBlock(
              d_numberWaveFunctions,
              d_locallyOwnedCells * d_numberNodesPerElement,
              Xtemp.begin(),
              dst.data(),
              d_basisOperatorPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                .begin());
          }
#endif
      }
    else
      {
        applyCOnVCconjtransX(dst);
      }
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyVCconjtransOnX(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    const dftfe::uInt                                          kPointIndex,
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
              &sphericalFunctionKetTimesVectorParFlattened,
    const bool flagScaleInternalMatrix)
  {
    if (!d_useGlobalCMatrix)
      {
        applyVCconjtransOnXCellLevel(
          src,
          kPointIndex,
          couplingtype,
          couplingMatrix,
          sphericalFunctionKetTimesVectorParFlattened,
          flagScaleInternalMatrix);
      }
    else
      {
        applyVCconjtransOnXUsingGlobalC(
          src,
          kPointIndex,
          couplingtype,
          couplingMatrix,
          sphericalFunctionKetTimesVectorParFlattened,
          flagScaleInternalMatrix);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    applyVCconjtransOnXCellLevel(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      const dftfe::uInt                                          kPointIndex,
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
                &sphericalFunctionKetTimesVectorParFlattened,
      const bool flagScaleInternalMatrix)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        initialiseOperatorActionOnX(kPointIndex);
        sphericalFunctionKetTimesVectorParFlattened.setValue(0.0);

        const dftfe::uInt inc = 1;
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
          cellWaveFunctionMatrix;

        cellWaveFunctionMatrix.resize(d_numberNodesPerElement *
                                        d_numberWaveFunctions,
                                      0.0);


        if (d_totalNonlocalElems)
          {
            for (dftfe::uInt iCell = 0; iCell < d_locallyOwnedCells; ++iCell)
              {
                if (atomSupportInElement(iCell))
                  {
                    for (dftfe::uInt iNode = 0; iNode < d_numberNodesPerElement;
                         ++iNode)
                      {
                        dealii::types::global_dof_index localNodeId =
                          (d_basisOperatorPtr
                             ->d_cellDofIndexToProcessDofIndexMap
                               [iCell * d_numberNodesPerElement + iNode]) *
                          d_numberWaveFunctions;
                        d_BLASWrapperPtr->xcopy(
                          d_numberWaveFunctions,
                          src.data() + localNodeId,
                          inc,
                          &cellWaveFunctionMatrix[d_numberWaveFunctions *
                                                  iNode],
                          inc);

                      } // Cell Extraction

                    applyCconjtransOnX(
                      cellWaveFunctionMatrix.data(),
                      std::pair<dftfe::uInt, dftfe::uInt>(iCell, iCell + 1));

                  } // if nonlocalAtomPResent
              }     // Cell Loop
            applyAllReduceOnCconjtransX(
              sphericalFunctionKetTimesVectorParFlattened);
            applyVOnCconjtransX(couplingtype,
                                couplingMatrix,
                                sphericalFunctionKetTimesVectorParFlattened,
                                false);



          } // nonlocal
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        initialiseOperatorActionOnX(kPointIndex);
        dftfe::utils::MemoryStorage<ValueType,
                                    dftfe::utils::MemorySpace::DEVICE>
          cellWaveFunctionMatrix;
        cellWaveFunctionMatrix.resize(d_locallyOwnedCells *
                                        d_numberNodesPerElement *
                                        d_numberWaveFunctions,
                                      0.0);
        initialiseCellWaveFunctionPointers(cellWaveFunctionMatrix,
                                           d_locallyOwnedCells);
        if (d_totalNonlocalElems > 0)
          {
            Assert(
              d_basisOperatorPtr->nVectors() == d_numberWaveFunctions,
              dealii::ExcMessage(
                "DFT-FE Error: d_BasisOperatorMemPtr in Atomic non local operator is not set with correct input size."));


            d_BLASWrapperPtr->stridedCopyToBlock(
              d_numberWaveFunctions,
              d_locallyOwnedCells * d_numberNodesPerElement,
              src.data(),
              cellWaveFunctionMatrix.begin(),
              d_basisOperatorPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                .begin());
            applyCconjtransOnX(
              cellWaveFunctionMatrix.data(),
              std::pair<dftfe::uInt, dftfe::uInt>(0, d_locallyOwnedCells));
          }

        sphericalFunctionKetTimesVectorParFlattened.setValue(0);
        applyAllReduceOnCconjtransX(
          sphericalFunctionKetTimesVectorParFlattened);

        applyVOnCconjtransX(couplingtype,
                            couplingMatrix,
                            sphericalFunctionKetTimesVectorParFlattened,
                            false);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    applyVCconjtransOnXUsingGlobalC(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      const dftfe::uInt                                          kPointIndex,
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
                &sphericalFunctionKetTimesVectorParFlattened,
      const bool flagScaleInternalMatrix)
  {
    initialiseOperatorActionOnX(kPointIndex);
    sphericalFunctionKetTimesVectorParFlattened.setValue(0.0);
    applyCconjtransOnX(src);
    applyAllReduceOnCconjtransX(sphericalFunctionKetTimesVectorParFlattened);

    applyVOnCconjtransX(couplingtype,
                        couplingMatrix,
                        sphericalFunctionKetTimesVectorParFlattened,
                        false);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCOnVCconjtransX(
    ValueType                                *Xout,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange)
  {
    Assert(
      !d_useGlobalCMatrix,
      dealii::ExcMessage(
        "DFT-FE Error: applyCOnVCconjtransX() is called for cell level C matrix route without it being initialised "));
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType   zero(0.0), one(1.0);
        const dftfe::uInt inc = 1;
        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        for (dftfe::Int iElem = cellRange.first; iElem < cellRange.second;
             iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);


                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    dftfe::uInt atomId = atomIdsInElement[iAtom];

                    dftfe::uInt       Znum = atomicNumber[atomId];
                    const dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];
                    d_BLASWrapperPtr->xgemm(
                      'N',
                      'N',
                      d_numberWaveFunctions,
                      d_numberNodesPerElement,
                      numberSphericalFunctions,
                      &one,
                      &d_sphericalFnTimesWavefunMatrix[atomId][0],
                      d_numberWaveFunctions,
                      &d_CMatrixEntriesTranspose[atomId][nonZeroElementMatrixId]
                                                [d_kPointIndex *
                                                 d_numberNodesPerElement *
                                                 numberSphericalFunctions],
                      numberSphericalFunctions,
                      &one,
                      &Xout[(iElem - cellRange.first) *
                            d_numberNodesPerElement * d_numberWaveFunctions],
                      d_numberWaveFunctions);

                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first !=
        // d_nonlocalElements AssertThrow(
        //   cellRange.second - cellRange.first == d_locallyOwnedCells,
        //   dealii::ExcMessage(
        //     "DFT-FE Error: Inconsistent cellRange in use. All the nonlocal
        //     Cells must be in range."));
        dftfe::uInt iCellBatch = cellRange.first / d_cellsBlockSize;
        dftfe::uInt numberOfNonLocalElementsInRange =
          d_nonLocalElementsInCellRange[iCellBatch];
        // pcout<<"iCellBatch out of NumBatches: "<<iCellBatch<<"
        // "<<d_numCellBatches<<std::endl;
        long long int strideA =
          d_numberWaveFunctions * d_maxSingleAtomContribution;
        long long int strideB =
          d_maxSingleAtomContribution * d_numberNodesPerElement;
        long long int strideC = d_numberWaveFunctions * d_numberNodesPerElement;
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);
        dftfe::uInt elementIndex         = 0;
        if (iCellBatch > 0)
          {
            for (dftfe::uInt iCell = 0; iCell < iCellBatch; ++iCell)
              {
                elementIndex += d_nonLocalElementsInCellRange[iCell];
              }
          }
        if (numberOfNonLocalElementsInRange > 0)
          {
            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              d_numberWaveFunctions,
              d_numberNodesPerElement,
              d_maxSingleAtomContribution,
              &scalarCoeffAlpha,
              d_sphericalFnTimesVectorAllCellsDevice.begin() +
                elementIndex * d_numberWaveFunctions *
                  d_maxSingleAtomContribution,
              d_numberWaveFunctions,
              strideA,
              d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
                  .begin() +
                elementIndex * d_numberNodesPerElement *
                  d_maxSingleAtomContribution,
              d_maxSingleAtomContribution,
              strideB,
              &scalarCoeffBeta,
              d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin() +
                elementIndex * d_numberNodesPerElement * d_numberWaveFunctions,
              d_numberWaveFunctions,
              strideC,
              numberOfNonLocalElementsInRange);

            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              addNonLocalContribution(
                numberOfNonLocalElementsInRange,
                cellRange.first,
                elementIndex,
                d_numberWaveFunctions,
                d_numberNodesPerElement,
                d_iElemNonLocalToElemIndexMap,
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice,
                Xout);
          }
      }
#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCOnVCconjtransX(
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &Xout)
  {
    Assert(
      d_useGlobalCMatrix,
      dealii::ExcMessage(
        "DFT-FE Error: applyCOnVCconjtransX() is called for global C matrix route without it being initialised "));

    const ValueType scalarCoeffAlpha = ValueType(1.0),
                    scalarCoeffBeta  = ValueType(1.0);
    const char        transA = 'N', transB = 'N';
    const char        doTransMatrix = 'C';
    const dftfe::uInt inc           = 1;

    std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    for (dftfe::uInt iAtomicNum = 0; iAtomicNum < d_setOfAtomicNumber.size();
         iAtomicNum++)
      {
        dftfe::uInt Znum = *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
        dftfe::uInt numSphFunc =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        dftfe::uInt numAtomsPerSpecies = d_listOfiAtomInSpecies[Znum].size();
        ;

        dftfe::uInt totalAtomicWaveFunctions = numSphFunc * numAtomsPerSpecies;

        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            for (dftfe::uInt atomIndex = 0; atomIndex < numAtomsPerSpecies;
                 atomIndex++)
              {
                dftfe::uInt iAtom  = d_listOfiAtomInSpecies[Znum][atomIndex];
                dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
                d_BLASWrapperPtr->xcopy(
                  numSphFunc * d_numberWaveFunctions,
                  &d_sphericalFnTimesWavefunMatrix[atomId][0],
                  1,
                  &d_dotProductAtomicWaveInputWaveTemp[iAtomicNum]
                                                      [atomIndex * numSphFunc *
                                                       d_numberWaveFunctions],
                  1);
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            for (dftfe::uInt atomIndex = 0; atomIndex < numAtomsPerSpecies;
                 atomIndex++)
              {
                dftfe::uInt iAtom  = d_listOfiAtomInSpecies[Znum][atomIndex];
                dftfe::uInt atomId = atomIdsInCurrentProcess[iAtom];
                d_BLASWrapperPtr->xcopy(
                  numSphFunc * d_numberWaveFunctions,
                  d_couplingMatrixTimesVectorDevice.begin() +
                    iAtom * d_maxSingleAtomContribution * d_numberWaveFunctions,
                  1,
                  &d_dotProductAtomicWaveInputWaveTemp[iAtomicNum]
                                                      [atomIndex * numSphFunc *
                                                       d_numberWaveFunctions],
                  1);
              }
          }
#endif
        if (totalAtomicWaveFunctions > 0)
          {
            d_BLASWrapperPtr->xgemm(
              transA,
              transB,
              d_numberWaveFunctions,
              d_totalLocallyOwnedNodes,
              totalAtomicWaveFunctions,
              &scalarCoeffAlpha,
              d_dotProductAtomicWaveInputWaveTemp[iAtomicNum].data(),
              d_numberWaveFunctions,
              d_CMatrixGlobal[d_kPointIndex][iAtomicNum].data(),
              totalAtomicWaveFunctions,
              &scalarCoeffBeta,
              Xout.data(), // directly add to the output
              d_numberWaveFunctions);
          }
      }
  }

#if defined(DFTFE_WITH_DEVICE)
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                       &cellWaveFunctionMatrix,
      const dftfe::uInt cellsBlockSize,
      const std::vector<nonLocalContractionVectorType>
        NonLocalContractionVectorType)
  {
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        d_cellsBlockSize = cellsBlockSize;
        freeDeviceVectors(NonLocalContractionVectorType);
        const dftfe::uInt numCells       = d_locallyOwnedCells;
        dftfe::uInt       numCellBatches = d_locallyOwnedCells / cellsBlockSize;
        const dftfe::uInt cellRemSize    = d_locallyOwnedCells % cellsBlockSize;
        if (cellRemSize > 0)
          numCellBatches += 1;
        d_numCellBatches = numCellBatches;

        d_nonLocalElementsInCellRange.clear();
        d_nonLocalElementsInCellRange.resize(numCellBatches, 0);
        d_wfcStartPointerInCellRange.clear();
        d_wfcStartPointerInCellRange.resize(numCellBatches);
        for (dftfe::uInt iCellBatch = 0; iCellBatch < numCellBatches;
             ++iCellBatch)
          {
            dftfe::uInt startCell = iCellBatch * cellsBlockSize;
            dftfe::uInt endCell =
              std::min(startCell + cellsBlockSize, numCells);
            for (dftfe::uInt iCell = startCell; iCell < endCell; ++iCell)
              {
                const std::vector<dftfe::Int> &atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iCell);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                d_nonLocalElementsInCellRange[iCellBatch] +=
                  numOfAtomsInElement;
              }
            d_wfcStartPointerInCellRange[iCellBatch] =
              cellWaveFunctionMatrix.begin() +
              (d_memoryOptMode ?
                 0 :
                 startCell * d_numberNodesPerElement * d_numberWaveFunctions);
          }
        freeDeviceVectors(NonLocalContractionVectorType);

        hostWfcPointersInCellRange.clear();
        deviceWfcPointersInCellRange.clear();
        hostWfcPointersInCellRange.resize(numCellBatches);
        deviceWfcPointersInCellRange.resize(numCellBatches);

        for (dftfe::Int iVec = 0; iVec < NonLocalContractionVectorType.size();
             iVec++)
          {
            if ((NonLocalContractionVectorType[iVec] ==
                 nonLocalContractionVectorType::CconjTransX) &&
                !d_useGlobalCMatrix)
              {
                hostPointerCDaggerOutTempInCellRange.clear();
                hostPointerCDaggerInCellRange.clear();
                devicePointerCDaggerInCellRange.clear();
                devicePointerCDaggerOutTempInCellRange.clear();

                hostPointerCDaggerInCellRange.resize(numCellBatches);
                hostPointerCDaggerOutTempInCellRange.resize(numCellBatches);
                devicePointerCDaggerInCellRange.resize(numCellBatches);
                devicePointerCDaggerOutTempInCellRange.resize(numCellBatches);
              }
            else if (NonLocalContractionVectorType[iVec] ==
                     nonLocalContractionVectorType::CRconjTransX)
              {
                hostPointerCRDaggerOutTempInCellRange.clear();
                hostPointerCRDaggerInCellRange.clear();
                devicePointerCRDaggerInCellRange.clear();
                devicePointerCRDaggerOutTempInCellRange.clear();

                hostPointerCRDaggerOutTempInCellRange.resize(3);
                hostPointerCRDaggerInCellRange.resize(3);
                devicePointerCRDaggerInCellRange.resize(3);
                devicePointerCRDaggerOutTempInCellRange.resize(3);

                for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                  {
                    hostPointerCRDaggerInCellRange[iDim].resize(numCellBatches);
                    hostPointerCRDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerCRDaggerInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerCRDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                  }
              }
            else if (NonLocalContractionVectorType[iVec] ==
                     nonLocalContractionVectorType::DconjTransX)
              {
                hostPointerDDaggerOutTempInCellRange.clear();
                hostPointerDDaggerInCellRange.clear();
                devicePointerDDaggerInCellRange.clear();
                devicePointerDDaggerOutTempInCellRange.clear();

                hostPointerDDaggerOutTempInCellRange.resize(3);
                hostPointerDDaggerInCellRange.resize(3);
                devicePointerDDaggerInCellRange.resize(3);
                devicePointerDDaggerOutTempInCellRange.resize(3);

                for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                  {
                    hostPointerDDaggerInCellRange[iDim].resize(numCellBatches);
                    hostPointerDDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerDDaggerInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerDDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                  }
              }
            else if (NonLocalContractionVectorType[iVec] ==
                     nonLocalContractionVectorType::DDyadicRconjTransX)
              {
                hostPointerDdyadicRDaggerOutTempInCellRange.clear();
                hostPointerDdyadicRDaggerInCellRange.clear();
                devicePointerDdyadicRDaggerInCellRange.clear();
                devicePointerDdyadicRDaggerOutTempInCellRange.clear();

                hostPointerDdyadicRDaggerOutTempInCellRange.resize(9);
                hostPointerDdyadicRDaggerInCellRange.resize(9);
                devicePointerDdyadicRDaggerInCellRange.resize(9);
                devicePointerDdyadicRDaggerOutTempInCellRange.resize(9);

                for (dftfe::Int iDim = 0; iDim < 9; iDim++)
                  {
                    hostPointerDdyadicRDaggerInCellRange[iDim].resize(
                      numCellBatches);
                    hostPointerDdyadicRDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerDdyadicRDaggerInCellRange[iDim].resize(
                      numCellBatches);
                    devicePointerDdyadicRDaggerOutTempInCellRange[iDim].resize(
                      numCellBatches);
                  }
              }
          }


        for (dftfe::uInt iCellBatch = 0; iCellBatch < numCellBatches;
             ++iCellBatch)
          {
            const dftfe::uInt nonLocalElements =
              d_nonLocalElementsInCellRange[iCellBatch];
            hostWfcPointersInCellRange[iCellBatch] =
              (ValueType **)malloc(nonLocalElements * sizeof(ValueType *));
            dftfe::utils::deviceMalloc(
              (void **)&deviceWfcPointersInCellRange[iCellBatch],
              nonLocalElements * sizeof(ValueType *));
            for (dftfe::Int iVec = 0;
                 iVec < NonLocalContractionVectorType.size();
                 iVec++)
              {
                if ((NonLocalContractionVectorType[iVec] ==
                     nonLocalContractionVectorType::CconjTransX) &&
                    !d_useGlobalCMatrix)
                  {
                    hostPointerCDaggerInCellRange[iCellBatch] =
                      (ValueType **)malloc(nonLocalElements *
                                           sizeof(ValueType *));
                    hostPointerCDaggerOutTempInCellRange[iCellBatch] =
                      (ValueType **)malloc(nonLocalElements *
                                           sizeof(ValueType *));
                    dftfe::utils::deviceMalloc(
                      (void **)&devicePointerCDaggerInCellRange[iCellBatch],
                      nonLocalElements * sizeof(ValueType *));
                    dftfe::utils::deviceMalloc(
                      (void *
                         *)&devicePointerCDaggerOutTempInCellRange[iCellBatch],
                      nonLocalElements * sizeof(ValueType *));
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::CRconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                      {
                        hostPointerCRDaggerInCellRange[iDim][iCellBatch] =
                          (ValueType **)malloc(nonLocalElements *
                                               sizeof(ValueType *));
                        hostPointerCRDaggerOutTempInCellRange
                          [iDim][iCellBatch] = (ValueType **)malloc(
                            nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void *
                             *)&devicePointerCRDaggerInCellRange[iDim]
                                                                [iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void **)&devicePointerCRDaggerOutTempInCellRange
                            [iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::DconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                      {
                        hostPointerDDaggerInCellRange[iDim][iCellBatch] =
                          (ValueType **)malloc(nonLocalElements *
                                               sizeof(ValueType *));
                        hostPointerDDaggerOutTempInCellRange[iDim][iCellBatch] =
                          (ValueType **)malloc(nonLocalElements *
                                               sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void **)&devicePointerDDaggerInCellRange[iDim]
                                                                   [iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void **)&devicePointerDDaggerOutTempInCellRange
                            [iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::DDyadicRconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 9; iDim++)
                      {
                        hostPointerDdyadicRDaggerInCellRange[iDim][iCellBatch] =
                          (ValueType **)malloc(nonLocalElements *
                                               sizeof(ValueType *));
                        hostPointerDdyadicRDaggerOutTempInCellRange
                          [iDim][iCellBatch] = (ValueType **)malloc(
                            nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void **)&devicePointerDdyadicRDaggerInCellRange
                            [iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMalloc(
                          (void *
                             *)&devicePointerDdyadicRDaggerOutTempInCellRange
                            [iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
              }

            dftfe::uInt startCell = iCellBatch * cellsBlockSize;
            dftfe::uInt endCell =
              std::min(startCell + cellsBlockSize, numCells);
            dftfe::uInt i = 0;
            for (dftfe::uInt iCell = startCell; iCell < endCell; ++iCell)
              {
                const std::vector<dftfe::Int> &atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iCell);
                dftfe::Int numOfAtomsInElement = atomIdsInElement.size();
                for (dftfe::Int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    const dftfe::uInt atomId = atomIdsInElement[iAtom];

                    dftfe::uInt countElem = 0;
                    auto        it        = std::find_if(
                      d_elementIdToNonLocalElementIdMap[iCell].begin(),
                      d_elementIdToNonLocalElementIdMap[iCell].end(),
                      [&atomId](const std::pair<dftfe::uInt, dftfe::uInt> &p) {
                        return p.first == atomId;
                      });
                    if (it != d_elementIdToNonLocalElementIdMap[iCell].end())
                      countElem = it->second;
                    else
                      {
                        AssertThrow(
                          false,
                          dealii::ExcMessage(
                            "DFT-FE Error: Inconsistent element id to nonlocal element id map."));
                      }

                    hostWfcPointersInCellRange[iCellBatch][i] =
                      cellWaveFunctionMatrix.begin() +
                      (d_memoryOptMode ? iCell - startCell : iCell) *
                        d_numberNodesPerElement * d_numberWaveFunctions;
                    for (dftfe::Int iVec = 0;
                         iVec < NonLocalContractionVectorType.size();
                         iVec++)
                      {
                        if ((NonLocalContractionVectorType[iVec] ==
                             nonLocalContractionVectorType::CconjTransX) &&
                            !d_useGlobalCMatrix)
                          {
                            hostPointerCDaggerInCellRange[iCellBatch][i] =
                              d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
                                .data() +
                              countElem * d_numberNodesPerElement *
                                d_maxSingleAtomContribution;
                            hostPointerCDaggerOutTempInCellRange
                              [iCellBatch][i] =
                                d_sphericalFnTimesVectorAllCellsDevice.data() +
                                countElem * d_numberWaveFunctions *
                                  d_maxSingleAtomContribution;
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                 nonLocalContractionVectorType::CRconjTransX)
                          {
                            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                              {
                                hostPointerCRDaggerInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_IntegralFEMShapeFunctionValueTimesXTimesAtomicSphericalFunctionConjugateDevice
                                      .data() +
                                    countElem * d_numberNodesPerElement *
                                      d_maxSingleAtomContribution +
                                    iDim * d_numberNodesPerElement *
                                      d_maxSingleAtomContribution *
                                      d_totalNonlocalElems;
                                hostPointerCRDaggerOutTempInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_sphericalFnTimesXTimesVectorAllCellsDevice
                                      .data() +
                                    countElem * d_numberWaveFunctions *
                                      d_maxSingleAtomContribution +
                                    iDim * d_numberWaveFunctions *
                                      d_totalNonlocalElems *
                                      d_maxSingleAtomContribution;
                              }
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                 nonLocalContractionVectorType::DconjTransX)
                          {
                            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                              {
                                hostPointerDDaggerInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_IntegralGradientFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
                                      .data() +
                                    countElem * d_numberNodesPerElement *
                                      d_maxSingleAtomContribution +
                                    iDim * d_numberNodesPerElement *
                                      d_maxSingleAtomContribution *
                                      d_totalNonlocalElems;
                                hostPointerDDaggerOutTempInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_sphericalFnTimesGradientVectorAllCellsDevice
                                      .data() +
                                    countElem * d_numberWaveFunctions *
                                      d_maxSingleAtomContribution +
                                    iDim * d_numberWaveFunctions *
                                      d_totalNonlocalElems *
                                      d_maxSingleAtomContribution;
                              }
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                 nonLocalContractionVectorType::
                                   DDyadicRconjTransX)
                          {
                            for (dftfe::Int iDim = 0; iDim < 9; iDim++)
                              {
                                hostPointerDdyadicRDaggerInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_IntegralGradientFEMShapeFunctionValueDyadicAtomicSphericalFunctionTimesRConjugateDevice
                                      .data() +
                                    (countElem * d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     iDim * d_numberNodesPerElement *
                                       d_maxSingleAtomContribution *
                                       d_totalNonlocalElems);
                                hostPointerDdyadicRDaggerOutTempInCellRange
                                  [iDim][iCellBatch][i] =
                                    d_sphericalFnTimesRDyadicGradientVectorAllCellsDevice
                                      .data() +
                                    countElem * d_numberWaveFunctions *
                                      d_maxSingleAtomContribution +
                                    iDim * d_numberWaveFunctions *
                                      d_totalNonlocalElems *
                                      d_maxSingleAtomContribution;
                              }
                          }
                      }


                    i++;
                  } // iAtom
              }     // iCell
            dftfe::utils::deviceMemcpyH2D(
              deviceWfcPointersInCellRange[iCellBatch],
              hostWfcPointersInCellRange[iCellBatch],
              nonLocalElements * sizeof(ValueType *));
            for (dftfe::Int iVec = 0;
                 iVec < NonLocalContractionVectorType.size();
                 iVec++)
              {
                if ((NonLocalContractionVectorType[iVec] ==
                     nonLocalContractionVectorType::CconjTransX) &&
                    !d_useGlobalCMatrix)
                  {
                    dftfe::utils::deviceMemcpyH2D(
                      devicePointerCDaggerInCellRange[iCellBatch],
                      hostPointerCDaggerInCellRange[iCellBatch],
                      nonLocalElements * sizeof(ValueType *));
                    dftfe::utils::deviceMemcpyH2D(
                      devicePointerCDaggerOutTempInCellRange[iCellBatch],
                      hostPointerCDaggerOutTempInCellRange[iCellBatch],
                      nonLocalElements * sizeof(ValueType *));
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::CRconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                      {
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerCRDaggerInCellRange[iDim][iCellBatch],
                          hostPointerCRDaggerInCellRange[iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerCRDaggerOutTempInCellRange[iDim]
                                                                 [iCellBatch],
                          hostPointerCRDaggerOutTempInCellRange[iDim]
                                                               [iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::DconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                      {
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerDDaggerInCellRange[iDim][iCellBatch],
                          hostPointerDDaggerInCellRange[iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerDDaggerOutTempInCellRange[iDim]
                                                                [iCellBatch],
                          hostPointerDDaggerOutTempInCellRange[iDim]
                                                              [iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
                else if (NonLocalContractionVectorType[iVec] ==
                         nonLocalContractionVectorType::DDyadicRconjTransX)
                  {
                    for (dftfe::Int iDim = 0; iDim < 9; iDim++)
                      {
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerDdyadicRDaggerInCellRange[iDim]
                                                                [iCellBatch],
                          hostPointerDdyadicRDaggerInCellRange[iDim]
                                                              [iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                        dftfe::utils::deviceMemcpyH2D(
                          devicePointerDdyadicRDaggerOutTempInCellRange
                            [iDim][iCellBatch],
                          hostPointerDdyadicRDaggerOutTempInCellRange
                            [iDim][iCellBatch],
                          nonLocalElements * sizeof(ValueType *));
                      }
                  }
              }

          } // iCellBatch
        d_isMallocCalled  = true;
        d_wfcStartPointer = cellWaveFunctionMatrix.begin();
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::freeDeviceVectors(
    const std::vector<nonLocalContractionVectorType>
      NonLocalContractionVectorType)
  {
    if (!d_useGlobalCMatrix)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            if (d_isMallocCalled)
              {
                for (dftfe::uInt iCellBatch = 0; iCellBatch < d_numCellBatches;
                     ++iCellBatch)
                  {
                    free(hostWfcPointersInCellRange[iCellBatch]);
                    dftfe::utils::deviceFree(
                      deviceWfcPointersInCellRange[iCellBatch]);
                  }

                for (int iVec = 0; iVec < NonLocalContractionVectorType.size();
                     ++iVec)
                  {
                    for (dftfe::uInt iCellBatch = 0;
                         iCellBatch < d_numCellBatches;
                         ++iCellBatch)
                      {
                        if (NonLocalContractionVectorType[iVec] ==
                            nonLocalContractionVectorType::CconjTransX)
                          {
                            free(hostPointerCDaggerInCellRange[iCellBatch]);
                            free(
                              hostPointerCDaggerOutTempInCellRange[iCellBatch]);
                            dftfe::utils::deviceFree(
                              devicePointerCDaggerInCellRange[iCellBatch]);
                            dftfe::utils::deviceFree(
                              devicePointerCDaggerOutTempInCellRange
                                [iCellBatch]);
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                 nonLocalContractionVectorType::CRconjTransX)
                          {
                            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                              {
                                free(
                                  hostPointerCRDaggerInCellRange[iDim]
                                                                [iCellBatch]);
                                free(hostPointerCRDaggerOutTempInCellRange
                                       [iDim][iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerCRDaggerInCellRange[iDim]
                                                                  [iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerCRDaggerOutTempInCellRange
                                    [iDim][iCellBatch]);
                              }
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                   nonLocalContractionVectorType::DconjTransX &&
                                 d_computeIonForces)
                          {
                            for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                              {
                                free(hostPointerDDaggerInCellRange[iDim]
                                                                  [iCellBatch]);
                                free(hostPointerDDaggerOutTempInCellRange
                                       [iDim][iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerDDaggerInCellRange[iDim]
                                                                 [iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerDDaggerOutTempInCellRange
                                    [iDim][iCellBatch]);
                              }
                          }
                        else if (NonLocalContractionVectorType[iVec] ==
                                   nonLocalContractionVectorType::
                                     DDyadicRconjTransX &&
                                 d_computeCellStress)
                          {
                            for (dftfe::Int iDim = 0; iDim < 9; iDim++)
                              {
                                free(hostPointerDdyadicRDaggerInCellRange
                                       [iDim][iCellBatch]);
                                free(hostPointerDdyadicRDaggerOutTempInCellRange
                                       [iDim][iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerDdyadicRDaggerInCellRange
                                    [iDim][iCellBatch]);
                                dftfe::utils::deviceFree(
                                  devicePointerDdyadicRDaggerOutTempInCellRange
                                    [iDim][iCellBatch]);
                              }
                          }
                      }
                  }
              }
            d_isMallocCalled = false;
          }
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    copyDistributedVectorToPaddedMemoryStorageVectorDevice(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::utils::MemoryStorage<ValueType, memorySpace> &paddedVector)
  {
    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const dftfe::uInt totalEntries =
      atomIdsInProcessor.size() * d_maxSingleAtomContribution;
    dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
      copyFromDealiiParallelNonLocalVecToPaddedVector(
        d_numberWaveFunctions,
        totalEntries,
        sphericalFunctionKetTimesVectorParFlattened.begin(),
        paddedVector.begin(),
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.begin());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    copyPaddedMemoryStorageVectorToDistributeVectorDevice(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &paddedVector,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const dftfe::uInt totalEntries =
      atomIdsInProcessor.size() * d_maxSingleAtomContribution;
    dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
      copyToDealiiParallelNonLocalVecFromPaddedVector(
        d_numberWaveFunctions,
        totalEntries,
        paddedVector.begin(),
        sphericalFunctionKetTimesVectorParFlattened.begin(),
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.begin());
  }


#endif

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    intitialisePartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex)
  {
    if (updateSparsity)
      initialisePartitioner();
    initKpoints(kPointWeights, kPointCoordinates);
    computeCMatrixEntries(basisOperationsPtr,
                          BLASWrapperHostPtr,
                          quadratureIndex);
    if (d_useGlobalCMatrix)
      computeGlobalCMatrixVector(basisOperationsPtr, BLASWrapperHostPtr);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  template <typename ValueTypeSrc>

  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    copyPartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex,
      const std::shared_ptr<
        AtomicCenteredNonLocalOperator<ValueTypeSrc, memorySpace>>
        nonLocalOperatorSrc)
  {
    if (updateSparsity)
      initialisePartitioner();
    initKpoints(kPointWeights, kPointCoordinates);
    if (d_useGlobalCMatrix)
      {
        copyGlobalCMatrix(nonLocalOperatorSrc,
                          basisOperationsPtr,
                          quadratureIndex);
      }
    else
      {
        copyCMatrixEntries(nonLocalOperatorSrc,
                           basisOperationsPtr,
                           quadratureIndex);
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  template <typename ValueTypeSrc>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::copyGlobalCMatrix(
    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueTypeSrc, memorySpace>>
      nonLocalOperatorSrc,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                      basisOperationsPtr,
    const dftfe::uInt quadratureIndex)
  {
    d_locallyOwnedCells = basisOperationsPtr->nCells();

    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const dftfe::uInt numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    d_numberNodesPerElement    = basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numCells = d_locallyOwnedCells;
    const dftfe::uInt numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomCenteredSphericalFunctionContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomCenteredSphericalFunctionContainer
        ->getPeriodicImageCoordinatesList();
    const dftfe::uInt maxkPoints = d_kPointWeights.size();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomCenteredSphericalFunctionContainer->getSphericalFunctions();


    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);
    d_atomCenteredKpointIndexedSphericalFnQuadValues.clear();
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.clear();
    d_cellIdToAtomIdsLocalCompactSupportMap.clear();
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    d_nonTrivialSphericalFnPerCell.clear();
    d_nonTrivialSphericalFnPerCell.resize(numCells, 0);

    d_nonTrivialSphericalFnsCellStartIndex.clear();
    d_nonTrivialSphericalFnsCellStartIndex.resize(numCells, 0);

    d_atomIdToNonTrivialSphericalFnCellStartIndex.clear();

    std::map<dftfe::uInt, std::vector<dftfe::uInt>>
                             globalAtomIdToNonTrivialSphericalFnsCellStartIndex;
    std::vector<dftfe::uInt> accumTemp(numCells, 0);
    // Loop over atoms to determine sizes of various vectors for forces
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        dftfe::uInt       atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom] =
          std::vector<dftfe::uInt>(numCells, 0);
        globalAtomIdToNonTrivialSphericalFnsCellStartIndex[atomId] =
          std::vector<dftfe::uInt>(numCells, 0);
        for (dftfe::Int iElemComp = 0;
             iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const dftfe::uInt elementId =
              elementIndexesInAtomCompactSupport[iElemComp];

            d_cellIdToAtomIdsLocalCompactSupportMap[elementId].push_back(iAtom);

            d_nonTrivialSphericalFnPerCell[elementId] += numSphericalFunctions;
            d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom][elementId] =
              accumTemp[elementId];
            globalAtomIdToNonTrivialSphericalFnsCellStartIndex
              [atomId][elementId] = accumTemp[elementId];
            accumTemp[elementId] += numSphericalFunctions;
          }
      }

    d_sumNonTrivialSphericalFnOverAllCells =
      std::accumulate(d_nonTrivialSphericalFnPerCell.begin(),
                      d_nonTrivialSphericalFnPerCell.end(),
                      0);

    dftfe::uInt accumNonTrivialSphericalFnCells = 0;
    for (dftfe::Int iElem = 0; iElem < numCells; ++iElem)
      {
        d_nonTrivialSphericalFnsCellStartIndex[iElem] =
          accumNonTrivialSphericalFnCells;
        accumNonTrivialSphericalFnCells +=
          d_nonTrivialSphericalFnPerCell[iElem];
      }
    if (!d_floatingNuclearCharges)
      {
        d_atomCenteredKpointIndexedSphericalFnQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints,
          ValueType(0));
        d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints * 3,
          ValueType(0));
      }

    std::vector<std::vector<dftfe::uInt>> sphericalFnKetTimesVectorLocalIds;
    sphericalFnKetTimesVectorLocalIds.clear();
    sphericalFnKetTimesVectorLocalIds.resize(d_totalAtomsInCurrentProc);
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


        for (dftfe::uInt alpha = 0; alpha < numSphericalFunctions; ++alpha)
          {
            dftfe::uInt globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;

            dftfe::uInt localId = d_SphericalFunctionKetTimesVectorPar[0]
                                    .get_partitioner()
                                    ->global_to_local(globalId);
            sphericalFnKetTimesVectorLocalIds[iAtom].push_back(localId);
          }
      }

    d_sphericalFnTimesVectorFlattenedVectorLocalIds.clear();
    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.clear();
    for (dftfe::uInt ielem = 0; ielem < numCells; ++ielem)
      {
        for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            bool isNonTrivial = false;
            for (dftfe::uInt i = 0;
                 i < d_cellIdToAtomIdsLocalCompactSupportMap[ielem].size();
                 i++)
              if (d_cellIdToAtomIdsLocalCompactSupportMap[ielem][i] == iAtom)
                {
                  isNonTrivial = true;
                  break;
                }
            if (isNonTrivial)
              {
                dftfe::uInt       atomId = atomIdsInProc[iAtom];
                const dftfe::uInt Znum   = atomicNumber[atomId];
                const dftfe::uInt numSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (dftfe::uInt iAlpha = 0; iAlpha < numSphericalFunctions;
                     ++iAlpha)
                  {
                    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.push_back(
                      ielem);
                    d_sphericalFnTimesVectorFlattenedVectorLocalIds.push_back(
                      sphericalFnKetTimesVectorLocalIds[iAtom][iAlpha]);
                  }
              }
          }
      }


#if defined(DFTFE_WITH_DEVICE)

    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
          .clear();
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
          .clear();

        std::vector<dftfe::uInt> atomIdsInCurrentProcess =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();

        d_sphericalFnIdsParallelNumberingMap.clear();
        d_sphericalFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);
        d_sphericalFnIdsPaddedParallelNumberingMap.clear();
        d_sphericalFnIdsPaddedParallelNumberingMap.resize(
          atomIdsInCurrentProcess.size() * d_maxSingleAtomContribution, -1);

        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution, -1);

        d_nonlocalElemIdToLocalElemIdMap.clear();
        d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);

        d_mapSphericalFnTimesVectorAllCellsReduction.clear();
        d_mapSphericalFnTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution,
          d_totalNonLocalEntries + 1);



        dftfe::uInt countElemNode    = 0;
        dftfe::uInt countElem        = 0;
        dftfe::uInt countAlpha       = 0;
        dftfe::uInt numShapeFnsAccum = 0;

        dftfe::Int totalElements = 0;
        d_mapiAtomTosphFuncWaveStart.resize(d_totalAtomsInCurrentProc);
        for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            const dftfe::uInt        atomId = atomIdsInCurrentProcess[iAtom];
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
              d_atomCenteredSphericalFunctionContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            dftfe::uInt totalAtomIdElementIterators =
              elementIndexesInAtomCompactSupport.size();
            totalElements += totalAtomIdElementIterators;
            const dftfe::uInt Znum = atomicNumber[atomId];
            const dftfe::uInt numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


            for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                dftfe::uInt globalId =
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    [std::make_pair(atomId, alpha)];

                const dftfe::uInt id = d_SphericalFunctionKetTimesVectorPar[0]
                                         .get_partitioner()
                                         ->global_to_local(globalId);

                if (alpha == 0)
                  {
                    d_mapiAtomTosphFuncWaveStart[iAtom] = countAlpha;
                  }
                d_sphericalFnIdsParallelNumberingMap[countAlpha] = id;
                d_sphericalFnIdsPaddedParallelNumberingMap
                  [iAtom * d_maxSingleAtomContribution + alpha] = id;
                for (dftfe::uInt iElemComp = 0;
                     iElemComp < totalAtomIdElementIterators;
                     iElemComp++)
                  {
                    d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                      [d_numberCellsAccumNonLocalAtoms[iAtom] *
                         d_maxSingleAtomContribution +
                       iElemComp * d_maxSingleAtomContribution + alpha] =
                        iAtom * d_maxSingleAtomContribution + alpha;
                  }
                countAlpha++;
              }


            for (dftfe::uInt iElemComp = 0;
                 iElemComp < totalAtomIdElementIterators;
                 ++iElemComp)
              {
                const dftfe::uInt elementId =
                  elementIndexesInAtomCompactSupport[iElemComp];
                d_nonlocalElemIdToLocalElemIdMap[countElem] = elementId;

                if (!d_useGlobalCMatrix)
                  {
                    for (dftfe::uInt ikpoint = 0;
                         ikpoint < d_kPointWeights.size();
                         ikpoint++)
                      for (dftfe::uInt iNode = 0;
                           iNode < d_numberNodesPerElement;
                           ++iNode)
                        {
                          for (dftfe::uInt alpha = 0;
                               alpha < numberSphericalFunctions;
                               ++alpha)
                            {
                              d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                                [ikpoint * d_totalNonlocalElems *
                                   d_numberNodesPerElement *
                                   d_maxSingleAtomContribution +
                                 countElem * d_maxSingleAtomContribution *
                                   d_numberNodesPerElement +
                                 d_numberNodesPerElement * alpha + iNode] =
                                  d_CMatrixEntriesConjugate
                                    [atomId][iElemComp]
                                    [ikpoint * d_numberNodesPerElement *
                                       numberSphericalFunctions +
                                     d_numberNodesPerElement * alpha + iNode];

                              d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
                                [ikpoint * d_totalNonlocalElems *
                                   d_numberNodesPerElement *
                                   d_maxSingleAtomContribution +
                                 countElem * d_numberNodesPerElement *
                                   d_maxSingleAtomContribution +
                                 d_maxSingleAtomContribution * iNode + alpha] =
                                  d_CMatrixEntriesTranspose
                                    [atomId][iElemComp]
                                    [ikpoint * d_numberNodesPerElement *
                                       numberSphericalFunctions +
                                     numberSphericalFunctions * iNode + alpha];
                            }
                        }
                  }

                for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                     ++alpha)
                  {
                    const dftfe::uInt index =
                      countElem * d_maxSingleAtomContribution + alpha;
                    d_mapSphericalFnTimesVectorAllCellsReduction[index] =
                      numShapeFnsAccum + alpha;
                  }
                countElem++;
              }

            numShapeFnsAccum += numberSphericalFunctions;
          }


        d_sphericalFnIdsParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.resize(
          d_sphericalFnIdsPaddedParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.resize(
          d_sphericalFnIdsParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsParallelNumberingMap);
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsPaddedParallelNumberingMap);
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.resize(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size());
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.copyFrom(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);

        d_mapSphericalFnTimesVectorAllCellsReductionDevice.clear();
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.resize(
          d_mapSphericalFnTimesVectorAllCellsReduction.size());
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.copyFrom(
          d_mapSphericalFnTimesVectorAllCellsReduction);

        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (dftfe::uInt i = 0; i < d_totalNonlocalElems; i++)
          {
            dftfe::uInt iCell = d_nonlocalElemIdToLocalElemIdMap[i];

            d_nonlocalElemIdToCellIdVector.push_back(iCell);
            for (dftfe::Int iNode = 0; iNode < d_numberNodesPerElement; iNode++)
              {
                dftfe::uInt localNodeId =
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    [iCell * d_numberNodesPerElement + iNode];
                d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                  .push_back(localNodeId);
              }
          }
      }

#endif

    d_totalLocallyOwnedNodes = basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();

    const ValueType alpha1 = 1.0;

    std::vector<dftfe::uInt> atomicNumbers =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();

    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    d_atomStartIndexGlobal.clear();
    d_atomStartIndexGlobal.resize(atomicNumbers.size(), 0);

    dftfe::uInt                                     counter = 0;
    std::map<dftfe::uInt, std::vector<dftfe::uInt>> listOfAtomIdsInSpecies;
    for (dftfe::uInt atomId = 0; atomId < atomicNumbers.size(); atomId++)
      {
        const dftfe::uInt Znum = atomicNumbers[atomId];
        d_setOfAtomicNumber.insert(Znum);
        d_atomStartIndexGlobal[atomId] = counter;
        dftfe::uInt numSphFunc =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        counter += numSphFunc;
      }
    std::map<dftfe::uInt, dftfe::uInt> mapSpeciesIdToAtomicNum;
    d_totalNumSphericalFunctionsGlobal = counter;

    for (dftfe::uInt iAtomicNum = 0; iAtomicNum < d_setOfAtomicNumber.size();
         iAtomicNum++)
      {
        dftfe::uInt Znum = *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
        listOfAtomIdsInSpecies[Znum].resize(0);
        d_listOfiAtomInSpecies[Znum].resize(0);
        mapSpeciesIdToAtomicNum[Znum] = iAtomicNum;
      }

    d_mapAtomIdToSpeciesIndex.resize(atomicNumbers.size());
    std::fill(d_mapAtomIdToSpeciesIndex.begin(),
              d_mapAtomIdToSpeciesIndex.end(),
              0);

    d_mapiAtomToSpeciesIndex.resize(atomIdsInCurrentProcess.size());
    std::fill(d_mapiAtomToSpeciesIndex.begin(),
              d_mapiAtomToSpeciesIndex.end(),
              0);

    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId                = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum                  = atomicNumbers[atomId];
        dftfe::uInt iAtomicNum            = mapSpeciesIdToAtomicNum[Znum];
        d_mapAtomIdToSpeciesIndex[atomId] = listOfAtomIdsInSpecies[Znum].size();
        d_mapiAtomToSpeciesIndex[iAtom]   = d_listOfiAtomInSpecies[Znum].size();
        listOfAtomIdsInSpecies[Znum].push_back(atomId);
        d_listOfiAtomInSpecies[Znum].push_back(iAtom);
      }

    d_CMatrixGlobal.resize(d_kPointWeights.size());

    d_dotProductAtomicWaveInputWaveTemp.resize(d_setOfAtomicNumber.size());
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); kPoint++)
      {
        d_CMatrixGlobal[kPoint].resize(d_setOfAtomicNumber.size());
        for (dftfe::uInt iAtomicNum = 0;
             iAtomicNum < d_setOfAtomicNumber.size();
             iAtomicNum++)
          {
            dftfe::uInt Znum =
              *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
            dftfe::uInt numSphFunc =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numAtomsPerSpecies =
              d_listOfiAtomInSpecies[Znum].size();
            d_CMatrixGlobal[kPoint][iAtomicNum].resize(
              numAtomsPerSpecies * numSphFunc * d_totalLocallyOwnedNodes);
            d_CMatrixGlobal[kPoint][iAtomicNum].setValue(0.0);
          }
      }

    const std::vector<
      std::vector<dftfe::utils::MemoryStorage<ValueTypeSrc, memorySpace>>>
      &globalCMatrixSrc = nonLocalOperatorSrc->getGlobalCMatrix();
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); kPoint++)
      {
        for (dftfe::uInt iAtomicNum = 0;
             iAtomicNum < d_setOfAtomicNumber.size();
             iAtomicNum++)
          {
            dftfe::uInt Znum =
              *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);

            dftfe::uInt numSphFunc =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numAtomsPerSpecies =
              d_listOfiAtomInSpecies[Znum].size();


            if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
              {
                for (dftfe::uInt iNode = 0;
                     iNode < globalCMatrixSrc[kPoint][iAtomicNum].size();
                     iNode++)
                  {
                    d_CMatrixGlobal[kPoint][iAtomicNum].data()[iNode] =
                      globalCMatrixSrc[kPoint][iAtomicNum].data()[iNode];
                  }
              }
#if defined(DFTFE_WITH_DEVICE)
            else
              {
                std::vector<ValueTypeSrc> CmatrixGlobalTempSrc(
                  d_totalLocallyOwnedNodes * numAtomsPerSpecies * numSphFunc);

                globalCMatrixSrc[kPoint][iAtomicNum].copyTo(
                  CmatrixGlobalTempSrc);

                std::vector<ValueType> CmatrixGlobalTempDst(
                  d_totalLocallyOwnedNodes * numAtomsPerSpecies * numSphFunc);

                for (dftfe::uInt iNode = 0;
                     iNode < globalCMatrixSrc[kPoint][iAtomicNum].size();
                     iNode++)
                  {
                    CmatrixGlobalTempDst[iNode] = CmatrixGlobalTempSrc[iNode];
                  }

                d_CMatrixGlobal[kPoint][iAtomicNum]
                  .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                    CmatrixGlobalTempDst.data(),
                    d_totalLocallyOwnedNodes * numAtomsPerSpecies * numSphFunc,
                    0,
                    0);
              }
#endif
          }
      }
    // deallocate the cell wise vectors
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesTranspose.clear();


#if defined(DFTFE_WITH_DEVICE)
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
      .clear();
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  template <typename ValueTypeSrc>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::copyCMatrixEntries(
    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueTypeSrc, memorySpace>>
      nonLocalOperatorSrc,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                      basisOperationsPtr,
    const dftfe::uInt quadratureIndex)
  {
    d_locallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const dftfe::uInt numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    d_numberNodesPerElement    = basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numCells = d_locallyOwnedCells;
    const dftfe::uInt numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomCenteredSphericalFunctionContainer->getAtomCoordinates();
    const std::map<dftfe::uInt, std::vector<double>> &periodicImageCoord =
      d_atomCenteredSphericalFunctionContainer
        ->getPeriodicImageCoordinatesList();
    const dftfe::uInt maxkPoints = d_kPointWeights.size();
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomCenteredSphericalFunctionContainer->getSphericalFunctions();


    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);
    d_atomCenteredKpointIndexedSphericalFnQuadValues.clear();
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.clear();
    d_cellIdToAtomIdsLocalCompactSupportMap.clear();
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    d_nonTrivialSphericalFnPerCell.clear();
    d_nonTrivialSphericalFnPerCell.resize(numCells, 0);

    d_nonTrivialSphericalFnsCellStartIndex.clear();
    d_nonTrivialSphericalFnsCellStartIndex.resize(numCells, 0);

    d_atomIdToNonTrivialSphericalFnCellStartIndex.clear();
    std::map<dftfe::uInt, std::vector<dftfe::uInt>>
                             globalAtomIdToNonTrivialSphericalFnsCellStartIndex;
    std::vector<dftfe::uInt> accumTemp(numCells, 0);
    // Loop over atoms to determine sizes of various vectors for forces
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        dftfe::uInt       atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom] =
          std::vector<dftfe::uInt>(numCells, 0);
        globalAtomIdToNonTrivialSphericalFnsCellStartIndex[atomId] =
          std::vector<dftfe::uInt>(numCells, 0);

        for (dftfe::Int iElemComp = 0;
             iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const dftfe::uInt elementId =
              elementIndexesInAtomCompactSupport[iElemComp];

            d_cellIdToAtomIdsLocalCompactSupportMap[elementId].push_back(iAtom);

            d_nonTrivialSphericalFnPerCell[elementId] += numSphericalFunctions;
            d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom][elementId] =
              accumTemp[elementId];
            globalAtomIdToNonTrivialSphericalFnsCellStartIndex
              [atomId][elementId] = accumTemp[elementId];
            accumTemp[elementId] += numSphericalFunctions;
          }
      }

    d_sumNonTrivialSphericalFnOverAllCells =
      std::accumulate(d_nonTrivialSphericalFnPerCell.begin(),
                      d_nonTrivialSphericalFnPerCell.end(),
                      0);

    dftfe::uInt accumNonTrivialSphericalFnCells = 0;
    for (dftfe::Int iElem = 0; iElem < numCells; ++iElem)
      {
        d_nonTrivialSphericalFnsCellStartIndex[iElem] =
          accumNonTrivialSphericalFnCells;
        accumNonTrivialSphericalFnCells +=
          d_nonTrivialSphericalFnPerCell[iElem];
      }
    if (!d_floatingNuclearCharges)
      {
        d_atomCenteredKpointIndexedSphericalFnQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints,
          ValueType(0));
        d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.resize(
          maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
            numberQuadraturePoints * 3,
          ValueType(0));
        // Assert Check
        const std::vector<ValueTypeSrc>
          atomCenteredKpointIndexedSphericalFnQuadValueSrc =
            nonLocalOperatorSrc
              ->getAtomCenteredKpointIndexedSphericalFnQuadValues();
        const std::vector<ValueTypeSrc>
          atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues =
            nonLocalOperatorSrc
              ->getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues();
        for (dftfe::uInt iTemp = 0;
             iTemp < atomCenteredKpointIndexedSphericalFnQuadValueSrc.size();
             iTemp++)
          {
            d_atomCenteredKpointIndexedSphericalFnQuadValues[iTemp] =
              atomCenteredKpointIndexedSphericalFnQuadValueSrc[iTemp];
            d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
              [3 * iTemp + 0] =
                atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                  [3 * iTemp + 0];
            d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
              [3 * iTemp + 1] =
                atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                  [3 * iTemp + 1];
            d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
              [3 * iTemp + 2] =
                atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                  [3 * iTemp + 2];
          }
      }

    std::vector<std::vector<dftfe::uInt>> sphericalFnKetTimesVectorLocalIds;
    sphericalFnKetTimesVectorLocalIds.clear();
    sphericalFnKetTimesVectorLocalIds.resize(d_totalAtomsInCurrentProc);
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        const dftfe::uInt Znum   = atomicNumber[atomId];
        const dftfe::uInt numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


        for (dftfe::uInt alpha = 0; alpha < numSphericalFunctions; ++alpha)
          {
            dftfe::uInt globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;

            dftfe::uInt localId = d_SphericalFunctionKetTimesVectorPar[0]
                                    .get_partitioner()
                                    ->global_to_local(globalId);
            sphericalFnKetTimesVectorLocalIds[iAtom].push_back(localId);
          }
      }

    d_sphericalFnTimesVectorFlattenedVectorLocalIds.clear();
    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.clear();
    for (dftfe::uInt ielem = 0; ielem < numCells; ++ielem)
      {
        for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            bool isNonTrivial = false;
            for (dftfe::uInt i = 0;
                 i < d_cellIdToAtomIdsLocalCompactSupportMap[ielem].size();
                 i++)
              if (d_cellIdToAtomIdsLocalCompactSupportMap[ielem][i] == iAtom)
                {
                  isNonTrivial = true;
                  break;
                }
            if (isNonTrivial)
              {
                dftfe::uInt       atomId = atomIdsInProc[iAtom];
                const dftfe::uInt Znum   = atomicNumber[atomId];
                const dftfe::uInt numSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (dftfe::uInt iAlpha = 0; iAlpha < numSphericalFunctions;
                     ++iAlpha)
                  {
                    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.push_back(
                      ielem);
                    d_sphericalFnTimesVectorFlattenedVectorLocalIds.push_back(
                      sphericalFnKetTimesVectorLocalIds[iAtom][iAlpha]);
                  }
              }
          }
      }
    for (dftfe::uInt iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        dftfe::uInt       ChargeId = atomIdsInProc[iAtom];
        dealii::Point<3>  nuclearCoordinates(atomCoordinates[3 * ChargeId + 0],
                                            atomCoordinates[3 * ChargeId + 1],
                                            atomCoordinates[3 * ChargeId + 2]);
        const dftfe::uInt atomId = ChargeId;
        const dftfe::uInt Znum   = atomicNumber[ChargeId];
        const dftfe::uInt NumRadialSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt NumTotalSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[ChargeId];
        const dftfe::uInt numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        if (numberElementsInAtomCompactSupport > 0)
          {
            d_CMatrixEntriesConjugate[ChargeId].resize(
              numberElementsInAtomCompactSupport);
            d_CMatrixEntriesTranspose[ChargeId].resize(
              numberElementsInAtomCompactSupport);
          }
        for (dftfe::Int iElemComp = 0;
             iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            d_CMatrixEntriesConjugate[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));
            d_CMatrixEntriesTranspose[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));
            const std::vector<ValueTypeSrc> CMatrixEntriesConjugateSrc =
              nonLocalOperatorSrc->getCmatrixEntriesConjugate(ChargeId,
                                                              iElemComp);
            const std::vector<ValueTypeSrc> CMatrixEntriesTransposeSrc =
              nonLocalOperatorSrc->getCmatrixEntriesTranspose(ChargeId,
                                                              iElemComp);
            for (dftfe::Int iTemp = 0;
                 iTemp < CMatrixEntriesConjugateSrc.size();
                 iTemp++)
              {
                d_CMatrixEntriesConjugate[ChargeId][iElemComp][iTemp] =
                  CMatrixEntriesConjugateSrc[iTemp];
                d_CMatrixEntriesTranspose[ChargeId][iElemComp][iTemp] =
                  CMatrixEntriesTransposeSrc[iTemp];
              }
          }

      } // iAtom


    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (dftfe::uInt iCell = 0; iCell < d_locallyOwnedCells; iCell++)
          {
            if (atomSupportInElement(iCell))
              {
                d_nonlocalElemIdToCellIdVector.push_back(iCell);
                for (dftfe::Int iNode = 0; iNode < d_numberNodesPerElement;
                     iNode++)
                  {
                    // dftfe::uInt localNodeId =
                    //   basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    //     [iCell * d_numberNodesPerElement + iNode];
                    // d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                    //   .push_back(localNodeId);
                  }
              }
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_elementIdToNonLocalElementIdMap.clear();
        d_elementIdToNonLocalElementIdMap.resize(d_locallyOwnedCells);
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
          .clear();
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
          .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                    d_numberNodesPerElement * d_maxSingleAtomContribution,
                  ValueType(0.0));
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
          .clear();
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
          .resize(d_kPointWeights.size() * d_totalNonlocalElems *
                    d_numberNodesPerElement * d_maxSingleAtomContribution,
                  ValueType(0.0));
        std::vector<dftfe::uInt> atomIdsInCurrentProcess =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<dftfe::uInt> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();

        d_sphericalFnIdsParallelNumberingMap.clear();
        d_sphericalFnIdsParallelNumberingMap.resize(d_totalNonLocalEntries, 0);
        d_sphericalFnIdsPaddedParallelNumberingMap.clear();
        d_sphericalFnIdsPaddedParallelNumberingMap.resize(
          atomIdsInCurrentProcess.size() * d_maxSingleAtomContribution, -1);

        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution, -1);

        d_nonlocalElemIdToLocalElemIdMap.clear();
        d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);

        d_mapSphericalFnTimesVectorAllCellsReduction.clear();
        d_mapSphericalFnTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution,
          d_totalNonLocalEntries + 1);



        dftfe::uInt countElemNode    = 0;
        dftfe::uInt countElem        = 0;
        dftfe::uInt countAlpha       = 0;
        dftfe::uInt numShapeFnsAccum = 0;

        dftfe::Int totalElements = 0;
        d_mapiAtomTosphFuncWaveStart.resize(d_totalAtomsInCurrentProc);

        std::map<dftfe::uInt, dftfe::uInt> atomIdToNumShapeFnsAccumulated;
        std::map<dftfe::uInt, dftfe::uInt> atomIdToMaxShapeFnsAccumulated;
        for (dftfe::Int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            const dftfe::uInt        atomId = atomIdsInCurrentProcess[iAtom];
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
              d_atomCenteredSphericalFunctionContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            dftfe::uInt totalAtomIdElementIterators =
              elementIndexesInAtomCompactSupport.size();
            totalElements += totalAtomIdElementIterators;
            const dftfe::uInt Znum = atomicNumber[atomId];
            const dftfe::uInt numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


            for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                dftfe::uInt globalId =
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    [std::make_pair(atomId, alpha)];

                const dftfe::uInt id = d_SphericalFunctionKetTimesVectorPar[0]
                                         .get_partitioner()
                                         ->global_to_local(globalId);

                if (alpha == 0)
                  {
                    d_mapiAtomTosphFuncWaveStart[iAtom] = countAlpha;
                  }
                d_sphericalFnIdsParallelNumberingMap[countAlpha] = id;
                d_sphericalFnIdsPaddedParallelNumberingMap
                  [iAtom * d_maxSingleAtomContribution + alpha] = id;

                countAlpha++;
              }

            atomIdToNumShapeFnsAccumulated[atomId] = numShapeFnsAccum;
            atomIdToMaxShapeFnsAccumulated[atomId] =
              iAtom * d_maxSingleAtomContribution;
            numShapeFnsAccum += numberSphericalFunctions;
          }

        const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
        for (dftfe::uInt iElem = 0; iElem < d_locallyOwnedCells; iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<dftfe::Int> &atomIdsInCell =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                for (dftfe::uInt iAtom = 0; iAtom < atomIdsInCell.size();
                     iAtom++)
                  {
                    dftfe::uInt atomId = atomIdsInCell[iAtom];
                    dftfe::uInt Znum   = atomicNumber[atomId];
                    dftfe::uInt numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const dftfe::Int nonZeroElementMatrixId =
                      sparsityPattern.find(atomId)->second[iElem];
                    if (!d_useGlobalCMatrix)
                      {
                        for (dftfe::uInt ikpoint = 0;
                             ikpoint < d_kPointWeights.size();
                             ikpoint++)
                          for (dftfe::uInt iNode = 0;
                               iNode < d_numberNodesPerElement;
                               ++iNode)
                            {
                              for (dftfe::uInt alpha = 0;
                                   alpha < numberSphericalFunctions;
                                   ++alpha)
                                {
                                  d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
                                    [ikpoint * d_totalNonlocalElems *
                                       d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     countElem * d_maxSingleAtomContribution *
                                       d_numberNodesPerElement +
                                     d_numberNodesPerElement * alpha +
                                     iNode] = d_CMatrixEntriesConjugate
                                      [atomId][nonZeroElementMatrixId]
                                      [ikpoint * d_numberNodesPerElement *
                                         numberSphericalFunctions +
                                       d_numberNodesPerElement * alpha + iNode];

                                  d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
                                    [ikpoint * d_totalNonlocalElems *
                                       d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     countElem * d_numberNodesPerElement *
                                       d_maxSingleAtomContribution +
                                     d_maxSingleAtomContribution * iNode +
                                     alpha] = d_CMatrixEntriesTranspose
                                      [atomId][nonZeroElementMatrixId]
                                      [ikpoint * d_numberNodesPerElement *
                                         numberSphericalFunctions +
                                       numberSphericalFunctions * iNode +
                                       alpha];
                                }
                            }
                      }
                    d_nonlocalElemIdToLocalElemIdMap[countElem] = iElem;
                    for (dftfe::uInt alpha = 0;
                         alpha < numberSphericalFunctions;
                         ++alpha)
                      {
                        const dftfe::uInt index =
                          countElem * d_maxSingleAtomContribution + alpha;
                        d_mapSphericalFnTimesVectorAllCellsReduction[index] =
                          atomIdToNumShapeFnsAccumulated[atomId] + alpha;
                        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                          [countElem * d_maxSingleAtomContribution + alpha] =
                            atomIdToMaxShapeFnsAccumulated[atomId] + alpha;
                      }
                    d_elementIdToNonLocalElementIdMap[iElem].push_back(
                      std::make_pair(atomId, countElem));
                    countElem++;
                  }
              }
          }

        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
          .resize(
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
              .size() /
            d_kPointWeights.size());
        d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
          .resize(
            d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
              .size() /
            d_kPointWeights.size());



        d_sphericalFnIdsParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.clear();
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.resize(
          d_sphericalFnIdsPaddedParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.resize(
          d_sphericalFnIdsParallelNumberingMap.size());
        d_sphericalFnIdsParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsParallelNumberingMap);
        d_sphericalFnIdsPaddedParallelNumberingMapDevice.copyFrom(
          d_sphericalFnIdsPaddedParallelNumberingMap);
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.resize(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.size());
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.copyFrom(
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec);

        d_mapSphericalFnTimesVectorAllCellsReductionDevice.clear();
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.resize(
          d_mapSphericalFnTimesVectorAllCellsReduction.size());
        d_mapSphericalFnTimesVectorAllCellsReductionDevice.copyFrom(
          d_mapSphericalFnTimesVectorAllCellsReduction);

        d_nonlocalElemIdToCellIdVector.clear();



        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (dftfe::uInt i = 0; i < d_totalNonlocalElems; i++)
          {
            dftfe::uInt iCell = d_nonlocalElemIdToLocalElemIdMap[i];

            d_nonlocalElemIdToCellIdVector.push_back(iCell);
            for (dftfe::Int iNode = 0; iNode < d_numberNodesPerElement; iNode++)
              {
                dftfe::uInt localNodeId =
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    [iCell * d_numberNodesPerElement + iNode];
                d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                  .push_back(localNodeId);
              }
          }
      }
#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::isGlobalCMatrix()
    const
  {
    return d_useGlobalCMatrix;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonlocalElementToCellIdVector() const
  {
    return (d_nonlocalElemIdToCellIdVector);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getFlattenedNonLocalCellDofIndexToProcessDofIndexMap() const
  {
    return (d_flattenedNonLocalCellDofIndexToProcessDofIndexMap);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::paddingCouplingMatrix(
    const std::vector<ValueType> &entries,
    std::vector<ValueType>       &entriesPadded,
    const CouplingStructure       couplingtype)
  {
    entriesPadded.clear();
    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<dftfe::uInt> atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    if (couplingtype == CouplingStructure::diagonal)
      {
        entriesPadded.resize(atomIdsInProcessor.size() *
                             d_maxSingleAtomContribution);
        dftfe::uInt index = 0;
        for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const dftfe::uInt atomId = atomIdsInProcessor[iAtom];
            const dftfe::uInt Znum   = atomicNumber[atomId];
            const dftfe::uInt numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            for (dftfe::uInt alpha = 0; alpha < numberOfTotalSphericalFunctions;
                 alpha++)
              {
                entriesPadded[iAtom * d_maxSingleAtomContribution + alpha] =
                  entries[index];
                index++;
              }
          }
      }
    else if (couplingtype == CouplingStructure::dense)
      {
        entriesPadded.resize(atomIdsInProcessor.size() *
                             d_maxSingleAtomContribution *
                             d_maxSingleAtomContribution);
        dftfe::uInt index = 0;
        for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const dftfe::uInt atomId = atomIdsInProcessor[iAtom];
            const dftfe::uInt Znum   = atomicNumber[atomId];
            const dftfe::uInt numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

            for (dftfe::Int alpha_i = 0;
                 alpha_i < numberOfTotalSphericalFunctions;
                 alpha_i++)
              {
                for (dftfe::Int alpha_j = 0;
                     alpha_j < numberOfTotalSphericalFunctions;
                     alpha_j++)
                  {
                    entriesPadded[iAtom * d_maxSingleAtomContribution *
                                    d_maxSingleAtomContribution +
                                  alpha_i * d_maxSingleAtomContribution +
                                  alpha_j] = entries[index];
                    index++;
                  }
              }
          }
      }
    else if (couplingtype == CouplingStructure::blockDiagonal)
      {
        entriesPadded.resize(atomIdsInProcessor.size() *
                             d_maxSingleAtomContribution *
                             d_maxSingleAtomContribution * 4);
        dftfe::uInt index = 0;
        for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const dftfe::uInt atomId = atomIdsInProcessor[iAtom];
            const dftfe::uInt Znum   = atomicNumber[atomId];
            const dftfe::uInt numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

            for (dftfe::Int alpha_i = 0;
                 alpha_i < 2 * numberOfTotalSphericalFunctions;
                 alpha_i++)
              {
                for (dftfe::Int alpha_j = 0;
                     alpha_j < 2 * numberOfTotalSphericalFunctions;
                     alpha_j++)
                  {
                    entriesPadded[iAtom * d_maxSingleAtomContribution *
                                    d_maxSingleAtomContribution * 4 +
                                  alpha_i * 2 * d_maxSingleAtomContribution +
                                  alpha_j] = entries[index];
                    index++;
                  }
              }
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<
    std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>>> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::getGlobalCMatrix()
    const
  {
    return d_CMatrixGlobal;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    computeGlobalCMatrixVector(
      std::shared_ptr<dftfe::basis::FEBasisOperations<
        dataTypes::number,
        double,
        dftfe::utils::MemorySpace::HOST>> basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperHostPtr)
  {
    d_totalLocallyOwnedNodes = basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt numberNodesPerElement =
      basisOperationsPtr->nDofsPerCell();
    const ValueType          alpha1 = 1.0;
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<dftfe::uInt> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    d_atomStartIndexGlobal.clear();
    d_atomStartIndexGlobal.resize(atomicNumbers.size(), 0);
    dftfe::uInt                                     counter = 0;
    std::map<dftfe::uInt, std::vector<dftfe::uInt>> listOfAtomIdsInSpecies;
    for (dftfe::uInt atomId = 0; atomId < atomicNumbers.size(); atomId++)
      {
        const dftfe::uInt Znum = atomicNumbers[atomId];
        d_setOfAtomicNumber.insert(Znum);
        d_atomStartIndexGlobal[atomId] = counter;
        dftfe::uInt numSphFunc =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        counter += numSphFunc;
      }
    std::map<dftfe::uInt, dftfe::uInt> mapSpeciesIdToAtomicNum;
    d_totalNumSphericalFunctionsGlobal = counter;

    for (dftfe::uInt iAtomicNum = 0; iAtomicNum < d_setOfAtomicNumber.size();
         iAtomicNum++)
      {
        dftfe::uInt Znum = *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
        listOfAtomIdsInSpecies[Znum].resize(0);
        d_listOfiAtomInSpecies[Znum].resize(0);
        mapSpeciesIdToAtomicNum[Znum] = iAtomicNum;
      }

    d_mapAtomIdToSpeciesIndex.resize(atomicNumbers.size());
    std::fill(d_mapAtomIdToSpeciesIndex.begin(),
              d_mapAtomIdToSpeciesIndex.end(),
              0);

    d_mapiAtomToSpeciesIndex.resize(atomIdsInCurrentProcess.size());
    std::fill(d_mapiAtomToSpeciesIndex.begin(),
              d_mapiAtomToSpeciesIndex.end(),
              0);

    for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        dftfe::uInt atomId                = atomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum                  = atomicNumbers[atomId];
        dftfe::uInt iAtomicNum            = mapSpeciesIdToAtomicNum[Znum];
        d_mapAtomIdToSpeciesIndex[atomId] = listOfAtomIdsInSpecies[Znum].size();
        d_mapiAtomToSpeciesIndex[iAtom]   = d_listOfiAtomInSpecies[Znum].size();
        listOfAtomIdsInSpecies[Znum].push_back(atomId);
        d_listOfiAtomInSpecies[Znum].push_back(iAtom);
      }

    d_CMatrixGlobal.resize(d_kPointWeights.size());

    dftfe::linearAlgebra::MultiVector<ValueType,
                                      dftfe::utils::MemorySpace::HOST>
      Pmatrix;
    Pmatrix.reinit(basisOperationsPtr->mpiPatternP2P,
                   d_totalNumSphericalFunctionsGlobal);

    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); kPoint++)
      {
        d_CMatrixGlobal[kPoint].resize(d_setOfAtomicNumber.size());
        for (dftfe::uInt iAtomicNum = 0;
             iAtomicNum < d_setOfAtomicNumber.size();
             iAtomicNum++)
          {
            dftfe::uInt Znum =
              *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);
            dftfe::uInt numSphFunc =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numAtomsPerSpecies =
              d_listOfiAtomInSpecies[Znum].size();
            d_CMatrixGlobal[kPoint][iAtomicNum].resize(
              numAtomsPerSpecies * numSphFunc * d_totalLocallyOwnedNodes);
            d_CMatrixGlobal[kPoint][iAtomicNum].setValue(0.0);
          }
      }

    d_dotProductAtomicWaveInputWaveTemp.resize(d_setOfAtomicNumber.size());

    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); kPoint++)
      {
        Pmatrix.setValue(0);
        for (dftfe::Int iAtom = 0; iAtom < atomIdsInCurrentProcess.size();
             iAtom++)
          {
            dftfe::uInt atomId     = atomIdsInCurrentProcess[iAtom];
            dftfe::uInt startIndex = d_atomStartIndexGlobal[atomId];
            dftfe::uInt Znum       = atomicNumbers[atomId];
            dftfe::uInt numberOfSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            std::vector<dftfe::uInt> elementIndexesInAtomCompactSupport =
              d_atomCenteredSphericalFunctionContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            dftfe::Int numberElementsInAtomCompactSupport =
              elementIndexesInAtomCompactSupport.size();
            for (dftfe::Int iElem = 0;
                 iElem < numberElementsInAtomCompactSupport;
                 iElem++)
              {
                dftfe::uInt elementIndex =
                  elementIndexesInAtomCompactSupport[iElem];
                std::vector<ValueType> CMatrixEntries =
                  getCmatrixEntries(kPoint, atomId, elementIndex);
                AssertThrow(
                  CMatrixEntries.size() ==
                    numberOfSphericalFunctions * numberNodesPerElement,
                  dealii::ExcMessage(
                    "NonLocal Opertor::Initialization No. of  projectors mismatch in CmatrixEntries. Check input data "));
                for (dftfe::Int iDof = 0; iDof < numberNodesPerElement; iDof++)
                  {
                    long int dofIndex =
                      basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                        [elementIndex * numberNodesPerElement + iDof];
                    BLASWrapperHostPtr->xaxpy(
                      numberOfSphericalFunctions,
                      &alpha1,
                      &CMatrixEntries[iDof * numberOfSphericalFunctions],
                      1,
                      Pmatrix.data() +
                        (dofIndex * d_totalNumSphericalFunctionsGlobal +
                         startIndex),
                      1);
                  } // iDof


              } // iElem
          }     // iAtom
        basisOperationsPtr->d_constraintInfo[basisOperationsPtr->d_dofHandlerID]
          .distribute_slave_to_master(Pmatrix);
        Pmatrix.accumulateAddLocallyOwned();
        Pmatrix.zeroOutGhosts();

        for (dftfe::uInt iAtomicNum = 0;
             iAtomicNum < d_setOfAtomicNumber.size();
             iAtomicNum++)
          {
            dftfe::uInt Znum =
              *std::next(d_setOfAtomicNumber.begin(), iAtomicNum);

            dftfe::uInt numSphFunc =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            dftfe::uInt numAtomsPerSpecies =
              d_listOfiAtomInSpecies[Znum].size();

            std::vector<ValueType> CmatrixGlobalTemp(
              d_totalLocallyOwnedNodes * numAtomsPerSpecies * numSphFunc);

            for (dftfe::uInt iNode = 0; iNode < d_totalLocallyOwnedNodes;
                 iNode++)
              {
                for (dftfe::uInt atomIndex = 0;
                     atomIndex < d_listOfiAtomInSpecies[Znum].size();
                     atomIndex++)
                  {
                    dftfe::uInt iAtom = d_listOfiAtomInSpecies[Znum][atomIndex];
                    dftfe::uInt atomId     = atomIdsInCurrentProcess[iAtom];
                    dftfe::uInt startIndex = d_atomStartIndexGlobal[atomId];

                    BLASWrapperHostPtr->xcopy(
                      numSphFunc,
                      Pmatrix.data() +
                        (iNode * d_totalNumSphericalFunctionsGlobal +
                         startIndex),
                      1,
                      &CmatrixGlobalTemp[iNode * numAtomsPerSpecies *
                                           numSphFunc +
                                         atomIndex * numSphFunc],
                      1);
                  }
              }
            d_CMatrixGlobal[kPoint][iAtomicNum]
              .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                CmatrixGlobalTemp.data(),
                d_totalLocallyOwnedNodes * numAtomsPerSpecies * numSphFunc,
                0,
                0);
          }
      }


    // deallocate the cell wise vectors
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesTranspose.clear();


#if defined(DFTFE_WITH_DEVICE)
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugate
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTranspose
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionConjugateDevice
      .clear();
    d_IntegralFEMShapeFunctionValueTimesAtomicSphericalFunctionTransposeDevice
      .clear();
#endif
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::vector<ValueType>
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::getCmatrixEntries(
    dftfe::Int  kPointIndex,
    dftfe::uInt atomId,
    dftfe::Int  iElem) const
  {
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> sparsityPattern =
      d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    const dftfe::Int nonZeroElementMatrixId =
      sparsityPattern.find(atomId)->second[iElem];
    const dftfe::uInt numberSphericalFunctions =
      d_atomCenteredSphericalFunctionContainer
        ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[atomId]);
    std::vector<ValueType> Ctemp(d_numberNodesPerElement *
                                   numberSphericalFunctions,
                                 0.0);

    for (dftfe::Int i = 0; i < Ctemp.size(); i++)
      {
        Ctemp[i] =
          d_CMatrixEntriesTranspose[atomId][nonZeroElementMatrixId]
                                   [kPointIndex * d_numberNodesPerElement *
                                      numberSphericalFunctions +
                                    i];
      }

    return Ctemp;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getOwnedAtomIdsInCurrentProcessor() const
  {
    return d_OwnedAtomIdsInCurrentProcessor;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::uInt> &
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::getAtomIdsInCurrentProcessor()
    const
  {
    return d_atomCenteredSphericalFunctionContainer
      ->getAtomIdsInCurrentProcess();
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    atomPresentInCellRange(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange) const
  {
    bool flag = false;
    for (dftfe::uInt iElem = cellRange.first; iElem < cellRange.second; iElem++)
      {
        flag =
          d_atomCenteredSphericalFunctionContainer->atomSupportInElement(iElem);
        if (flag == true)
          return true;
      }
    return flag;
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCmatrixEntriesConjugate(const dftfe::uInt chargeId,
                               const dftfe::uInt iElemComp) const
  {
    return (d_CMatrixEntriesConjugate[chargeId][iElemComp]);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<ValueType> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCmatrixEntriesTranspose(const dftfe::uInt chargeId,
                               const dftfe::uInt iElemComp) const
  {
    return (d_CMatrixEntriesTranspose[chargeId][iElemComp]);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNumberOfSphericalFunctionsForAtomId(dftfe::uInt atomId)
  {
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    return (
      d_atomCenteredSphericalFunctionContainer
        ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumbers[atomId]));
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    computeInnerProductOverSphericalFnsWaveFns(
      const dftfe::Int vectorDimension,
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &VCconjTransXsphericalFunctionKetTimesVectorParFlattened,
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      const std::map<dftfe::uInt, dftfe::uInt> nonlocalAtomIdToGlobalIdMap,
      std::vector<ValueType>                  &outputVector)
  {
    const std::vector<dftfe::uInt> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const dftfe::uInt numberOfAtoms = atomicNumber.size();
    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempVec(vectorDimension,
                                                                0.0);
    for (dftfe::Int iAtom = 0; iAtom < d_OwnedAtomIdsInCurrentProcessor.size();
         iAtom++)
      {
        dftfe::uInt atomId   = d_OwnedAtomIdsInCurrentProcessor[iAtom];
        dftfe::uInt globalId = nonlocalAtomIdToGlobalIdMap.find(atomId)->second;
        tempVec.copyFrom(outputVector,
                         vectorDimension,
                         vectorDimension == 9 ? 0 : globalId * vectorDimension,
                         0);
        dftfe::uInt Znum = atomicNumber[atomId];
        dftfe::uInt numberOfSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        const dftfe::uInt id =
          d_SphericalFunctionKetTimesVectorPar[0]
            .get_partitioner()
            ->global_to_local(
              d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, 0)]);
        const ValueType scalarCoeffOne = ValueType(1.0);
        d_BLASWrapperPtr->xgemm(
          'N',
          'C',
          1,
          vectorDimension,
          d_numberWaveFunctions * numberOfSphericalFunctions,
          &scalarCoeffOne,
          VCconjTransXsphericalFunctionKetTimesVectorParFlattened.begin() +
            id * d_numberWaveFunctions,
          1,
          sphericalFunctionKetTimesVectorParFlattened.begin() +
            id * vectorDimension * d_numberWaveFunctions,
          vectorDimension,
          &scalarCoeffOne,
          tempVec.data(),
          1);
        tempVec.copyTo(outputVector,
                       vectorDimension,
                       0,
                       vectorDimension == 9 ? 0 : globalId * vectorDimension);
      }
  }

  template class AtomicCenteredNonLocalOperator<
    dataTypes::number,
    dftfe::utils::MemorySpace::HOST>;
  template class AtomicCenteredNonLocalOperator<
    dataTypes::numberFP32,
    dftfe::utils::MemorySpace::HOST>;

  template void
  AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                 dftfe::utils::MemorySpace::HOST>::
    copyCMatrixEntries(const std::shared_ptr<AtomicCenteredNonLocalOperator<
                         dataTypes::number,
                         dftfe::utils::MemorySpace::HOST>> nonLocalOperatorSrc,
                       std::shared_ptr<dftfe::basis::FEBasisOperations<
                         dataTypes::number,
                         double,
                         dftfe::utils::MemorySpace::HOST>> basisOperationsPtr,
                       const dftfe::uInt                   quadratureIndex);
  template void
  AtomicCenteredNonLocalOperator<dataTypes::number,
                                 dftfe::utils::MemorySpace::HOST>::
    copyCMatrixEntries(const std::shared_ptr<AtomicCenteredNonLocalOperator<
                         dataTypes::numberFP32,
                         dftfe::utils::MemorySpace::HOST>> nonLocalOperatorSrc,
                       std::shared_ptr<dftfe::basis::FEBasisOperations<
                         dataTypes::number,
                         double,
                         dftfe::utils::MemorySpace::HOST>> basisOperationsPtr,
                       const dftfe::uInt                   quadratureIndex);

  template void
  AtomicCenteredNonLocalOperator<dataTypes::number,
                                 dftfe::utils::MemorySpace::HOST>::
    copyPartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex,
      const std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                       dftfe::utils::MemorySpace::HOST>>
        nonLocalOperatorSrc);
  template void
  AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                 dftfe::utils::MemorySpace::HOST>::
    copyPartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex,
      const std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::number,
                                       dftfe::utils::MemorySpace::HOST>>
        nonLocalOperatorSrc);
#if defined(DFTFE_WITH_DEVICE)
  template class AtomicCenteredNonLocalOperator<
    dataTypes::number,
    dftfe::utils::MemorySpace::DEVICE>;
  template class AtomicCenteredNonLocalOperator<
    dataTypes::numberFP32,
    dftfe::utils::MemorySpace::DEVICE>;
  template void
  AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                 dftfe::utils::MemorySpace::DEVICE>::
    copyCMatrixEntries(
      const std::shared_ptr<AtomicCenteredNonLocalOperator<
        dataTypes::number,
        dftfe::utils::MemorySpace::DEVICE>> nonLocalOperatorSrc,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
                        basisOperationsPtr,
      const dftfe::uInt quadratureIndex);
  template void
  AtomicCenteredNonLocalOperator<dataTypes::number,
                                 dftfe::utils::MemorySpace::DEVICE>::
    copyCMatrixEntries(
      const std::shared_ptr<AtomicCenteredNonLocalOperator<
        dataTypes::numberFP32,
        dftfe::utils::MemorySpace::DEVICE>> nonLocalOperatorSrc,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
                        basisOperationsPtr,
      const dftfe::uInt quadratureIndex);

  template void
  AtomicCenteredNonLocalOperator<dataTypes::number,
                                 dftfe::utils::MemorySpace::DEVICE>::
    copyPartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex,
      const std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                       dftfe::utils::MemorySpace::DEVICE>>
        nonLocalOperatorSrc);
  template void
  AtomicCenteredNonLocalOperator<dataTypes::numberFP32,
                                 dftfe::utils::MemorySpace::DEVICE>::
    copyPartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                        BLASWrapperHostPtr,
      const dftfe::uInt quadratureIndex,
      const std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::number,
                                       dftfe::utils::MemorySpace::DEVICE>>
        nonLocalOperatorSrc);
#endif

} // namespace dftfe
