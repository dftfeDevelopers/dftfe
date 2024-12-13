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
// @author Kartick Ramakrishnan, Sambit Das, Phani Motamarri, Vishal Subramanian
//
#include <AtomicCenteredNonLocalOperator.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <AtomicCenteredNonLocalOperatorKernelsDevice.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
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
      const bool      memOptMode)
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
    d_memoryOptMode = memOptMode;
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
    d_atomCenteredSphericalFunctionContainer
      ->getTotalAtomsAndNonLocalElementsInCurrentProcessor(
        d_totalAtomsInCurrentProc,
        d_totalNonlocalElems,
        d_numberCellsForEachAtom,
        d_numberCellsAccumNonLocalAtoms);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseOperatorActionOnX(unsigned int kPointIndex)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_kPointIndex = kPointIndex;

        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<unsigned int> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            unsigned int atomId = atomIdsInProcessor[iAtom];

            d_sphericalFnTimesWavefunMatrix[atomId].setValue(0.0);
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_kPointIndex = kPointIndex;

        if (!d_memoryOptMode)
          {
            for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
              {
                hostPointerCDagger[i] =
                  d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice
                    .begin() +
                  d_kPointIndex * d_totalNonlocalElems *
                    d_numberNodesPerElement * d_maxSingleAtomContribution +
                  i * d_numberNodesPerElement * d_maxSingleAtomContribution;
              }

            dftfe::utils::deviceMemcpyH2D(devicePointerCDagger,
                                          hostPointerCDagger,
                                          d_totalNonlocalElems *
                                            sizeof(ValueType *));
          }
        else
          {
            d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.copyFrom(
              d_cellHamiltonianMatrixNonLocalFlattenedConjugate,
              d_totalNonlocalElems * d_numberNodesPerElement *
                d_maxSingleAtomContribution,
              d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
                d_maxSingleAtomContribution,
              0);
            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.copyFrom(
              d_cellHamiltonianMatrixNonLocalFlattenedTranspose,
              d_totalNonlocalElems * d_numberNodesPerElement *
                d_maxSingleAtomContribution,
              d_kPointIndex * d_totalNonlocalElems * d_numberNodesPerElement *
                d_maxSingleAtomContribution,
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
    const unsigned int                  quadratureIndex)
  {
    d_locallyOwnedCells = basisOperationsPtr->nCells();
    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const unsigned int numberAtomsOfInterest =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const unsigned int numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();
    d_numberNodesPerElement     = basisOperationsPtr->nDofsPerCell();
    const unsigned int numCells = d_locallyOwnedCells;
    const dftfe::utils::MemoryStorage<double, // ValueType for complex
                                      dftfe::utils::MemorySpace::HOST>
      &shapeValQuads =
        basisOperationsPtr
          ->shapeFunctionBasisData(); // shapeFunctionData() for complex
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = basisOperationsPtr->quadPoints();
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST>
                                     JxwVector = basisOperationsPtr->JxW();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::vector<double> &atomCoordinates =
      d_atomCenteredSphericalFunctionContainer->getAtomCoordinates();
    const std::map<unsigned int, std::vector<double>> &periodicImageCoord =
      d_atomCenteredSphericalFunctionContainer
        ->getPeriodicImageCoordinatesList();
    const unsigned int maxkPoints = d_kPointWeights.size();
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      sphericalFunction =
        d_atomCenteredSphericalFunctionContainer->getSphericalFunctions();

    // std::vector<ValueType> sphericalFunctionBasis(maxkPoints *
    //                                                  numberQuadraturePoints,
    //                                                0.0);
    d_CMatrixEntriesConjugate.clear();
    d_CMatrixEntriesConjugate.resize(numberAtomsOfInterest);
    d_CMatrixEntriesTranspose.clear();
    d_CMatrixEntriesTranspose.resize(numberAtomsOfInterest);
    d_atomCenteredKpointIndexedSphericalFnQuadValues.clear();
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.clear();
    d_cellIdToAtomIdsLocalCompactSupportMap.clear();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();

    d_nonTrivialSphericalFnPerCell.clear();
    d_nonTrivialSphericalFnPerCell.resize(numCells, 0);

    d_nonTrivialSphericalFnsCellStartIndex.clear();
    d_nonTrivialSphericalFnsCellStartIndex.resize(numCells, 0);

    d_atomIdToNonTrivialSphericalFnCellStartIndex.clear();
    std::map<unsigned int, std::vector<unsigned int>>
                              globalAtomIdToNonTrivialSphericalFnsCellStartIndex;
    std::vector<unsigned int> accumTemp(numCells, 0);
    // Loop over atoms to determine sizes of various vectors for forces
    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        unsigned int       atomId = atomIdsInProc[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[atomId];
        const unsigned int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();
        d_atomIdToNonTrivialSphericalFnCellStartIndex[iAtom] =
          std::vector<unsigned int>(numCells, 0);
        globalAtomIdToNonTrivialSphericalFnsCellStartIndex[atomId] =
          std::vector<unsigned int>(numCells, 0);
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const unsigned int elementId =
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

    unsigned int accumNonTrivialSphericalFnCells = 0;
    for (int iElem = 0; iElem < numCells; ++iElem)
      {
        d_nonTrivialSphericalFnsCellStartIndex[iElem] =
          accumNonTrivialSphericalFnCells;
        accumNonTrivialSphericalFnCells +=
          d_nonTrivialSphericalFnPerCell[iElem];
      }
    d_atomCenteredKpointIndexedSphericalFnQuadValues.resize(
      maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
        numberQuadraturePoints,
      ValueType(0));
    d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues.resize(
      maxkPoints * d_sumNonTrivialSphericalFnOverAllCells *
        numberQuadraturePoints * 3,
      ValueType(0));

    std::vector<std::vector<unsigned int>> sphericalFnKetTimesVectorLocalIds;
    sphericalFnKetTimesVectorLocalIds.clear();
    sphericalFnKetTimesVectorLocalIds.resize(d_totalAtomsInCurrentProc);
    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int numSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


        for (unsigned int alpha = 0; alpha < numSphericalFunctions; ++alpha)
          {
            unsigned int globalId =
              d_sphericalFunctionIdsNumberingMapCurrentProcess
                .find(std::make_pair(atomId, alpha))
                ->second;

            unsigned int localId = d_SphericalFunctionKetTimesVectorPar[0]
                                     .get_partitioner()
                                     ->global_to_local(globalId);
            sphericalFnKetTimesVectorLocalIds[iAtom].push_back(localId);
          }
      }

    d_sphericalFnTimesVectorFlattenedVectorLocalIds.clear();
    d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap.clear();
    for (unsigned int ielem = 0; ielem < numCells; ++ielem)
      {
        for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            bool isNonTrivial = false;
            for (unsigned int i = 0;
                 i < d_cellIdToAtomIdsLocalCompactSupportMap[ielem].size();
                 i++)
              if (d_cellIdToAtomIdsLocalCompactSupportMap[ielem][i] == iAtom)
                {
                  isNonTrivial = true;
                  break;
                }
            if (isNonTrivial)
              {
                unsigned int       atomId = atomIdsInProc[iAtom];
                const unsigned int Znum   = atomicNumber[atomId];
                const unsigned int numSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (unsigned int iAlpha = 0; iAlpha < numSphericalFunctions;
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


    for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
      {
        unsigned int       ChargeId = atomIdsInProc[iAtom];
        dealii::Point<3>   nuclearCoordinates(atomCoordinates[3 * ChargeId + 0],
                                            atomCoordinates[3 * ChargeId + 1],
                                            atomCoordinates[3 * ChargeId + 2]);
        const unsigned int atomId = ChargeId;
        std::vector<double> imageCoordinates =
          periodicImageCoord.find(atomId)->second;
        const unsigned int Znum = atomicNumber[ChargeId];
        const unsigned int NumRadialSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        const unsigned int NumTotalSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        std::vector<unsigned int> elementIndexesInAtomCompactSupport =
          d_atomCenteredSphericalFunctionContainer
            ->d_elementIndexesInAtomCompactSupport[ChargeId];
        const unsigned int numberElementsInAtomCompactSupport =
          elementIndexesInAtomCompactSupport.size();

        unsigned int imageIdsSize = imageCoordinates.size() / 3;

        if (numberElementsInAtomCompactSupport > 0)
          {
            d_CMatrixEntriesConjugate[ChargeId].resize(
              numberElementsInAtomCompactSupport);
            d_CMatrixEntriesTranspose[ChargeId].resize(
              numberElementsInAtomCompactSupport);
          }
#ifdef USE_COMPLEX
        std::vector<double> sphericalFunctionBasisRealTimesJxW(
          numberElementsInAtomCompactSupport * maxkPoints *
            NumTotalSphericalFunctions * numberQuadraturePoints,
          0.0);
        std::vector<double> sphericalFunctionBasisImagTimesJxW(
          numberElementsInAtomCompactSupport * maxkPoints *
            NumTotalSphericalFunctions * numberQuadraturePoints,
          0.0);
#else
        std::vector<double> sphericalFunctionBasisTimesJxW(
          numberElementsInAtomCompactSupport * NumTotalSphericalFunctions *
            numberQuadraturePoints,
          0.0);
#endif
        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            const unsigned int elementIndex =
              elementIndexesInAtomCompactSupport[iElemComp];
            for (unsigned int alpha = 0; alpha < NumRadialSphericalFunctions;
                 ++alpha)
              {
                std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
                  sphericalFunction.find(std::make_pair(Znum, alpha))->second;
                unsigned int       lQuantumNumber = sphFn->getQuantumNumberl();
                const unsigned int startIndex =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalSphericalFunctionIndexStart(Znum, alpha);
                unsigned int endIndex = startIndex + 2 * lQuantumNumber + 1;
                std::vector<double> sphericalFunctionBasisReal(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  0.0);
                std::vector<double> sphericalFunctionBasisImag(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  0.0);
                std::vector<dataTypes::number> sphericalFunctionBasis(
                  maxkPoints * numberQuadraturePoints *
                    (2 * lQuantumNumber + 1),
                  ValueType(0.0));
                std::vector<dataTypes::number>
                  sphericalFunctionBasisTimesImageDist(
                    maxkPoints * numberQuadraturePoints *
                      (2 * lQuantumNumber + 1) * 3,
                    ValueType(0.0));
                for (int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
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

                    for (int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        x[0] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint] -
                          chargePoint[0];
                        x[1] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint + 1] -
                          chargePoint[1];
                        x[2] =
                          quadraturePointsVector[elementIndex *
                                                   numberQuadraturePoints * 3 +
                                                 3 * iQuadPoint + 2] -
                          chargePoint[2];
                        sphericalHarmonicUtils::convertCartesianToSpherical(
                          x, r, theta, phi);
                        if (r <= sphFn->getRadialCutOff())
                          {
                            radialVal = sphFn->getRadialValue(r);

                            unsigned int tempIndex = 0;
                            for (int mQuantumNumber = int(-lQuantumNumber);
                                 mQuantumNumber <= int(lQuantumNumber);
                                 mQuantumNumber++)
                              {
                                sphericalHarmonicUtils::getSphericalHarmonicVal(
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
                                for (int kPoint = 0; kPoint < maxkPoints;
                                     ++kPoint)
                                  {
                                    angle =
                                      d_kPointCoordinates[3 * kPoint + 0] *
                                        pointMinusLatticeVector[0] +
                                      d_kPointCoordinates[3 * kPoint + 1] *
                                        pointMinusLatticeVector[1] +
                                      d_kPointCoordinates[3 * kPoint + 2] *
                                        pointMinusLatticeVector[2];

                                    sphericalFunctionBasisReal
                                      [kPoint * numberQuadraturePoints *
                                         (2 * lQuantumNumber + 1) +
                                       tempIndex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      cos(angle) * sphericalFunctionValue;
                                    sphericalFunctionBasisImag
                                      [kPoint * numberQuadraturePoints *
                                         (2 * lQuantumNumber + 1) +
                                       tempIndex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      -sin(angle) * sphericalFunctionValue;

                                    sphericalFunctionBasis
                                      [kPoint * numberQuadraturePoints *
                                         (2 * lQuantumNumber + 1) +
                                       tempIndex * numberQuadraturePoints +
                                       iQuadPoint] +=
                                      ValueType(cos(angle) *
                                                  sphericalFunctionValue,
                                                -sin(angle) *
                                                  sphericalFunctionValue);

                                    for (unsigned int iDim = 0; iDim < 3;
                                         ++iDim)
                                      sphericalFunctionBasisTimesImageDist
                                        [kPoint * numberQuadraturePoints *
                                           (2 * lQuantumNumber + 1) * 3 +
                                         tempIndex * numberQuadraturePoints *
                                           3 +
                                         iQuadPoint * 3 + iDim] +=
                                        dataTypes::number(
                                          cos(angle) * sphericalFunctionValue *
                                            x[iDim],
                                          -sin(angle) * sphericalFunctionValue *
                                            x[iDim]);
                                  } // k-Point Loop
#else
                                sphericalFunctionBasis
                                  [tempIndex * numberQuadraturePoints +
                                   iQuadPoint] += sphericalFunctionValue;
                                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                                  sphericalFunctionBasisTimesImageDist
                                    [tempIndex * numberQuadraturePoints * 3 +
                                     iQuadPoint * 3 + iDim] +=
                                    sphericalFunctionValue * x[iDim];
                                  // sphericalFunctionBasis[iQuadPoint] +=
                                  // sphericalFunctionValue;
#endif
                                tempIndex++;
                              } // Angular momentum m loop
                          }     // inside r <= Rmax

                      } // quad loop

                  } // image atom loop
                const unsigned int startIndex1 =
                  d_nonTrivialSphericalFnsCellStartIndex[elementIndex];
                const unsigned int startIndex2 =
                  globalAtomIdToNonTrivialSphericalFnsCellStartIndex
                    [ChargeId][elementIndex];
                for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  {
                    for (unsigned int tempIndex = startIndex;
                         tempIndex < endIndex;
                         tempIndex++)
                      {
                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          d_atomCenteredKpointIndexedSphericalFnQuadValues
                            [kPoint * d_sumNonTrivialSphericalFnOverAllCells *
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

                        for (int iQuadPoint = 0;
                             iQuadPoint < numberQuadraturePoints;
                             ++iQuadPoint)
                          for (unsigned int iDim = 0; iDim < 3; ++iDim)
                            d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues
                              [kPoint * d_sumNonTrivialSphericalFnOverAllCells *
                                 numberQuadraturePoints * 3 +
                               startIndex1 * numberQuadraturePoints * 3 +
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



#ifdef USE_COMPLEX
                for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
                  for (unsigned int beta = startIndex; beta < endIndex; beta++)
                    {
                      for (int iQuadPoint = 0;
                           iQuadPoint < numberQuadraturePoints;
                           ++iQuadPoint)
                        {
                          sphericalFunctionBasisRealTimesJxW
                            [iElemComp * maxkPoints *
                               NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             kPoint * NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             beta * numberQuadraturePoints + iQuadPoint] =
                              sphericalFunctionBasisReal
                                [kPoint * numberQuadraturePoints *
                                   (2 * lQuantumNumber + 1) +
                                 (beta - startIndex) * numberQuadraturePoints +
                                 iQuadPoint] *
                              real(JxwVector[elementIndex *
                                               numberQuadraturePoints +
                                             iQuadPoint]);
                          sphericalFunctionBasisImagTimesJxW
                            [iElemComp * maxkPoints *
                               NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             kPoint * NumTotalSphericalFunctions *
                               numberQuadraturePoints +
                             beta * numberQuadraturePoints + iQuadPoint] =
                              sphericalFunctionBasisImag
                                [kPoint * numberQuadraturePoints *
                                   (2 * lQuantumNumber + 1) +
                                 (beta - startIndex) * numberQuadraturePoints +
                                 iQuadPoint] *
                              real(JxwVector[elementIndex *
                                               numberQuadraturePoints +
                                             iQuadPoint]);
                        } // quadPoint

                      // sphericalFunctionBasisTimesJxW
                      //   [iElemComp * maxkPoints * NumTotalSphericalFunctions
                      //   *
                      //      numberQuadraturePoints +
                      //    kPoint * NumTotalSphericalFunctions *
                      //      numberQuadraturePoints +
                      //    beta * numberQuadraturePoints + iQuadPoint] =
                      //     sphericalFunctionBasis[kPoint *
                      //                                  numberQuadraturePoints
                      //                                  +
                      //                                iQuadPoint] *
                      //     JxwVector[elementIndex*numberQuadraturePoints +
                      //     iQuadPoint];
                    } // beta
#else
                for (unsigned int beta = startIndex; beta < endIndex; beta++)
                  {
                    for (int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        sphericalFunctionBasisTimesJxW
                          [iElemComp * NumTotalSphericalFunctions *
                             numberQuadraturePoints +
                           beta * numberQuadraturePoints + iQuadPoint] =
                            sphericalFunctionBasis[(beta - startIndex) *
                                                     numberQuadraturePoints +
                                                   iQuadPoint] *
                            JxwVector[elementIndex * numberQuadraturePoints +
                                      iQuadPoint];

                        // sphericalFunctionBasisTimesJxW[iElemComp *
                        // NumTotalSphericalFunctions *
                        //                         numberQuadraturePoints +
                        //                       beta * numberQuadraturePoints +
                        //                       iQuadPoint] =
                        //   sphericalFunctionBasis[iQuadPoint] *
                        //   JxwVector[elementIndex*numberQuadraturePoints +
                        //   iQuadPoint];
                      } // quadPoint
                  }     // beta
#endif
              } // alpha loop


          } // element loop

        const char         transA = 'N', transB = 'N';
        const double       scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
        const unsigned int inc = 1;
        const unsigned int n = numberElementsInAtomCompactSupport * maxkPoints *
                               NumTotalSphericalFunctions;
        const unsigned int  m = d_numberNodesPerElement;
        const unsigned int  k = numberQuadraturePoints;
        std::vector<double> projectorMatricesReal(m * n, 0.0);
        std::vector<double> projectorMatricesImag(m * n, 0.0);
        // std::vector<ValueType> projectorMatricesReal(m * n, 0.0);
        if (numberElementsInAtomCompactSupport > 0)
          {
#ifdef USE_COMPLEX
            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &sphericalFunctionBasisRealTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesReal[0],
                   &m);

            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &sphericalFunctionBasisImagTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesImag[0],
                   &m);
#else
            dgemm_(&transA,
                   &transB,
                   &m,
                   &n,
                   &k,
                   &scalarCoeffAlpha,
                   &shapeValQuads[0],
                   &m,
                   &sphericalFunctionBasisTimesJxW[0],
                   &k,
                   &scalarCoeffBeta,
                   &projectorMatricesReal[0],
                   &m);
#endif
            // d_BLASWrapperPtrHost->xgemm(&transA,
            //        &transB,
            //        &m,
            //        &n,
            //        &k,
            //        &scalarCoeffAlpha,
            //        &shapeValQuads[0],
            //        &m,
            //        &sphericalFunctionBasisTimesJxW[0],
            //        &k,
            //        &scalarCoeffBeta,
            //        &projectorMatrices[0],
            //        &m);
          }

        for (int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport;
             ++iElemComp)
          {
            d_CMatrixEntriesConjugate[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));
            d_CMatrixEntriesTranspose[ChargeId][iElemComp].resize(
              d_numberNodesPerElement * NumTotalSphericalFunctions * maxkPoints,
              ValueType(0.0));

            std::vector<ValueType> &CMatrixEntriesConjugateAtomElem =
              d_CMatrixEntriesConjugate[ChargeId][iElemComp];


            std::vector<ValueType> &CMatrixEntriesTransposeAtomElem =
              d_CMatrixEntriesTranspose[ChargeId][iElemComp];



            for (int kPoint = 0; kPoint < maxkPoints; ++kPoint)
              {
                for (int beta = 0; beta < NumTotalSphericalFunctions; ++beta)
                  for (int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
                    {
                      const unsigned int flattenedIndex =
                        iElemComp * maxkPoints * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        kPoint * NumTotalSphericalFunctions *
                          d_numberNodesPerElement +
                        beta * d_numberNodesPerElement + iNode;
                      const double tempReal =
                        projectorMatricesReal[flattenedIndex];
                      const double tempImag =
                        projectorMatricesImag[flattenedIndex];
                      if (isnan(tempReal))
                        std::cout
                          << "Real->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << beta << " " << iNode << std::endl;
                      if (isnan(tempImag))
                        std::cout
                          << "Imag->Processor number and indices has nan: "
                          << d_this_mpi_process << " " << iElemComp << " "
                          << kPoint << " "
                          << " " << beta << " " << iNode << std::endl;
                        // const ValueType temp =
                        // projectorMatrices[flattenedIndex];
#ifdef USE_COMPLEX
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * beta + iNode]
                          .real(tempReal);
                      CMatrixEntriesConjugateAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         d_numberNodesPerElement * beta + iNode]
                          .imag(-tempImag);

                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + beta]
                          .real(tempReal);
                      CMatrixEntriesTransposeAtomElem
                        [kPoint * d_numberNodesPerElement *
                           NumTotalSphericalFunctions +
                         NumTotalSphericalFunctions * iNode + beta]
                          .imag(tempImag);



#else
                      CMatrixEntriesConjugateAtomElem[d_numberNodesPerElement *
                                                        beta +
                                                      iNode] = tempReal;

                      CMatrixEntriesTransposeAtomElem
                        [NumTotalSphericalFunctions * iNode + beta] = tempReal;
#endif
                    } // node loop
              }       // k point loop
          }           // non-trivial element loop



      } // ChargeId loop
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (unsigned int iCell = 0; iCell < d_locallyOwnedCells; iCell++)
          {
            if (atomSupportInElement(iCell))
              {
                d_nonlocalElemIdToCellIdVector.push_back(iCell);
                for (int iNode = 0; iNode < d_numberNodesPerElement; iNode++)
                  {
                    dftfe::global_size_type localNodeId =
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
        d_cellHamiltonianMatrixNonLocalFlattenedConjugate.clear();
        d_cellHamiltonianMatrixNonLocalFlattenedConjugate.resize(
          d_kPointWeights.size() * d_totalNonlocalElems *
            d_numberNodesPerElement * d_maxSingleAtomContribution,
          ValueType(0.0));
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.clear();
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
          d_kPointWeights.size() * d_totalNonlocalElems *
            d_numberNodesPerElement * d_maxSingleAtomContribution,
          ValueType(0.0));
        std::vector<unsigned int> atomIdsInCurrentProcess =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<unsigned int> &atomicNumber =
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
        d_sphericalFnTimesVectorAllCellsReduction.clear();
        d_sphericalFnTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomContribution *
            d_totalNonLocalEntries,
          ValueType(0.0));
        d_cellNodeIdMapNonLocalToLocal.clear();
        d_cellNodeIdMapNonLocalToLocal.resize(d_totalNonlocalElems *
                                              d_numberNodesPerElement);



        unsigned int countElemNode    = 0;
        unsigned int countElem        = 0;
        unsigned int countAlpha       = 0;
        unsigned int numShapeFnsAccum = 0;

        int totalElements = 0;
        for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            const unsigned int        atomId = atomIdsInCurrentProcess[iAtom];
            std::vector<unsigned int> elementIndexesInAtomCompactSupport =
              d_atomCenteredSphericalFunctionContainer
                ->d_elementIndexesInAtomCompactSupport[atomId];
            unsigned int totalAtomIdElementIterators =
              elementIndexesInAtomCompactSupport.size();
            totalElements += totalAtomIdElementIterators;
            const unsigned int Znum = atomicNumber[atomId];
            const unsigned int numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


            for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                unsigned int globalId =
                  d_sphericalFunctionIdsNumberingMapCurrentProcess
                    [std::make_pair(atomId, alpha)];

                const unsigned int id = d_SphericalFunctionKetTimesVectorPar[0]
                                          .get_partitioner()
                                          ->global_to_local(globalId);

                d_sphericalFnIdsParallelNumberingMap[countAlpha] = id;
                d_sphericalFnIdsPaddedParallelNumberingMap
                  [iAtom * d_maxSingleAtomContribution + alpha] = id;
                for (unsigned int iElemComp = 0;
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
            for (unsigned int iElemComp = 0;
                 iElemComp < totalAtomIdElementIterators;
                 ++iElemComp)
              {
                const unsigned int elementId =
                  elementIndexesInAtomCompactSupport[iElemComp];

                for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    dftfe::global_size_type localNodeId =
                      basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                        [elementId * d_numberNodesPerElement + iNode];
                    d_cellNodeIdMapNonLocalToLocal[countElemNode] =
                      elementId * d_numberNodesPerElement + iNode;
                    countElemNode++;
                  }
              }

            for (unsigned int iElemComp = 0;
                 iElemComp < totalAtomIdElementIterators;
                 ++iElemComp)
              {
                const unsigned int elementId =
                  elementIndexesInAtomCompactSupport[iElemComp];
                d_nonlocalElemIdToLocalElemIdMap[countElem] = elementId;

                for (unsigned int ikpoint = 0; ikpoint < d_kPointWeights.size();
                     ikpoint++)
                  for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                       ++iNode)
                    {
                      for (unsigned int alpha = 0;
                           alpha < numberSphericalFunctions;
                           ++alpha)
                        {
                          d_cellHamiltonianMatrixNonLocalFlattenedConjugate
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

                          d_cellHamiltonianMatrixNonLocalFlattenedTranspose
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


                for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                     ++alpha)
                  {
                    const unsigned int columnStartId =
                      (numShapeFnsAccum + alpha) * d_totalNonlocalElems *
                      d_maxSingleAtomContribution;
                    const unsigned int columnRowId =
                      countElem * d_maxSingleAtomContribution + alpha;
                    d_sphericalFnTimesVectorAllCellsReduction[columnStartId +
                                                              columnRowId] =
                      ValueType(1.0);
                  }

                countElem++;
              }

            numShapeFnsAccum += numberSphericalFunctions;
          }

        if (!d_memoryOptMode)
          {
            d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.resize(
              d_cellHamiltonianMatrixNonLocalFlattenedConjugate.size());
            d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.copyFrom(
              d_cellHamiltonianMatrixNonLocalFlattenedConjugate);

            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.resize(
              d_cellHamiltonianMatrixNonLocalFlattenedTranspose.size());
            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.copyFrom(
              d_cellHamiltonianMatrixNonLocalFlattenedTranspose);
          }
        else
          {
            d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice.resize(
              d_cellHamiltonianMatrixNonLocalFlattenedConjugate.size() /
              d_kPointWeights.size());
            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.resize(
              d_cellHamiltonianMatrixNonLocalFlattenedTranspose.size() /
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
        d_sphericalFnTimesVectorAllCellsReductionDevice.clear();
        d_sphericalFnTimesVectorAllCellsReductionDevice.resize(
          d_sphericalFnTimesVectorAllCellsReduction.size());
        d_sphericalFnTimesVectorAllCellsReductionDevice.copyFrom(
          d_sphericalFnTimesVectorAllCellsReduction);

        d_cellNodeIdMapNonLocalToLocalDevice.clear();
        d_cellNodeIdMapNonLocalToLocalDevice.resize(
          d_cellNodeIdMapNonLocalToLocal.size());

        d_cellNodeIdMapNonLocalToLocalDevice.copyFrom(
          d_cellNodeIdMapNonLocalToLocal);
        d_nonlocalElemIdToCellIdVector.clear();
        d_flattenedNonLocalCellDofIndexToProcessDofIndexVector.clear();
        for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
          {
            unsigned int iCell = d_nonlocalElemIdToLocalElemIdMap[i];

            d_nonlocalElemIdToCellIdVector.push_back(iCell);
            for (int iNode = 0; iNode < d_numberNodesPerElement; iNode++)
              {
                dftfe::global_size_type localNodeId =
                  basisOperationsPtr->d_cellDofIndexToProcessDofIndexMap
                    [iCell * d_numberNodesPerElement + iNode];
                d_flattenedNonLocalCellDofIndexToProcessDofIndexVector
                  .push_back(localNodeId);
              }
          }
        freeDeviceVectors();
        hostWfcPointers =
          (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));
        hostPointerCDagger =
          (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));
        hostPointerCDaggeOutTemp =
          (ValueType **)malloc(d_totalNonlocalElems * sizeof(ValueType *));


        dftfe::utils::deviceMalloc((void **)&deviceWfcPointers,
                                   d_totalNonlocalElems * sizeof(ValueType *));


        dftfe::utils::deviceMalloc((void **)&devicePointerCDagger,
                                   d_totalNonlocalElems * sizeof(ValueType *));

        dftfe::utils::deviceMalloc((void **)&devicePointerCDaggerOutTemp,
                                   d_totalNonlocalElems * sizeof(ValueType *));

        d_isMallocCalled = true;
        if (d_memoryOptMode)
          {
            for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
              {
                hostPointerCDagger[i] =
                  d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice
                    .begin() +
                  i * d_numberNodesPerElement * d_maxSingleAtomContribution;
              }

            dftfe::utils::deviceMemcpyH2D(devicePointerCDagger,
                                          hostPointerCDagger,
                                          d_totalNonlocalElems *
                                            sizeof(ValueType *));
          }
      }



#endif
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseFlattenedDataStructure(
      unsigned int waveFunctionBlockSize,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened)
  {
    std::vector<dftfe::global_size_type> tempNonLocalCellDofVector(
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

    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        d_numberWaveFunctions = waveFunctionBlockSize;

        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
          waveFunctionBlockSize,
          sphericalFunctionKetTimesVectorParFlattened);
        d_sphericalFnTimesWavefunMatrix.clear();
        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        const std::vector<unsigned int> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
          {
            unsigned int atomId = atomIdsInProcessor[iAtom];
            unsigned int Znum   = atomicNumber[atomId];
            unsigned int numberSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            d_sphericalFnTimesWavefunMatrix[atomId].resize(
              numberSphericalFunctions * d_numberWaveFunctions, ValueType(0.0));
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        d_numberWaveFunctions = waveFunctionBlockSize;
        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          d_SphericalFunctionKetTimesVectorPar[0].get_partitioner(),
          waveFunctionBlockSize,
          sphericalFunctionKetTimesVectorParFlattened);
        d_sphericalFnTimesVectorAllCellsDevice.clear();
        d_sphericalFnTimesVectorAllCellsDevice.resize(
          d_totalNonlocalElems * d_numberWaveFunctions *
            d_maxSingleAtomContribution,
          ValueType(0.0));
        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomCenteredSphericalFunctionContainer
            ->getAtomIdsInCurrentProcess();
        d_sphericalFnTimesVectorDevice.clear();
        d_sphericalFnTimesVectorDevice.resize(atomIdsInProcessor.size() *
                                                d_numberWaveFunctions *
                                                d_maxSingleAtomContribution,
                                              ValueType(0.0));
        d_couplingMatrixTimesVectorDevice.clear();
        d_couplingMatrixTimesVectorDevice.resize(atomIdsInProcessor.size() *
                                                   d_numberWaveFunctions *
                                                   d_maxSingleAtomContribution,
                                                 ValueType(0.0));

        d_cellHamMatrixTimesWaveMatrixNonLocalDevice.clear();
        d_cellHamMatrixTimesWaveMatrixNonLocalDevice.resize(
          d_numberWaveFunctions * d_totalNonlocalElems *
            d_numberNodesPerElement,
          ValueType(0.0));

        for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
          {
            hostPointerCDaggeOutTemp[i] =
              d_sphericalFnTimesVectorAllCellsDevice.begin() +
              i * d_numberWaveFunctions * d_maxSingleAtomContribution;
          }

        dftfe::utils::deviceMemcpyH2D(devicePointerCDaggerOutTemp,
                                      hostPointerCDaggeOutTemp,
                                      d_totalNonlocalElems *
                                        sizeof(ValueType *));

        d_sphericalFnTimesWavefunctionMatrix.clear();
        d_sphericalFnTimesWavefunctionMatrix.resize(d_numberWaveFunctions *
                                                    d_totalNonLocalEntries);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::initialisePartitioner()
  {
    std::vector<unsigned int> atomIdsInCurrentProcess =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const unsigned int numberAtoms =
      d_atomCenteredSphericalFunctionContainer->getNumAtomCentersSize();
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    // //
    // // data structures for memory optimization of projectorKetTimesVector
    // //
    std::vector<unsigned int> atomIdsAllProcessFlattened;
    MPI_Barrier(d_mpi_communicator);
    pseudoUtils::exchangeLocalList(atomIdsInCurrentProcess,
                                   atomIdsAllProcessFlattened,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<unsigned int> atomIdsSizeCurrentProcess(1);
    atomIdsSizeCurrentProcess[0] = atomIdsInCurrentProcess.size();
    std::vector<unsigned int> atomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(atomIdsSizeCurrentProcess,
                                   atomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);

    std::vector<std::vector<unsigned int>> atomIdsInAllProcess(
      d_n_mpi_processes);
    unsigned int count = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        for (unsigned int j = 0; j < atomIdsSizesAllProcess[iProc]; j++)
          {
            atomIdsInAllProcess[iProc].push_back(
              atomIdsAllProcessFlattened[count]);
            count++;
          }
      }
    atomIdsAllProcessFlattened.clear();

    dealii::IndexSet ownedAtomIdsInCurrentProcess;
    ownedAtomIdsInCurrentProcess.set_size(numberAtoms); // Check this
    ownedAtomIdsInCurrentProcess.add_indices(atomIdsInCurrentProcess.begin(),
                                             atomIdsInCurrentProcess.end());
    dealii::IndexSet ghostAtomIdsInCurrentProcess(ownedAtomIdsInCurrentProcess);
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
      {
        if (iProc < d_this_mpi_process)
          {
            dealii::IndexSet temp;
            temp.set_size(numberAtoms);
            temp.add_indices(atomIdsInAllProcess[iProc].begin(),
                             atomIdsInAllProcess[iProc].end());
            ownedAtomIdsInCurrentProcess.subtract_set(temp);
          }
      }

    ghostAtomIdsInCurrentProcess.subtract_set(ownedAtomIdsInCurrentProcess);

    std::vector<unsigned int> ownedAtomIdsSizeCurrentProcess(1);
    ownedAtomIdsSizeCurrentProcess[0] =
      ownedAtomIdsInCurrentProcess.n_elements();
    std::vector<unsigned int> ownedAtomIdsSizesAllProcess;
    pseudoUtils::exchangeLocalList(ownedAtomIdsSizeCurrentProcess,
                                   ownedAtomIdsSizesAllProcess,
                                   d_n_mpi_processes,
                                   d_mpi_communicator);
    // // renumbering to make contiguous set of nonLocal atomIds
    std::map<int, int> oldToNewAtomIds;
    std::map<int, int> newToOldAtomIds;
    unsigned int       startingCount = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
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
        unsigned int newAtomId = oldToNewAtomIds[*it];
        ghostAtomIdsInCurrentProcessRenum.add_index(newAtomId);
      }

    if (d_this_mpi_process == 0 && false)
      {
        for (std::map<int, int>::const_iterator it = oldToNewAtomIds.begin();
             it != oldToNewAtomIds.end();
             it++)
          std::cout << " old nonlocal atom id: " << it->first
                    << " new nonlocal atomid: " << it->second << std::endl;

        std::cout
          << "number of local owned non local atom ids in all processors"
          << '\n';
        for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
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

    int                       numberLocallyOwnedSphericalFunctions = 0;
    int                       numberGhostSphericalFunctions        = 0;
    std::vector<unsigned int> coarseNodeIdsCurrentProcess;
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

    std::vector<unsigned int> ghostAtomIdNumberSphericalFunctions;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        const unsigned temp = d_atomCenteredSphericalFunctionContainer
                                ->getTotalNumberOfSphericalFunctionsPerAtom(
                                  atomicNumber[newToOldAtomIds[*it]]);
        numberGhostSphericalFunctions += temp;
        ghostAtomIdNumberSphericalFunctions.push_back(temp);
      }

    std::vector<unsigned int>
      numberLocallyOwnedSphericalFunctionsCurrentProcess(1);
    numberLocallyOwnedSphericalFunctionsCurrentProcess[0] =
      numberLocallyOwnedSphericalFunctions;
    std::vector<unsigned int> numberLocallyOwnedSphericalFunctionsAllProcess;
    pseudoUtils::exchangeLocalList(
      numberLocallyOwnedSphericalFunctionsCurrentProcess,
      numberLocallyOwnedSphericalFunctionsAllProcess,
      d_n_mpi_processes,
      d_mpi_communicator);

    startingCount = 0;
    for (unsigned int iProc = 0; iProc < d_n_mpi_processes; iProc++)
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
    std::vector<unsigned int> v(numberLocallyOwnedSphericalFunctions);
    std::iota(std::begin(v), std::end(v), startingCount);
    d_locallyOwnedSphericalFunctionIdsCurrentProcess.add_indices(v.begin(),
                                                                 v.end());

    std::vector<unsigned int> coarseNodeIdsAllProcess;
    for (unsigned int i = 0; i < coarseNodeIdsCurrentProcess.size(); ++i)
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
    unsigned int localGhostCount = 0;
    for (dealii::IndexSet::ElementIterator it =
           ghostAtomIdsInCurrentProcessRenum.begin();
         it != ghostAtomIdsInCurrentProcessRenum.end();
         it++)
      {
        std::vector<unsigned int> g(
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

    for (dealii::IndexSet::ElementIterator it =
           ownedAtomIdsInCurrentProcess.begin();
         it != ownedAtomIdsInCurrentProcess.end();
         it++)
      {
        const int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);
        d_OwnedAtomIdsInCurrentProcessor.push_back(*it);

        for (unsigned int i = 0; i < numberSphericalFunctions; ++i)
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
        const int numberSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[*it]);

        for (unsigned int i = 0; i < numberSphericalFunctions; ++i)
          {
            d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
              *it, i)] = coarseNodeIdsAllProcess[oldToNewAtomIds[*it]] + i;
          }
      }

    if (false)
      {
        for (std::map<std::pair<unsigned int, unsigned int>,
                      unsigned int>::const_iterator it =
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
    std::vector<std::pair<unsigned int, unsigned int>> localIds;
    for (int iAtom = 0; iAtom < atomIdsInCurrentProcess.size(); iAtom++)
      {
        const unsigned int atomId = atomIdsInCurrentProcess[iAtom];
        unsigned int       globalId =
          d_sphericalFunctionIdsNumberingMapCurrentProcess[std::make_pair(
            atomId, 0)];

        const unsigned int id = d_SphericalFunctionKetTimesVectorPar[0]
                                  .get_partitioner()
                                  ->global_to_local(globalId);
        localIds.push_back(std::pair<unsigned int, unsigned int>(id, iAtom));
      }
    std::sort(localIds.begin(), localIds.end());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::getTotalAtomInCurrentProcessor()
    const
  {
    return (d_totalAtomsInCurrentProc);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonLocalElementsInCurrentProcessor() const
  {
    return (d_totalNonlocalElems);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonLocalEntriesCurrentProcessor() const
  {
    return (d_totalNonLocalEntries);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType,
                                 memorySpace>::getMaxSingleAtomEntries() const
  {
    return (d_maxSingleAtomContribution);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::atomSupportInElement(
    unsigned int iElem) const
  {
    return (
      d_atomCenteredSphericalFunctionContainer->atomSupportInElement(iElem));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getGlobalDofAtomIdSphericalFnPair(const unsigned int atomId,
                                      const unsigned int alpha) const
  {
    return d_sphericalFunctionIdsNumberingMapCurrentProcess
      .find(std::make_pair(atomId, alpha))
      ->second;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getLocalIdOfDistributedVec(const unsigned int globalId) const
  {
    return (d_SphericalFunctionKetTimesVectorPar[0]
              .get_partitioner()
              ->global_to_local(globalId));
  }


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
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getSphericalFnTimesVectorFlattenedVectorLocalIds() const

  {
    return d_sphericalFnTimesVectorFlattenedVectorLocalIds;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getAtomIdToNonTrivialSphericalFnCellStartIndex() const
  {
    return d_atomIdToNonTrivialSphericalFnCellStartIndex;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const unsigned int
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getTotalNonTrivialSphericalFnsOverAllCells() const
  {
    return d_sumNonTrivialSphericalFnOverAllCells;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::map<unsigned int, std::vector<unsigned int>> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCellIdToAtomIdsLocalCompactSupportMap() const
  {
    return d_cellIdToAtomIdsLocalCompactSupportMap;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonTrivialSphericalFnsPerCell() const
  {
    return d_nonTrivialSphericalFnPerCell;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonTrivialSphericalFnsCellStartIndex() const
  {
    return d_nonTrivialSphericalFnsCellStartIndex;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<unsigned int> &
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
            const std::vector<unsigned int> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            const std::vector<unsigned int> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            unsigned int       startIndex = 0;
            const unsigned int inc        = 1;

            for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
              {
                const unsigned int atomId = atomIdsInProc[iAtom];
                const unsigned int Znum   = atomicNumber[atomId];
                const unsigned int numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


                for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                     alpha++)
                  {
                    const unsigned int localId =
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
          }
#endif
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const ValueType *
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getCconjtansXLocalDataStructure(const unsigned int iAtom)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const unsigned int atomId = d_atomCenteredSphericalFunctionContainer
                                      ->getAtomIdsInCurrentProcess()[iAtom];
        return d_sphericalFnTimesWavefunMatrix[atomId].begin();
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // extract from d_sphericalFnTimesWavefunctionMatrix
        const std::vector<unsigned int> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const unsigned int Znum = atomicNumber[iAtom];
        const unsigned int numberOfSphericalFunctions =
          d_atomCenteredSphericalFunctionContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        d_tempConjtansX.clear();
        d_tempConjtansX.resize(numberOfSphericalFunctions *
                               d_numberWaveFunctions);
        const unsigned int offset = iAtom * d_maxSingleAtomContribution;
        d_tempConjtansX.copyFrom(d_sphericalFnTimesVectorDevice,
                                 numberOfSphericalFunctions *
                                   d_numberWaveFunctions,
                                 offset * d_numberWaveFunctions,
                                 0);
        return d_tempConjtansX.data();
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyVOnCconjtransX(
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      &                sphericalFunctionKetTimesVectorParFlattened,
    const bool         flagCopyResultsToMatrix,
    const unsigned int kPointIndex)
  {
    if (d_totalNonLocalEntries > 0)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            const std::vector<unsigned int> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            const std::vector<unsigned int> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            if (couplingtype == CouplingStructure::diagonal)
              {
                unsigned int       startIndex = 0;
                const unsigned int inc        = 1;
                for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
                  {
                    const unsigned int atomId = atomIdsInProc[iAtom];
                    const unsigned int Znum   = atomicNumber[atomId];
                    const unsigned int numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);


                    for (unsigned int alpha = 0;
                         alpha < numberSphericalFunctions;
                         alpha++)
                      {
                        ValueType nonlocalConstantV =
                          couplingMatrix[startIndex++];
                        const unsigned int localId =
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
            else if (couplingtype == CouplingStructure::dense)
              {
                unsigned int startIndex = 0;
                unsigned int totalShift =
                  couplingMatrix.size() / d_kPointWeights.size();
                const unsigned int inc   = 1;
                const ValueType    alpha = 1;
                const ValueType    beta  = 0;
                for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
                  {
                    const unsigned int atomId = atomIdsInProc[iAtom];
                    d_sphericalFnTimesWavefunMatrix[atomId].clear();

                    const unsigned int Znum = atomicNumber[atomId];
                    const unsigned int numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    d_sphericalFnTimesWavefunMatrix[atomId].resize(
                      numberSphericalFunctions * d_numberWaveFunctions, 0.0);
                    std::vector<ValueType> nonlocalConstantVmatrix(
                      numberSphericalFunctions * numberSphericalFunctions, 0.0);
                    d_BLASWrapperPtr->xcopy(
                      numberSphericalFunctions * numberSphericalFunctions,
                      &couplingMatrix[kPointIndex * totalShift + startIndex],
                      1,
                      &nonlocalConstantVmatrix[0],
                      1);

                    const unsigned int localId =
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
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool skipComm)
  {
    if (d_totalNonLocalEntries > 0)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          {
            const std::vector<unsigned int> atomIdsInProc =
              d_atomCenteredSphericalFunctionContainer
                ->getAtomIdsInCurrentProcess();
            const std::vector<unsigned int> &atomicNumber =
              d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
            for (int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; iAtom++)
              {
                const unsigned int atomId = atomIdsInProc[iAtom];
                unsigned int       Znum   = atomicNumber[atomId];
                const unsigned int numberSphericalFunctions =
                  d_atomCenteredSphericalFunctionContainer
                    ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                for (unsigned int alpha = 0; alpha < numberSphericalFunctions;
                     alpha++)
                  {
                    const unsigned int id =
                      d_sphericalFunctionIdsNumberingMapCurrentProcess
                        .find(std::make_pair(atomId, alpha))
                        ->second;
                    std::memcpy(
                      sphericalFunctionKetTimesVectorParFlattened.data() +
                        sphericalFunctionKetTimesVectorParFlattened
                            .getMPIPatternP2P()
                            ->globalToLocal(id) *
                          d_numberWaveFunctions,
                      d_sphericalFnTimesWavefunMatrix[atomId].begin() +
                        d_numberWaveFunctions * alpha,
                      d_numberWaveFunctions * sizeof(ValueType));


                    // d_BLASWrapperPtr->xcopy(
                    //   d_numberWaveFunctions,
                    //   &d_sphericalFnTimesWavefunMatrix[atomId]
                    //                                  [d_numberWaveFunctions *
                    //                                  alpha],
                    //   inc,
                    //   sphericalFunctionKetTimesVectorParFlattened.data() +
                    //     sphericalFunctionKetTimesVectorParFlattened.getMPIPatternP2P()
                    //     ->globalToLocal(id) *d_numberWaveFunctions,
                    //   inc);
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
            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              copyToDealiiParallelNonLocalVec(
                d_numberWaveFunctions,
                d_totalNonLocalEntries,
                d_sphericalFnTimesWavefunctionMatrix.begin(),
                sphericalFunctionKetTimesVectorParFlattened.begin(),
                d_sphericalFnIdsParallelNumberingMapDevice.begin());

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
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCconjtransOnX(
    const ValueType *                           X,
    const std::pair<unsigned int, unsigned int> cellRange)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType    zero(0.0), one(1.0);
        const unsigned int inc                            = 1;
        d_AllReduceCompleted                              = false;
        int                              numberOfElements = d_locallyOwnedCells;
        const std::vector<unsigned int> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        const std::map<unsigned int, std::vector<int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();

        double integValue = 0.0;
        for (int iElem = cellRange.first; iElem < cellRange.second; iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);
                int numOfAtomsInElement = atomIdsInElement.size();
                for (int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    unsigned int       atomId = atomIdsInElement[iAtom];
                    unsigned int       Znum   = atomicNumber[atomId];
                    const unsigned int numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const int nonZeroElementMatrixId =
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

                    //                    for ( unsigned int iOrb = 0; iOrb <
                    //                    d_CMatrixEntriesConjugate[atomId][nonZeroElementMatrixId].size();
                    //                        iOrb++)
                    //                      {
                    //                        integValue +=
                    //                        d_CMatrixEntriesConjugate[atomId][nonZeroElementMatrixId][iOrb];
                    //                      }

                  } // iAtom
              }
          } // iElem
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        // Assert check cellRange.second - cellRange.first != d_nonlocalElements
        AssertThrow(
          cellRange.second - cellRange.first == d_locallyOwnedCells,
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent cellRange in use. All the nonlocal Cells must be in range."));
        // Xpointer not same assert check
        AssertThrow(
          X == d_wfcStartPointer,
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent X called. Make sure the input X is correct."));
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);

        d_BLASWrapperPtr->xgemmBatched(
          'N',
          'N',
          d_numberWaveFunctions,
          d_maxSingleAtomContribution,
          d_numberNodesPerElement,
          &scalarCoeffAlpha,
          //(X.data() + cellRange.first),
          (const ValueType **)deviceWfcPointers,
          d_numberWaveFunctions,
          //(devicePointerCDagger.data() + cellRange.first),
          (const ValueType **)devicePointerCDagger,
          d_numberNodesPerElement,
          &scalarCoeffBeta,
          devicePointerCDaggerOutTemp,
          // devicePointerCDaggerOutTemp.data() + cellRange.first,
          d_numberWaveFunctions,
          d_totalNonlocalElems);



        d_BLASWrapperPtr->xgemm(
          'N',
          'N',
          d_numberWaveFunctions,
          d_totalNonLocalEntries,
          d_totalNonlocalElems * d_maxSingleAtomContribution,
          &scalarCoeffAlpha,
          d_sphericalFnTimesVectorAllCellsDevice.begin(),
          d_numberWaveFunctions,
          d_sphericalFnTimesVectorAllCellsReductionDevice.begin(),
          d_totalNonlocalElems * d_maxSingleAtomContribution,
          &scalarCoeffBeta,
          d_sphericalFnTimesWavefunctionMatrix.begin(),
          d_numberWaveFunctions);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCVCconjtransOnX(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    const unsigned int                                         kPointIndex,
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      &sphericalFunctionKetTimesVectorParFlattened,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &dst)
  {
    const unsigned int inc = 1;
    applyVCconjtransOnX(src,
                        kPointIndex,
                        couplingtype,
                        couplingMatrix,
                        sphericalFunctionKetTimesVectorParFlattened,
                        true);
    dftfe::utils::MemoryStorage<ValueType, memorySpace> Xtemp;
    Xtemp.resize(d_locallyOwnedCells * d_numberNodesPerElement *
                   d_numberWaveFunctions,
                 0.0);
    applyCOnVCconjtransX(Xtemp.data(), std::make_pair(0, d_locallyOwnedCells));
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        for (unsigned int iCell = 0; iCell < d_locallyOwnedCells; ++iCell)
          {
            for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                 ++iNode)
              {
                dealii::types::global_dof_index localNodeId =
                  (d_basisOperatorPtr->d_cellDofIndexToProcessDofIndexMap
                     [iCell * d_numberNodesPerElement + iNode]) *
                  d_numberWaveFunctions;
                d_BLASWrapperPtr->xcopy(d_numberWaveFunctions,
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



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyVCconjtransOnX(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    const unsigned int                                         kPointIndex,
    const CouplingStructure                                    couplingtype,
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      &        sphericalFunctionKetTimesVectorParFlattened,
    const bool flagScaleInternalMatrix)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        initialiseOperatorActionOnX(kPointIndex);
        sphericalFunctionKetTimesVectorParFlattened.setValue(0.0);

        const unsigned int inc = 1;
        dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
          cellWaveFunctionMatrix;

        cellWaveFunctionMatrix.resize(d_numberNodesPerElement *
                                        d_numberWaveFunctions,
                                      0.0);


        if (d_totalNonlocalElems)
          {
            for (unsigned int iCell = 0; iCell < d_locallyOwnedCells; ++iCell)
              {
                if (atomSupportInElement(iCell))
                  {
                    for (unsigned int iNode = 0;
                         iNode < d_numberNodesPerElement;
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
                      std::pair<unsigned int, unsigned int>(iCell, iCell + 1));

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
        initialiseCellWaveFunctionPointers(cellWaveFunctionMatrix);
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
              std::pair<unsigned int, unsigned int>(0, d_locallyOwnedCells));
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
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::applyCOnVCconjtransX(
    ValueType *                                 Xout,
    const std::pair<unsigned int, unsigned int> cellRange)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        const ValueType                                zero(0.0), one(1.0);
        const unsigned int                             inc = 1;
        const std::map<unsigned int, std::vector<int>> sparsityPattern =
          d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
        const std::vector<unsigned int> &atomicNumber =
          d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
        for (int iElem = cellRange.first; iElem < cellRange.second; iElem++)
          {
            if (atomSupportInElement(iElem))
              {
                const std::vector<int> atomIdsInElement =
                  d_atomCenteredSphericalFunctionContainer->getAtomIdsInElement(
                    iElem);


                int numOfAtomsInElement = atomIdsInElement.size();
                for (int iAtom = 0; iAtom < numOfAtomsInElement; iAtom++)
                  {
                    unsigned int atomId = atomIdsInElement[iAtom];

                    unsigned int       Znum = atomicNumber[atomId];
                    const unsigned int numberSphericalFunctions =
                      d_atomCenteredSphericalFunctionContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                    const int nonZeroElementMatrixId =
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
        // Assert check cellRange.second - cellRange.first != d_nonlocalElements
        AssertThrow(
          cellRange.second - cellRange.first == d_locallyOwnedCells,
          dealii::ExcMessage(
            "DFT-FE Error: Inconsistent cellRange in use. All the nonlocal Cells must be in range."));
        long long int strideA =
          d_numberWaveFunctions * d_maxSingleAtomContribution;
        long long int strideB =
          d_maxSingleAtomContribution * d_numberNodesPerElement;
        long long int strideC = d_numberWaveFunctions * d_numberNodesPerElement;
        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);


        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          d_numberWaveFunctions,
          d_numberNodesPerElement,
          d_maxSingleAtomContribution,
          &scalarCoeffAlpha,
          d_sphericalFnTimesVectorAllCellsDevice.begin(),
          d_numberWaveFunctions,
          strideA,
          d_memoryOptMode ?
            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.begin() :
            d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.begin() +
              d_kPointIndex * d_totalNonlocalElems *
                d_maxSingleAtomContribution * d_numberNodesPerElement,
          d_maxSingleAtomContribution,
          strideB,
          &scalarCoeffBeta,
          d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin(),
          d_numberWaveFunctions,
          strideC,
          d_totalNonlocalElems);

        for (unsigned int iAtom = 0; iAtom < d_totalAtomsInCurrentProc; ++iAtom)
          {
            const unsigned int accum  = d_numberCellsAccumNonLocalAtoms[iAtom];
            const unsigned int Ncells = d_numberCellsForEachAtom[iAtom];

            dftfe::AtomicCenteredNonLocalOperatorKernelsDevice::
              addNonLocalContribution(
                Ncells,
                d_numberNodesPerElement,
                d_numberWaveFunctions,
                accum,
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice,
                Xout,
                d_cellNodeIdMapNonLocalToLocalDevice);
          }
      }
#endif
  }
#if defined(DFTFE_WITH_DEVICE)
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix)
  {
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
          {
            hostWfcPointers[i] = cellWaveFunctionMatrix.begin() +
                                 d_nonlocalElemIdToLocalElemIdMap[i] *
                                   d_numberWaveFunctions *
                                   d_numberNodesPerElement;
          }
        d_wfcStartPointer = cellWaveFunctionMatrix.begin();
        dftfe::utils::deviceMemcpyH2D(deviceWfcPointers,
                                      hostWfcPointers,
                                      d_totalNonlocalElems *
                                        sizeof(ValueType *));
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::freeDeviceVectors()
  {
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        if (d_isMallocCalled)
          {
            free(hostWfcPointers);
            dftfe::utils::deviceFree(deviceWfcPointers);
            free(hostPointerCDagger);
            free(hostPointerCDaggeOutTemp);
            dftfe::utils::deviceFree(devicePointerCDagger);
            dftfe::utils::deviceFree(devicePointerCDaggerOutTemp);
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
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const unsigned int totalEntries =
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
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const unsigned int totalEntries =
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
      const unsigned int quadratureIndex)
  {
    if (updateSparsity)
      initialisePartitioner();
    initKpoints(kPointWeights, kPointCoordinates);
    computeCMatrixEntries(basisOperationsPtr, quadratureIndex);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getNonlocalElementToCellIdVector() const
  {
    return (d_nonlocalElemIdToCellIdVector);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getFlattenedNonLocalCellDofIndexToProcessDofIndexMap() const
  {
    return (d_flattenedNonLocalCellDofIndexToProcessDofIndexMap);
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::paddingCouplingMatrix(
    const std::vector<ValueType> &entries,
    std::vector<ValueType> &      entriesPadded,
    const CouplingStructure       couplingtype)
  {
    entriesPadded.clear();
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomCenteredSphericalFunctionContainer->getAtomIdsInCurrentProcess();
    const std::vector<unsigned int> atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    if (couplingtype == CouplingStructure::diagonal)
      {
        entriesPadded.resize(atomIdsInProcessor.size() *
                             d_maxSingleAtomContribution);
        unsigned int index = 0;
        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const unsigned int atomId = atomIdsInProcessor[iAtom];
            const unsigned int Znum   = atomicNumber[atomId];
            const unsigned int numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            for (unsigned int alpha = 0;
                 alpha < numberOfTotalSphericalFunctions;
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
        unsigned int index = 0;
        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const unsigned int atomId = atomIdsInProcessor[iAtom];
            const unsigned int Znum   = atomicNumber[atomId];
            const unsigned int numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

            for (int alpha_i = 0; alpha_i < numberOfTotalSphericalFunctions;
                 alpha_i++)
              {
                for (int alpha_j = 0; alpha_j < numberOfTotalSphericalFunctions;
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
        unsigned int index = 0;
        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const unsigned int atomId = atomIdsInProcessor[iAtom];
            const unsigned int Znum   = atomicNumber[atomId];
            const unsigned int numberOfTotalSphericalFunctions =
              d_atomCenteredSphericalFunctionContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

            for (int alpha_i = 0; alpha_i < 2 * numberOfTotalSphericalFunctions;
                 alpha_i++)
              {
                for (int alpha_j = 0;
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
  std::vector<ValueType>
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::getCmatrixEntries(
    int          kPointIndex,
    unsigned int atomId,
    int          iElem) const
  {
    const std::vector<unsigned int> &atomicNumber =
      d_atomCenteredSphericalFunctionContainer->getAtomicNumbers();
    const std::map<unsigned int, std::vector<int>> sparsityPattern =
      d_atomCenteredSphericalFunctionContainer->getSparsityPattern();
    const int nonZeroElementMatrixId =
      sparsityPattern.find(atomId)->second[iElem];
    const unsigned int numberSphericalFunctions =
      d_atomCenteredSphericalFunctionContainer
        ->getTotalNumberOfSphericalFunctionsPerAtom(atomicNumber[atomId]);
    std::vector<ValueType> Ctemp(d_numberNodesPerElement *
                                   numberSphericalFunctions,
                                 0.0);

    for (int i = 0; i < Ctemp.size(); i++)
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
  const std::vector<unsigned int> &
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    getOwnedAtomIdsInCurrentProcessor() const
  {
    return d_OwnedAtomIdsInCurrentProcessor;
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  AtomicCenteredNonLocalOperator<ValueType, memorySpace>::
    atomPresentInCellRange(
      const std::pair<unsigned int, unsigned int> cellRange) const
  {
    bool flag = false;
    for (unsigned int iElem = cellRange.first; iElem < cellRange.second;
         iElem++)
      {
        flag =
          d_atomCenteredSphericalFunctionContainer->atomSupportInElement(iElem);
        if (flag == true)
          return true;
      }
    return flag;
  }

  template class AtomicCenteredNonLocalOperator<
    dataTypes::number,
    dftfe::utils::MemorySpace::HOST>;
  template class AtomicCenteredNonLocalOperator<
    dataTypes::numberFP32,
    dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class AtomicCenteredNonLocalOperator<
    dataTypes::number,
    dftfe::utils::MemorySpace::DEVICE>;
  template class AtomicCenteredNonLocalOperator<
    dataTypes::numberFP32,
    dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
