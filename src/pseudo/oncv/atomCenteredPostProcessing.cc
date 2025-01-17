#include <atomCenteredPostProcessing.h>
#include <deviceKernelsGeneric.h>
#include <constants.h>
#include <cassert>
#include <iomanip>

namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>

  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    atomCenteredOrbitalsPostProcessing(const MPI_Comm &   mpi_comm_parent,
                                       const MPI_Comm &   mpi_comm_domain,
                                       const std::string &scratchFolderName,
                                       const std::set<unsigned int> &atomTypes,
                                       const bool           reproducibleOutput,
                                       const int            verbosity,
                                       const bool           useDevice,
                                       const dftParameters *dftParamsPtr)
    : d_mpiCommParentPostProcessing(mpi_comm_parent)
    , d_mpiCommDomain(mpi_comm_domain)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_parent,
                      pcout,
                      dftParamsPtr->reproducible_output ||
                          dftParamsPtr->verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::every_call_and_summary,
                      dealii::TimerOutput::wall_times)
  {
    d_dftfeScratchFolderName = scratchFolderName;
    d_atomTypes              = atomTypes;
    d_reproducible_output    = reproducibleOutput;
    d_verbosity              = verbosity;
    d_useDevice              = useDevice;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::smearFunction(
    double               x,
    const dftParameters *dftParamsPtr)
  {
    double sigma = C_kb * dftParamsPtr->smearTval;
    double arg   = std::min((x / sigma) * (x / sigma), 200.0);
    return (std::exp(-arg) / sqrt(M_PI)) / (sigma * C_haToeV);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::initialise(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsDevicePtr,
#endif
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtrDevice,
#endif
    unsigned int                            sparsityPatternQuadratureId,
    unsigned int                            nlpspQuadratureId,
    const std::vector<std::vector<double>> &atomLocations,
    unsigned int                            numEigenValues)
  {
    MPI_Barrier(d_mpiCommDomain);
    d_BasisOperatorHostPtr = basisOperationsHostPtr;
    d_BLASWrapperHostPtr   = BLASWrapperPtrHost;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr   = BLASWrapperPtrDevice;
    d_BasisOperatorDevicePtr = basisOperationsDevicePtr;
#endif

    std::vector<unsigned int> atomicNumbers;
    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
      }

    d_sparsityPatternQuadratureId = sparsityPatternQuadratureId;
    d_nlpspQuadratureId           = nlpspQuadratureId;
    d_numEigenValues              = numEigenValues;

    createAtomCenteredSphericalFunctionsForOrbitals();

    d_atomicOrbitalFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicOrbitalFnsContainer->init(atomicNumbers, d_atomicOrbitalFnsMap);

    if (!d_useDevice)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperHostPtr,
            d_BasisOperatorHostPtr,
            d_atomicOrbitalFnsContainer,
            d_mpiCommDomain);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperDevicePtr,
            d_BasisOperatorDevicePtr,
            d_atomicOrbitalFnsContainer,
            d_mpiCommDomain);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double> &             kPointWeights,
      const std::vector<double> &             kPointCoordinates,
      const bool                              updateNonlocalSparsity)
  {
    std::vector<unsigned int> atomicNumbers;
    std::vector<double>       atomCoords;

    for (int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }

    d_atomicOrbitalFnsContainer->initaliseCoordinates(atomCoords,
                                                      periodicCoords,
                                                      imageIds);

    if (updateNonlocalSparsity)
      {
        MPI_Barrier(d_mpiCommDomain);
        d_atomicOrbitalFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        MPI_Barrier(d_mpiCommDomain);
      }

    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_nlpspQuadratureId);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  atomCenteredOrbitalsPostProcessing<ValueType,
                                     memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    createAtomCenteredSphericalFunctionsForOrbitals()
  {
    for (std::set<unsigned int>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         it++)
      {
        char pseudoAtomDataFile[256];
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());
        unsigned int  Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        std::set<std::pair<int, int>> radFunctionIds;
        std::string                   readLine;
        char                          waveFunctionFileName[512];

        nlNumsMap[Znum].clear();

        if (!readPseudoDataFileNames.is_open())
          {
            std::cerr << "Failed to open file: " << pseudoAtomDataFile
                      << std::endl;
            exit(-1);
          }

        while (std::getline(readPseudoDataFileNames, readLine))
          {
            if (readLine.find("psi") != std::string::npos)
              {
                std::string nlPart = readLine.substr(3, readLine.size() - 7);
                int nQuantumNumber = std::atoi(nlPart.substr(0, 1).c_str());
                int lQuantumNumber = std::atoi(nlPart.substr(1, 1).c_str());
                radFunctionIds.insert(
                  std::make_pair(nQuantumNumber, lQuantumNumber));
              }
          }
        std::vector<std::pair<int, int>> tempVec(radFunctionIds.begin(),
                                                 radFunctionIds.end());

        // std::sort(tempVec.begin(),
        //           tempVec.end(),
        //           [](const std::pair<int, int> &a,
        //              const std::pair<int, int> &b) {
        //             if (a.second == b.second)
        //               {
        //                 return a.first < b.first;
        //               }
        //             return a.second < b.second;
        //           });

        readPseudoDataFileNames.close();
        unsigned int alpha = 0;
        for (std::vector<std::pair<int, int>>::iterator i = tempVec.begin();
             i != tempVec.end();
             ++i)
          {
            int nQuantumNumber = i->first;
            int lQuantumNumber = i->second;
            strcpy(waveFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/psi" + std::to_string(nQuantumNumber) +
                    std::to_string(lQuantumNumber) + ".inp")
                     .c_str());

            nlNumsMap[Znum].push_back(
              std::make_pair(nQuantumNumber, lQuantumNumber));

            d_atomicOrbitalFnsMap[std::make_pair(Znum, alpha)] =
              std::make_shared<AtomCenteredSphericalFunctionProjectorSpline>(
                waveFunctionFileName,
                lQuantumNumber,
                -1, // radial power
                1,
                2,
                1E-12);
            alpha++;
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  atomCenteredOrbitalsPostProcessing<ValueType, memorySpace>::
    computeAtomCenteredEntries(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
      const unsigned int                      totalNumWaveFunctions,
      const std::vector<std::vector<double>> &eigenValues,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        &basisOperationsPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                 BLASWrapperPtrHost,
      const unsigned int         quadratureIndex,
      const std::vector<double> &kPointWeights,
      const MPI_Comm &           interBandGroupComm,
      const MPI_Comm &           interpoolComm,
      const dftParameters *      dftParamsPtr,
      double                     fermiEnergy,
      unsigned int               highestStateNscfSolve)
  {
    computing_timer.enter_subsection("PDOS computation");
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    unsigned int       numSpinComponents;
    numSpinComponents               = dftParamsPtr->spinPolarized + 1;
    const unsigned int numLocalDofs = basisOperationsPtr->nOwnedDofs();

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ?
        totalLocallyOwnedCells :
        1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    const unsigned int numNodesPerElement = basisOperationsPtr->nDofsPerCell();

    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    const unsigned int        bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    unsigned int BVec = std::min(dftParamsPtr->chebyWfcBlockSize,
                                 bandGroupLowHighPlusOneIndices[1]);

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlock;

    unsigned int previousSize = 0;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> resVec;

    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;

    std::vector<double> eigenValuesBlock;
    std::vector<double> smearedComponentBlock;

    std::vector<double> eigenValuesAllkPoints;

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter <= highestStateNscfSolve;
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(eigenValues[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    const double totalEigenValues = eigenValuesAllkPoints.size();
    const double intervalSize =
      dftParamsPtr->intervalSize / C_haToeV; // eV to Ha

    double lowerBoundEpsilon = std::floor(eigenValuesAllkPoints[0] * 100) / 100;
    double upperBoundEpsilon =
      std::ceil(eigenValuesAllkPoints[totalEigenValues - 1] * 100) / 100;

    MPI_Allreduce(MPI_IN_PLACE,
                  &lowerBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&lowerBoundEpsilon),
                  MPI_MIN,
                  interpoolComm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &upperBoundEpsilon,
                  1,
                  dataTypes::mpi_type_id(&upperBoundEpsilon),
                  MPI_MAX,
                  interpoolComm);

    lowerBoundEpsilon =
      lowerBoundEpsilon - 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);
    upperBoundEpsilon =
      upperBoundEpsilon + 0.1 * (upperBoundEpsilon - lowerBoundEpsilon);

    const unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);

    std::vector<unsigned int> atomicNumbers =
      d_atomicOrbitalFnsContainer->getAtomicNumbers();

    // spin,kpoint,atomId, beta x numWfc vector
    std::vector<std::vector<std::map<unsigned int, std::vector<double>>>>
      pdosKernelWithoutSmearFunction;
    pdosKernelWithoutSmearFunction.resize(numSpinComponents);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        pdosKernelWithoutSmearFunction[spinIndex].resize(kPointWeights.size());
      }

    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); kPoint++)
          {
            d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);

            std::map<unsigned int, std::vector<double>>
              &pdosKernelWithoutSmearFunctionSpinKpoint =
                pdosKernelWithoutSmearFunction[spinIndex][kPoint];

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);

                flattenedArrayBlock =
                  &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

                // this opertaion sets d_numberWaveFunctions =
                // currentBlockSize
                // size of resVec =
                // currentBlockSize*(number of atomic Orbitals for each
                // atom) summed over all atoms that have compact support on
                // the elements in the processor

                d_nonLocalOperator->initialiseFlattenedDataStructure(
                  currentBlockSize, resVec);

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * totalNumWaveFunctions *
                                        (numSpinComponents * kPoint +
                                         spinIndex) +
                                      iNode * totalNumWaveFunctions + jvec,
                                    currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      BLASWrapperPtrDevice->stridedCopyToBlockConstantStride(
                        currentBlockSize,
                        totalNumWaveFunctions,
                        numLocalDofs,
                        jvec,
                        X->data() + numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex),
                        flattenedArrayBlock->data());
#endif
                    basisOperationsPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                               quadratureIndex,
                                               false);

                    flattenedArrayBlock->updateGhostValues();
                    basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;

                            if (currentCellsBlockSize * currentBlockSize !=
                                previousSize)
                              {
                                tempCellNodalData.resize(currentCellsBlockSize *
                                                         currentBlockSize *
                                                         numNodesPerElement);
                                if constexpr (memorySpace ==
                                              dftfe::utils::MemorySpace::DEVICE)
                                  {
                                    d_nonLocalOperator
                                      ->initialiseCellWaveFunctionPointers(
                                        tempCellNodalData);
                                  }
                                previousSize =
                                  currentCellsBlockSize * currentBlockSize;
                              }

                            basisOperationsPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));
                            d_nonLocalOperator->applyCconjtransOnX(
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));
                          }
                      } // iblock
                    d_nonLocalOperator->applyAllReduceOnCconjtransX(resVec);

                    dftfe::utils::MemoryStorage<double, memorySpace> scalingVec;
                    scalingVec.resize(currentBlockSize, 1.0);
                    d_nonLocalOperator
                      ->copyBackFromDistributedVectorToLocalDataStructure(
                        resVec, scalingVec);


                    const std::vector<unsigned int> atomIdsInProcessor =
                      d_atomicOrbitalFnsContainer->getAtomIdsInCurrentProcess();


                    for (unsigned int iAtom = 0;
                         iAtom <
                         d_nonLocalOperator->getTotalAtomInCurrentProcessor();
                         iAtom++)
                      {
                        unsigned int atomId =
                          atomIdsInProcessor[iAtom]; // globa Id
                        unsigned int Znum = atomicNumbers[atomId];
                        unsigned int numberSphericalFunctions =
                          d_atomicOrbitalFnsContainer
                            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
                        // size beta x currentblocksize (row major)
                        std::vector<ValueType> tempVec;
                        tempVec.resize(numberSphericalFunctions *
                                       currentBlockSize);

                        std::memcpy(tempVec.data(),
                                    d_nonLocalOperator
                                      ->getCconjtansXLocalDataStructure(iAtom),
                                    numberSphericalFunctions *
                                      currentBlockSize * sizeof(ValueType));

                        if (jvec == 0)
                          {
                            pdosKernelWithoutSmearFunctionSpinKpoint[atomId]
                              .resize(numberSphericalFunctions *
                                      totalNumWaveFunctions);
                          }


                        // contains for every atomId a matrix of size beta_a x
                        // currentblocksize (absolute value squared value)
                        std::map<unsigned int, std::vector<double>>
                          extractedAtomicMapSquaredBlock;
                        extractedAtomicMapSquaredBlock[atomId].clear();
                        extractedAtomicMapSquaredBlock[atomId].resize(
                          tempVec.size());
                        std::transform(
                          tempVec.begin(),
                          tempVec.end(),
                          extractedAtomicMapSquaredBlock[atomId].begin(),
                          [](const auto &a) {
                            return std::abs(a) * std::abs(a);
                          });

                        std::memcpy(
                          &pdosKernelWithoutSmearFunctionSpinKpoint
                            [atomId][jvec * numberSphericalFunctions],
                          &(extractedAtomicMapSquaredBlock[atomId][0]),
                          extractedAtomicMapSquaredBlock[atomId].size() *
                            sizeof(double));
                      }
                  } // current bandgroup check
              }     // jVec
          }         // kpoint
      }             // spinIndex

    // Till now all the data has been collected

    std::vector<double> smearedValues;
    // spin, (atomId, beta x numEnergies)
    // summed over wfc blocks and kpoints
    std::vector<std::map<unsigned int, std::vector<double>>> summedOverBlocks;
    summedOverBlocks.resize(numSpinComponents);

    std::vector<double> numTotalAtomicOrbitals;
    std::vector<double> cumulativeNumAtomicOrbitals;

    numTotalAtomicOrbitals.resize(dftParamsPtr->natoms, 0.0);
    cumulativeNumAtomicOrbitals.resize(dftParamsPtr->natoms, 0.0);

    for (auto pair : pdosKernelWithoutSmearFunction[0][0])
      {
        unsigned     atomId = pair.first;
        unsigned int Znum   = atomicNumbers[atomId];
        unsigned int numberSphericalFunctions =
          d_atomicOrbitalFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        // two different processes can have the same atomIDs.
        numTotalAtomicOrbitals[atomId] = numberSphericalFunctions;
      }

    // Do an MPI_allreduce to get the atomIDs in one vector;
    // MAX operation ensures that only one of the identical non-zero element is
    // taken
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTotalAtomicOrbitals[0],
                  dftParamsPtr->natoms,
                  dataTypes::mpi_type_id(&numTotalAtomicOrbitals[0]),
                  MPI_MAX,
                  d_mpiCommDomain);

    for (unsigned int i = 0; i < dftParamsPtr->natoms; i++)
      {
        if (i == 0)
          {
            cumulativeNumAtomicOrbitals[i] = numTotalAtomicOrbitals[i];
          }
        else
          {
            cumulativeNumAtomicOrbitals[i] =
              numTotalAtomicOrbitals[i] + cumulativeNumAtomicOrbitals[i - 1];
          }
      }

    std::vector<double> mpiVectorSummedOverBlocks;

    mpiVectorSummedOverBlocks.resize(
      numSpinComponents *
        cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] * numberIntervals,
      0.0);

    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; spinIndex++)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); kPoint++)
          {
            const std::map<unsigned int, std::vector<double>>
              &spinKpointKernel =
                pdosKernelWithoutSmearFunction[spinIndex][kPoint];

            for (unsigned int epsInt = 0; epsInt < numberIntervals; epsInt++)
              {
                smearedValues.clear();
                double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                for (unsigned int iEigenVec = 0;
                     iEigenVec < totalNumWaveFunctions;
                     iEigenVec++)
                  {
                    double eigenValue =
                      eigenValues[kPoint][totalNumWaveFunctions * spinIndex +
                                          iEigenVec];
                    if (numSpinComponents == 2)
                      {
                        if (iEigenVec > highestStateNscfSolve)
                          {
                            smearedValues.push_back(0.0);
                          }
                        else
                          {
                            smearedValues.push_back(
                              (smearFunction(epsValue - eigenValue,
                                             dftParamsPtr)));
                          }
                      }
                    else
                      {
                        if (iEigenVec > highestStateNscfSolve)
                          {
                            smearedValues.push_back(0.0);
                          }
                        else
                          {
                            smearedValues.push_back(
                              (2 * smearFunction(epsValue - eigenValue,
                                                 dftParamsPtr)));
                          }
                      }
                  }
                for (auto &pair : spinKpointKernel)
                  {
                    unsigned int        atomId      = pair.first;
                    std::vector<double> kernelValue = pair.second;
                    unsigned int        Znum        = atomicNumbers[atomId];
                    unsigned int        numberSphericalFunctions =
                      d_atomicOrbitalFnsContainer
                        ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);

                    if (epsInt == 0)
                      {
                        summedOverBlocks[spinIndex][atomId].resize(
                          numberIntervals * numberSphericalFunctions);
                      }

                    const double zero(0.0), one(1.0);

                    for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                         jvec += BVec)
                      {
                        const unsigned int currentBlockSize =
                          std::min(BVec, totalNumWaveFunctions - jvec);

                        // In summedOverBlocks, summation of kpoints is done
                        BLASWrapperPtrHost->xgemm(
                          'N',
                          'N',
                          1,
                          numberSphericalFunctions,
                          currentBlockSize,
                          &kPointWeights[kPoint],
                          &smearedValues[jvec],
                          1,
                          &kernelValue[jvec * numberSphericalFunctions],
                          currentBlockSize,
                          &one,
                          &(summedOverBlocks[spinIndex][atomId]
                                            [epsInt *
                                             numberSphericalFunctions]),
                          1);
                      }
                  } // atomID
              }     // epsInt

          } // kpoint

        for (auto pair : pdosKernelWithoutSmearFunction[spinIndex][0])
          {
            unsigned int atomId = pair.first;

            MPI_Allreduce(MPI_IN_PLACE,
                          &summedOverBlocks[spinIndex][atomId][0],
                          summedOverBlocks[spinIndex][atomId].size(),
                          dataTypes::mpi_type_id(
                            &summedOverBlocks[spinIndex][atomId][0]),
                          MPI_SUM,
                          interpoolComm);

            if (atomId != 0)
              {
                std::memcpy(
                  &mpiVectorSummedOverBlocks
                    [spinIndex *
                       cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                       numberIntervals +
                     cumulativeNumAtomicOrbitals[atomId - 1] * numberIntervals],
                  &summedOverBlocks[spinIndex][atomId][0],
                  summedOverBlocks[spinIndex][atomId].size() * sizeof(double));
              }
            else
              {
                std::memcpy(
                  &mpiVectorSummedOverBlocks
                    [spinIndex *
                     cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                     numberIntervals],
                  &summedOverBlocks[spinIndex][atomId][0],
                  summedOverBlocks[spinIndex][atomId].size() * sizeof(double));
              }
          }

      } // spinIndex

    MPI_Allreduce(MPI_IN_PLACE,
                  &mpiVectorSummedOverBlocks[0],
                  cumulativeNumAtomicOrbitals[dftParamsPtr->natoms - 1] *
                    numberIntervals,
                  dataTypes::mpi_type_id(&mpiVectorSummedOverBlocks[0]),
                  MPI_MAX,
                  d_mpiCommDomain);

    if (dealii::Utilities::MPI::this_mpi_process(
          d_mpiCommParentPostProcessing) == 0)
      {
        LQnumToNameMap = {{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"}};
        double epsValue;
        double pdosSumUp;
        double pdosSumDown;

        if (dftParamsPtr->reproducible_output && dftParamsPtr->verbosity == 0)
          {
            pcout << "Writing PDOS outputs..." << std::endl;
          }

        for (unsigned int atomId = 0; atomId < dftParamsPtr->natoms; atomId++)
          {
            unsigned int              Znum = atomicNumbers[atomId];
            std::vector<unsigned int> nQuantumNums;
            std::vector<unsigned int> lQuantumNums;

            unsigned int wfc_number    = 0;
            unsigned int wfcStartIndex = 0;
            for (auto i : nlNumsMap[Znum])
              {
                unsigned int       nQuantumNum = i.first;
                unsigned int       lQuantumNum = i.second;
                std::ostringstream pdosFileName;

                pdosFileName << "pdosData_atom#" << atomId << "_wfc#"
                             << wfc_number << "(" << nQuantumNum
                             << LQnumToNameMap[lQuantumNum] << ")"
                             << ".out";
                if (dftParamsPtr->reproducible_output &&
                    dftParamsPtr->verbosity == 0 && atomId == 0)
                  {
                    pcout << "File: "
                          << "pdosData_atom#" << atomId << "_wfc#" << wfc_number
                          << "(" << nQuantumNum << LQnumToNameMap[lQuantumNum]
                          << ")"
                          << ".out" << std::endl;
                  }
                std::ofstream outFile(pdosFileName.str());
                outFile.setf(std::ios_base::fixed);
                outFile << std::setprecision(18);
                wfc_number += 1;
                if (outFile.is_open())
                  {
                    if (numSpinComponents == 1)
                      {
                        outFile << std::setw(15) << "# E(eV)" << std::setw(15)
                                << "sumPDOS";
                        if (lQuantumNum == 0)
                          {
                            outFile << std::setw(15) << "s" << std::endl;
                          }
                        else if (lQuantumNum == 1)
                          {
                            outFile << std::setw(15) << "py" << std::setw(15)
                                    << "pz" << std::setw(15) << "px"
                                    << std::endl;
                          }

                        else if (lQuantumNum == 2)
                          {
                            outFile << std::setw(15) << "dxy" << std::setw(15)
                                    << "dyz" << std::setw(15) << "dz2"
                                    << std::setw(15) << "dxz" << std::setw(15)
                                    << "dx2-y2" << std::endl;
                          }

                        else if (lQuantumNum == 3)
                          {
                            outFile << std::setw(15) << "fy(3x2-y2)"
                                    << std::setw(15) << "fxyz" << std::setw(15)
                                    << "fyz2" << std::setw(15) << "fz3"
                                    << std::setw(15) << "fxz2" << std::setw(15)
                                    << "fz(x2-y2)" << std::setw(15)
                                    << "fx(x2-3y2)" << std::endl;
                          }

                        if (dftParamsPtr->reproducible_output &&
                            dftParamsPtr->verbosity == 0 && atomId == 0)
                          {
                            pcout << std::setw(15) << "# E(eV)" << std::setw(15)
                                  << "sumPDOS";
                            if (lQuantumNum == 0)
                              {
                                pcout << std::setw(15) << "s" << std::endl;
                              }
                            else if (lQuantumNum == 1)
                              {
                                pcout << std::setw(15) << "py" << std::setw(15)
                                      << "pz" << std::setw(15) << "px"
                                      << std::endl;
                              }

                            else if (lQuantumNum == 2)
                              {
                                pcout << std::setw(15) << "dxy" << std::setw(15)
                                      << "dyz" << std::setw(15) << "dz2"
                                      << std::setw(15) << "dxz" << std::setw(15)
                                      << "dx2-y2" << std::endl;
                              }

                            else if (lQuantumNum == 3)
                              {
                                pcout << std::setw(15) << "fy(3x2-y2)"
                                      << std::setw(15) << "fxyz"
                                      << std::setw(15) << "fyz2"
                                      << std::setw(15) << "fz3" << std::setw(15)
                                      << "fxz2" << std::setw(15) << "fz(x2-y2)"
                                      << std::setw(15) << "fx(x2-3y2)"
                                      << std::endl;
                              }
                          }
                      }
                    else
                      {
                        outFile << std::setw(15) << "# E(eV)" << std::setw(15)
                                << "sumPDOS_up" << std::setw(15)
                                << "sumPDOS_down";

                        if (lQuantumNum == 0)
                          {
                            outFile << std::setw(15) << "s_up" << std::setw(15)
                                    << "s_down" << std::endl;
                          }
                        else if (lQuantumNum == 1)
                          {
                            outFile << std::setw(15) << "py_up" << std::setw(15)
                                    << "py_down" << std::setw(15) << "pz_up"
                                    << std::setw(15) << "pz_down"
                                    << std::setw(15) << "px_up" << std::setw(15)
                                    << "px_down" << std::endl;
                          }

                        else if (lQuantumNum == 2)
                          {
                            outFile
                              << std::setw(15) << "dxy_up" << std::setw(15)
                              << "dxy_down" << std::setw(15) << "dyz_up"
                              << std::setw(15) << "dyz_down" << std::setw(15)
                              << "dz2_up" << std::setw(15) << "dz2_down"
                              << std::setw(15) << "dxz_up" << std::setw(15)
                              << "dxz_down" << std::setw(15) << "dx2-y2_up"
                              << std::setw(15) << "dx2-y2_down" << std::endl;
                          }

                        else if (lQuantumNum == 3)
                          {
                            outFile
                              << std::setw(15) << "fy(3x2-y2)_up"
                              << std::setw(15) << "fy(3x2-y2)_down"
                              << std::setw(15) << "fxyz_up" << std::setw(15)
                              << "fxz_down" << std::setw(15) << "fyz2_up"
                              << std::setw(15) << "fyz2_down" << std::setw(15)
                              << "fz3_up" << std::setw(15) << "fz3_down"
                              << std::setw(15) << "fxz2_up" << std::setw(15)
                              << "fxz2_down" << std::setw(15) << "fz(x2-y2)_up"
                              << std::setw(15) << "fz(x2-y2)_down"
                              << std::setw(15) << "fx(x2-3y2)_up"
                              << std::setw(15) << "fx(x2-3y2)_down"
                              << std::endl;
                          }

                        if (dftParamsPtr->reproducible_output &&
                            dftParamsPtr->verbosity == 0 && atomId == 0)
                          {
                            pcout << std::setw(15) << "# E(eV)" << std::setw(15)
                                  << "sumPDOS_up" << std::setw(15)
                                  << "sumPDOS_down";
                            if (lQuantumNum == 0)
                              {
                                pcout << std::setw(15) << "s_up"
                                      << std::setw(15) << "s_down" << std::endl;
                              }
                            else if (lQuantumNum == 1)
                              {
                                pcout
                                  << std::setw(15) << "py_up" << std::setw(15)
                                  << "py_down" << std::setw(15) << "pz_up"
                                  << std::setw(15) << "pz_down" << std::setw(15)
                                  << "px_up" << std::setw(15) << "px_down"
                                  << std::endl;
                              }

                            else if (lQuantumNum == 2)
                              {
                                pcout
                                  << std::setw(15) << "dxy_up" << std::setw(15)
                                  << "dxy_down" << std::setw(15) << "dyz_up"
                                  << std::setw(15) << "dyz_down"
                                  << std::setw(15) << "dz2_up" << std::setw(15)
                                  << "dz2_down" << std::setw(15) << "dxz_up"
                                  << std::setw(15) << "dxz_down"
                                  << std::setw(15) << "dx2-y2_up"
                                  << std::setw(15) << "dx2-y2_down"
                                  << std::endl;
                              }

                            else if (lQuantumNum == 3)
                              {
                                pcout
                                  << std::setw(15) << "fy(3x2-y2)_up"
                                  << std::setw(15) << "fy(3x2-y2)_down"
                                  << std::setw(15) << "fxyz_up" << std::setw(15)
                                  << "fxz_down" << std::setw(15) << "fyz2_up"
                                  << std::setw(15) << "fyz2_down"
                                  << std::setw(15) << "fz3_up" << std::setw(15)
                                  << "fz3_down" << std::setw(15) << "fxz2_up"
                                  << std::setw(15) << "fxz2_down"
                                  << std::setw(15) << "fz(x2-y2)_up"
                                  << std::setw(15) << "fz(x2-y2)_down"
                                  << std::setw(15) << "fx(x2-3y2)_up"
                                  << std::setw(15) << "fx(x2-3y2)_down"
                                  << std::endl;
                              }
                          }
                      }
                    for (unsigned int epsInt = 0; epsInt < numberIntervals;
                         ++epsInt)
                      {
                        pdosSumUp   = 0.0;
                        pdosSumDown = 0.0;
                        epsValue    = lowerBoundEpsilon + epsInt * intervalSize;

                        outFile << std::setw(15) << epsValue * C_haToeV << "\t";

                        if (dftParamsPtr->reproducible_output &&
                            dftParamsPtr->verbosity == 0 && atomId == 0)
                          {
                            double epsValueTrunc =
                              std::floor(100000 * epsValue * C_haToeV) / 100000;
                            pcout << std::fixed << std::setprecision(5)
                                  << std::setw(15) << epsValueTrunc << "\t";
                          }

                        std::vector<double> pdosVec;

                        for (int spinIndex = 0; spinIndex < numSpinComponents;
                             spinIndex++)
                          {
                            std::vector<double>::iterator startIt;
                            if (atomId != 0)
                              startIt =
                                mpiVectorSummedOverBlocks.begin() +
                                spinIndex *
                                  cumulativeNumAtomicOrbitals
                                    [dftParamsPtr->natoms - 1] *
                                  numberIntervals +
                                cumulativeNumAtomicOrbitals[atomId - 1] *
                                  numberIntervals +
                                epsInt * numTotalAtomicOrbitals[atomId];

                            else
                              startIt = mpiVectorSummedOverBlocks.begin() +
                                        spinIndex *
                                          cumulativeNumAtomicOrbitals
                                            [dftParamsPtr->natoms - 1] *
                                          numberIntervals +
                                        epsInt * numTotalAtomicOrbitals[atomId];

                            auto endIt =
                              startIt + numTotalAtomicOrbitals[atomId];

                            for (int j = wfcStartIndex;
                                 j < wfcStartIndex + 2 * lQuantumNum + 1;
                                 j++)
                              {
                                pdosVec.push_back(*(startIt + j));
                                if (spinIndex == 0)
                                  pdosSumUp += *(startIt + j);
                                else if (spinIndex == 1)
                                  pdosSumDown += *(startIt + j);
                              }
                          }

                        if (numSpinComponents == 1)
                          {
                            outFile << std::setw(15) << pdosSumUp << "\t";
                            for (auto it = pdosVec.begin(); it != pdosVec.end();
                                 ++it)
                              {
                                outFile << std::setw(15) << *it << "\t";
                              }
                            outFile << std::endl;

                            if (dftParamsPtr->reproducible_output &&
                                dftParamsPtr->verbosity == 0 && atomId == 0)
                              {
                                pcout << std::setw(15)
                                      << std::floor(100000 * pdosSumUp) / 100000
                                      << "\t";
                                for (auto it = pdosVec.begin();
                                     it != pdosVec.end();
                                     ++it)
                                  {
                                    pcout << std::setw(15)
                                          << std::floor((*it) * 100000) / 100000
                                          << "\t";
                                  }
                                pcout << std::endl;
                              }
                          }
                        else
                          {
                            outFile << std::setw(15) << pdosSumUp << "\t";
                            outFile << std::setw(15) << pdosSumDown << "\t";

                            for (auto it = pdosVec.begin();
                                 it != pdosVec.begin() + pdosVec.size() / 2;
                                 ++it)
                              {
                                outFile << std::setw(15) << *it << "\t"
                                        << std::setw(15)
                                        << *(it + pdosVec.size() / 2) << "\t";
                              }
                            outFile << std::endl;

                            if (dftParamsPtr->reproducible_output &&
                                dftParamsPtr->verbosity == 0 && atomId == 0)
                              {
                                pcout << std::setw(15)
                                      << std::floor(pdosSumUp * 100000) / 100000
                                      << "\t";
                                pcout
                                  << std::setw(15)
                                  << std::floor(pdosSumDown * 100000) / 100000
                                  << "\t";

                                for (auto it = pdosVec.begin();
                                     it != pdosVec.begin() + pdosVec.size() / 2;
                                     ++it)
                                  {
                                    pcout << std::setw(15) << *it << "\t"
                                          << std::setw(15)
                                          << std::floor(
                                               (*(it + pdosVec.size() / 2)) *
                                               100000) /
                                               100000
                                          << "\t";
                                  }
                                pcout << std::endl;
                              }
                          }
                      }
                  }
                wfcStartIndex = wfcStartIndex + 2 * lQuantumNum + 1;
              }
          }
      }
    computing_timer.leave_subsection("PDOS computation");
  }

  template class atomCenteredOrbitalsPostProcessing<
    dataTypes::number,
    dftfe::utils::MemorySpace::HOST>;

#if defined(DFTFE_WITH_DEVICE)
  template class atomCenteredOrbitalsPostProcessing<
    dataTypes::number,
    dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
