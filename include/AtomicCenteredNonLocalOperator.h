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
// @author  Kartick Ramakrishnan, Vishal Subramanian, Sambit Das
//

#ifndef DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
#define DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
#include <MultiVector.h>
#include <headers.h>
#include <AtomCenteredSphericalFunctionContainer.h>
#include <sphericalHarmonicUtils.h>
#include <BLASWrapper.h>
#include <memory>
#include <MemorySpaceType.h>
#include "FEBasisOperations.h"
#include <headers.h>
#include <dftUtils.h>
#include <pseudoUtils.h>
#include <vectorUtilities.h>
#include <MPIPatternP2P.h>
#include <MultiVector.h>
#include <DeviceTypeConfig.h>
#include <cmath>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  /**
   * @brief Enum class that lists
   * used in the non-local Operator
   *
   */
  enum class CouplingStructure
  {
    diagonal,
    dense,
    blockDiagonal
  };



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class AtomicCenteredNonLocalOperator
  {
  public:
    AtomicCenteredNonLocalOperator(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperatorPtr,
      std::shared_ptr<AtomCenteredSphericalFunctionContainer>
                      atomCenteredSphericalFunctionContainer,
      const MPI_Comm &mpi_comm_parent,
      const bool      memOptMode = false);

    /**
     * @brief Resizes various internal data members and selects the kpoint of interest.
     * @param[in] kPointIndex specifies the k-point of interest
     */
    void
    initialiseOperatorActionOnX(unsigned int kPointIndex);
    /**
     * @brief initialises the multivector object, waveFunctionBlockSize and resizes various internal data members.
     * @param[in] waveFunctionBlockSize sets the wavefunction block size for the
     * action of the nonlocal operator.
     * @param[out] sphericalFunctionKetTimesVectorParFlattened, the multivector
     * that is initialised based on blocksize and partitioner.
     */
    void
    initialiseFlattenedDataStructure(
      unsigned int waveFunctionBlockSize,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened);
    /**
     * @brief calls internal function: initialisePartitioner, initialiseKpoint and computeCMatrixEntries
     * @param[in] updateSparsity flag on whether the sparstiy patten was
     * updated, hence the partitioner is updated.
     * @param[in] kPointWeights std::vector<double> of size number of kPoints
     * @param[out] kPointCoordinates std::vector<double> of kPoint coordinates
     * @param[in] basisOperationsPtr HOST FEBasisOperations shared_ptr required
     * to indetify the element ids and quad points
     * @param[in] quadratureIndex quadrature index for sampling the spherical
     * function. Quadrature Index is used to reinit basisOperationsPtr
     */
    void
    intitialisePartitionerKPointsAndComputeCMatrixEntries(
      const bool                 updateSparsity,
      const std::vector<double> &kPointWeights,
      const std::vector<double> &kPointCoordinates,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
                         basisOperationsPtr,
      const unsigned int quadratureIndex);
#if defined(DFTFE_WITH_DEVICE)
    // for device specific initialise
    /**
     * @brief
     * @param[in] totalAtomsInCurrentProcessor number of atoms in current
     * processor based on compact support
     * @param[out] totalNonLocalElements number of nonLocal elements in current
     * processor
     * @param[out] numberCellsForEachAtom number of cells associated which each
     * atom in the current processor. vecot of size totalAtomsInCurrentProcessor
     * @param[out] numberCellsAccumNonLocalAtoms number of cells accumulated
     * till iatom in current processor. vector of size
     * totalAtomsInCurrentProcessor
     */
    void
    initialiseCellWaveFunctionPointers(
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrix);

    void
    freeDeviceVectors();
#endif

    // Getter functions
    // Returns the vector that takes in nonlocalElementIndex and returns the
    // cellID
    const std::vector<unsigned int> &
    getNonlocalElementToCellIdVector() const;
    // Returns the number of atoms in current processor
    unsigned int
    getTotalAtomInCurrentProcessor() const;

    const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &
    getFlattenedNonLocalCellDofIndexToProcessDofIndexMap() const;

    unsigned int
    getTotalNonLocalElementsInCurrentProcessor() const;

    unsigned int
    getTotalNonLocalEntriesCurrentProcessor() const;

    unsigned int
    getMaxSingleAtomEntries() const;

    bool
    atomSupportInElement(unsigned int iElem) const;

    unsigned int
    getGlobalDofAtomIdSphericalFnPair(const unsigned int atomId,
                                      const unsigned int alpha) const;

    unsigned int
    getLocalIdOfDistributedVec(const unsigned int globalId) const;

    std::vector<unsigned int> &
    getNonLocalElemIdToLocalElemIdMap() const;

    std::vector<unsigned int> &
    getAtomWiseNumberCellsInCompactSupport() const;

    std::vector<unsigned int> &
    getAtomWiseNumberCellsAccumulated() const;

    const std::vector<ValueType> &
    getAtomCenteredKpointIndexedSphericalFnQuadValues() const;

    const std::vector<ValueType> &
    getAtomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues() const;

    const std::map<unsigned int, std::vector<unsigned int>> &
    getCellIdToAtomIdsLocalCompactSupportMap() const;

    const std::vector<unsigned int> &
    getNonTrivialSphericalFnsPerCell() const;

    const std::vector<unsigned int> &
    getNonTrivialSphericalFnsCellStartIndex() const;

    const unsigned int
    getTotalNonTrivialSphericalFnsOverAllCells() const;


    const std::vector<unsigned int> &
    getNonTrivialAllCellsSphericalFnAlphaToElemIdMap() const;

    /**
     * @brief Required in configurational forces. Cummulative sphercial Fn Id. The size is numCells in processor
     */
    const std::map<unsigned int, std::vector<unsigned int>> &
    getAtomIdToNonTrivialSphericalFnCellStartIndex() const;

    /**
     * @brief Returns the Flattened vector of sphericalFunctionIDs in order of atomIDs of atoms in processor.
     */
    const std::vector<unsigned int> &
    getSphericalFnTimesVectorFlattenedVectorLocalIds() const;

    const std::vector<unsigned int> &
    getOwnedAtomIdsInCurrentProcessor() const;

    void
    computeCconjtransCMatrix(
      const unsigned int atomId,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtr,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &Dinverse,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        PconjtransposePmatrix);
    // Calls for both device and host
    /**
     * @brief compute the action of coupling matrix on sphericalFunctionKetTimesVectorParFlattened.
     * @param[in] couplingtype structure of coupling matrix
     * @param[in] couplingMatrix entires of the coupling matrix V in
     * CVCconjtrans. Ensure that the coupling matrix is padded. Refer to
     * ONCVclass for template
     * @param[out] sphericalFunctionKetTimesVectorParFlattened multivector to
     * store results of CconjtransX which is initiliased using
     * initialiseFlattenedVector call. The results are stored in
     * sphericalFunctionKetTimesVectorParFlattened or internal data member based
     * on flagCopyResultsToMatrix.
     * @param[in] flagCopyResultsToMatrix flag to confirm whether to scal the
     * multivector sphericalFunctionKetTimesVectorParFlattened or store results
     * in internal data member.
     */
    void
    applyVOnCconjtransX(
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &                sphericalFunctionKetTimesVectorParFlattened,
      const bool         flagCopyResultsToMatrix = true,
      const unsigned int kPointIndex             = 0);

    /**
     * @brief After AllReduce function is called this will copy to the nonLocalOperatorClassDatastructure.
     */
    void
    copyBackFromDistributedVectorToLocalDataStructure(
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      const dftfe::utils::MemoryStorage<double, memorySpace> &scalingVector);
    /**
     * @brief copies the results from internal member to sphericalFunctionKetTimesVectorParFlattened, on which ghost values are called.
     * crucial operation for completion of the full CconjtranX on all cells
     * @param[in] sphericalFunctionKetTimesVectorParFlattened multivector to
     * store results of CconjtransX which is initiliased using
     * initialiseFlattenedVector call
     * @param[in] skip1 flag for compute-communication overlap in ChFSI on GPUs
     * @param[in] skip2 flag for compute-communication overlap in ChFSI on GPUs
     */
    void
    applyAllReduceOnCconjtransX(
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool skipComm = false);

    /**
     * @brief computes the results of CconjtransX on the cells of interst specied by cellRange
     * @param[in] X inpute cell level vector
     * @param[in] cellRange start and end element id in list of nonlocal
     * elements
     */
    void
    applyCconjtransOnX(const ValueType *                           X,
                       const std::pair<unsigned int, unsigned int> cellRange);


    // Returns the pointer of CTX stored in HOST memory for the atom Index in
    // the list of atoms with support in the processor.
    /**
     * @brief Returns the pointer of CTX stored in HOST memory
     * @param[in] iAtom atomIndex in the list of atoms with support in the
     * current processor. NOTE!! One must be careful here
     */
    const ValueType *
    getCconjtansXLocalDataStructure(const unsigned int iAtom);
    /**
     * @brief completes the VCconjX on nodal vector src. The src vector must have all ghost nodes and contraint nodes updated.
     * @param[in] src input nodal vector on which operator acts on.
     * @param[in] kPointIndex kPoint of interst for current operation
     * @param[in] couplingtype structure of coupling matrix
     * @param[in] couplingMatrix entires of the coupling matrix V in
     * CVCconjtrans. Ensure the coupling matrix is padded
     * @param[out] sphericalFunctionKetTimesVectorParFlattened multivector to
     * store results of CconjtransX which is initiliased using
     * initialiseFlattenedVector call
     */
    void
    applyVCconjtransOnX(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      const unsigned int                                         kPointIndex,
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &        sphericalFunctionKetTimesVectorParFlattened,
      const bool flagScaleInternalMatrix = false);

    /**
     * @brief completes the action of CVCconjtranspose on nodal vector src. The src vector must have all ghost nodes and contraint nodes updated.
     * @param[in] src input nodal vector on which operator acts on.
     * @param[in] kPointIndex kPoint of interst for current operation
     * @param[in] couplingtype structure of coupling matrix
     * @param[in] couplingMatrix entires of the coupling matrix V in
     * CVCconjtrans
     * @param[in] sphericalFunctionKetTimesVectorParFlattened multivector to
     * store results of CconjtransX which is initiliased using
     * initialiseFlattenedVector call
     * @param[out] dst output nodal vector where the results of the operator is
     * copied into.
     */
    void
    applyCVCconjtransOnX(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      const unsigned int                                         kPointIndex,
      const CouplingStructure                                    couplingtype,
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &couplingMatrix,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &dst);

    /**
     * @brief adds the result of CVCtX onto Xout for both CPU and GPU calls
     * @param[out] Xout memoryStorage object of size
     * cells*numberOfNodex*BlockSize. Typical case holds the results of H_{loc}X
     * @param[in] cellRange start and end element id in list of nonlocal
     * elements
     */
    void
    applyCOnVCconjtransX(ValueType *                                 Xout,
                         const std::pair<unsigned int, unsigned int> cellRange);

    std::vector<ValueType>
    getCmatrixEntries(int kPointIndex, unsigned int atomId, int iElem) const;

    bool
    atomPresentInCellRange(
      const std::pair<unsigned int, unsigned int> cellRange) const;
    /**
     * @brief Called only for GPU runs where the coupling matrix has to be padded
     * @param[in] entries COupling matrix entries without padding in the atomId
     * order
     * @param[out] entriesPadded Padding of coupling matrix entries
     * @param[in] couplingtype Determines the dimension of entriesPadded and the
     * padding mechanism elements
     */
    void
    paddingCouplingMatrix(const std::vector<ValueType> &entries,
                          std::vector<ValueType> &      entriesPadded,
                          const CouplingStructure       couplingtype);

  protected:
    bool                d_AllReduceCompleted;
    std::vector<double> d_kPointWeights;
    std::vector<double> d_kPointCoordinates;
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomCenteredSphericalFunctionContainer;
    std::shared_ptr<
      const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
                              d_mpiPatternP2P;
    std::vector<unsigned int> d_numberCellsForEachAtom;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperatorPtr;


    // Required by force.cc
    std::vector<ValueType> d_atomCenteredKpointIndexedSphericalFnQuadValues;
    // Required for stress compute
    std::vector<ValueType>
      d_atomCenteredKpointTimesSphericalFnTimesDistFromAtomQuadValues;

    /// map from cell number to set of non local atom ids (local numbering)
    std::map<unsigned int, std::vector<unsigned int>>
      d_cellIdToAtomIdsLocalCompactSupportMap;

    /// vector of size num physical cells
    std::vector<unsigned int> d_nonTrivialSphericalFnPerCell;

    /// vector of size num physical cell with starting index for each cell for
    /// the above array
    std::vector<unsigned int> d_nonTrivialSphericalFnsCellStartIndex;

    std::vector<unsigned int> d_nonTrivialAllCellsSphericalFnAlphaToElemIdMap;

    /// map from local nonlocal atomid to vector over cells
    std::map<unsigned int, std::vector<unsigned int>>
      d_atomIdToNonTrivialSphericalFnCellStartIndex;

    unsigned int d_sumNonTrivialSphericalFnOverAllCells;

    std::vector<unsigned int> d_sphericalFnTimesVectorFlattenedVectorLocalIds;

    // The above set of variables are needed in force class

#ifdef USE_COMPLEX
    std::vector<distributedCPUVec<std::complex<double>>>
      d_SphericalFunctionKetTimesVectorPar;

#else
    std::vector<distributedCPUVec<double>> d_SphericalFunctionKetTimesVectorPar;
#endif

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_sphericalFunctionIdsNumberingMapCurrentProcess;

    std::vector<unsigned int> d_OwnedAtomIdsInCurrentProcessor;
    dealii::IndexSet          d_locallyOwnedAtomCenteredFnIdsCurrentProcess;
    dealii::IndexSet          d_ghostAtomCenteredFnIdsCurrentProcess;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_AtomCenteredFnIdsNumberingMapCurrentProcess;
    std::vector<std::vector<
      std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>>>>
                               d_CMatrixEntries;
    dealii::ConditionalOStream pcout;
    const MPI_Comm             d_mpi_communicator;
    const unsigned int         d_this_mpi_process;
    const unsigned int         d_n_mpi_processes;
    dealii::IndexSet           d_locallyOwnedSphericalFunctionIdsCurrentProcess;
    dealii::IndexSet           d_ghostSphericalFunctionIdsCurrentProcess;

    unsigned int d_totalAtomsInCurrentProc; // number of atoms of interst with
                                            // compact in current processor
    unsigned int
      d_totalNonlocalElems; // number of nonlocal FE celss having nonlocal
                            // contribution in current processor
    unsigned int d_totalNonLocalEntries; // Total number of nonlocal components
    unsigned int
      d_maxSingleAtomContribution; // maximum number of nonlocal indexes across
                                   // all atoms of interset
    std::vector<unsigned int> d_numberCellsAccumNonLocalAtoms;
    unsigned int d_numberNodesPerElement; // Access from BasisOperator WHile
                                          // filling CMatrixEntries
    unsigned int d_locallyOwnedCells;
    unsigned int d_numberWaveFunctions;
    unsigned int d_kPointIndex;
    bool         d_memoryOptMode;
    bool         d_isMallocCalled = false;
    // Host CMatrix Entries are stored here
    std::vector<std::vector<std::vector<ValueType>>> d_CMatrixEntriesConjugate,
      d_CMatrixEntriesTranspose;

  private:
    /**
     * @brief stores the d_kpointWeights, d_kpointCoordinates. Other data members regarding are computed from container data object
     * @param[in] kPointWeights std::vector<double> of size number of kPoints
     * @param[out] kPointCoordinates std::vector<double> of kPoint coordinates
     */
    void
    initKpoints(const std::vector<double> &kPointWeights,
                const std::vector<double> &kPointCoordinates);
    /**
     * @brief creates the partitioner for the distributed vector based on sparsity patten from sphericalFn container.
     * @param[in] basisOperationsPtr HOST FEBasisOperations shared_ptr required
     * to indetify the element ids and quad points.
     */
    void
    initialisePartitioner();
    /**
     * @brief computes the entries in C matrix for CPUs and GPUs. On GPUs the entries are copied to a flattened vector on device memory.
     * Further on GPUs, various maps are created crucial for accessing and
     * padding entries in Cmatrix flattened device.
     * @param[in] basisOperationsPtr HOST FEBasisOperations shared_ptr required
     * to indetify the element ids and quad points
     * @param[in] quadratureIndex quadrature index for sampling the spherical
     * function. Quadrature Index is used to reinit basisOperationsPtr
     */
    void
    computeCMatrixEntries(
      std::shared_ptr<dftfe::basis::FEBasisOperations<
        dataTypes::number,
        double,
        dftfe::utils::MemorySpace::HOST>> basisOperationsPtr,
      const unsigned int                  quadratureIndex);

    std::map<
      unsigned int,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>>
      d_sphericalFnTimesWavefunMatrix;
    std::vector<dftfe::global_size_type>
      d_flattenedNonLocalCellDofIndexToProcessDofIndexVector;
    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
                              d_flattenedNonLocalCellDofIndexToProcessDofIndexMap;
    std::vector<unsigned int> d_nonlocalElemIdToCellIdVector;
#if defined(DFTFE_WITH_DEVICE)
    /**
     * @brief Copies the data from distributed Vector to Padded Memory storage object.
     * @param[in] sphericalFunctionKetTimesVectorParFlattened Distributed Vector
     * @param[out] paddedVector Padded Vector of size
     * noAtomsInProc*maxSingleAtomContribution*Nwfc
     */
    void
    copyDistributedVectorToPaddedMemoryStorageVectorDevice(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened,
      dftfe::utils::MemoryStorage<ValueType, memorySpace> &paddedVector);

    /**
     * @brief Copies Padded Memory storage object to Distributed vector.
     * @param[in] paddedVector Padded Vector of size
     * noAtomsInProc*maxSingleAtomContribution*Nwfc
     * @param[out] sphericalFunctionKetTimesVectorParFlattened Distributed
     * Vector
     *
     */
    void
    copyPaddedMemoryStorageVectorToDistributeVectorDevice(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> &paddedVector,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
        &sphericalFunctionKetTimesVectorParFlattened);


    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
      d_tempConjtansX;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                d_sphericalFnTimesWavefunctionMatrix;
    ValueType **hostPointerCDagger, **hostPointerCDaggeOutTemp,
      **hostWfcPointers;
    ValueType * d_wfcStartPointer;
    ValueType **devicePointerCDagger, **devicePointerCDaggerOutTemp,
      **deviceWfcPointers;
    std::vector<unsigned int> d_nonlocalElemIdToLocalElemIdMap;

    // The below memory storage objects receives the copy of the distributed
    // ketTimesWfc data in a padded form. THe padding is done by
    // copyDistributedVectorToPaddedMemoryStorageVector
    dftfe::utils::MemoryStorage<ValueType, memorySpace>
      d_sphericalFnTimesVectorDevice;
    // Data structures moved from KSOperatorDevice
    std::vector<ValueType> d_cellHamiltonianMatrixNonLocalFlattenedConjugate;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                           d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice;
    std::vector<ValueType> d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_cellHamMatrixTimesWaveMatrixNonLocalDevice;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
                           d_sphericalFnTimesVectorAllCellsDevice;
    std::vector<ValueType> d_sphericalFnTimesVectorAllCellsReduction;
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_sphericalFnTimesVectorAllCellsReductionDevice;

    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
      d_couplingMatrixTimesVectorDevice;

    std::vector<unsigned int> d_sphericalFnIdsParallelNumberingMap;
    std::vector<int>          d_sphericalFnIdsPaddedParallelNumberingMap;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_sphericalFnIdsParallelNumberingMapDevice;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE>
                     d_sphericalFnIdsPaddedParallelNumberingMapDevice;
    std::vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE>
                              d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice;
    std::vector<unsigned int> d_cellNodeIdMapNonLocalToLocal;

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_cellNodeIdMapNonLocalToLocalDevice;
#endif
  };



} // namespace dftfe
#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATOR_H
