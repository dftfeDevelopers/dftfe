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


#ifndef vectorUtilities_h
#define vectorUtilities_h

#include <headers.h>
#include <operator.h>



namespace dftfe
{
  /**
   *  @brief Contains generic utils functions related to custom partitioned flattened dealii vector
   *
   *  @author Phani Motamarri, Sambit Das
   */
  namespace vectorTools
  {
    /** @brief Create constraint matrix using serial mesh.
     *  Temporary fix for a bug (Issue #7053) in deal.ii until it is resolved.
     *
     *  @param[in] serial Triangulation which must be exactly same as the
     * parallel triangulation associated with dofHandlerPar
     *  @param[in] parallel DofHandler
     *  @param[out] periodic hanging constraints.
     *  @param[out] only hanging constraints
     */
    void
    createParallelConstraintMatrixFromSerial(
      const dealii::Triangulation<3, 3> &     serTria,
      const dealii::DoFHandler<3> &           dofHandlerPar,
      const MPI_Comm &                        mpi_comm_parent,
      const MPI_Comm &                        mpi_comm_domain,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      dealii::AffineConstraints<double> &     periodicHangingConstraints,
      dealii::AffineConstraints<double> &     onlyHangingConstraints,
      const int                               verbosity,
      const bool                              periodicX,
      const bool                              periodicY,
      const bool                              periodicZ);


    /** @brief Creates a custom partitioned flattened dealii vector.
     *  stores multiple components asociated with a node sequentially.
     *
     *  @param partitioner associated with single component vector
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArray custom partitioned dealii vector
     */
    template <typename T>
    void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                   partitioner,
      const unsigned int    blockSize,
      distributedCPUVec<T> &flattenedArray);



    /** @brief Creates a cell local index set map for flattened array
     *
     *  @param partitioner associated with the flattened array
     *  @param matrix_free_data object pointer associated with the matrix free data structure
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArrayMacroCellLocalProcIndexId macrocell's subcell local proc index map
     *  @return flattenedArrayCellLocalProcIndexId cell local proc index map
     */
    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<std::vector<dealii::types::global_dof_index>>
        &flattenedArrayMacroCellLocalProcIndexId,
      std::vector<std::vector<dealii::types::global_dof_index>>
        &flattenedArrayCellLocalProcIndexId);


    /** @brief Creates a cell local index set map for flattened array
     *
     *  @param partitioner associated with the flattened array
     *  @param matrix_free_data object pointer associated with the matrix free data structure
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArrayMacroCellLocalProcIndexId macrocell's subcell local proc index map
     *  @return flattenedArrayCellLocalProcIndexId cell local proc index map
     */
    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<dealii::types::global_dof_index>
        &                        flattenedArrayMacroCellLocalProcIndexId,
      std::vector<unsigned int> &normalCellIdToMacroCellIdMap,
      std::vector<unsigned int> &macroCellIdToNormalCellIdMap,
      std::vector<dealii::types::global_dof_index>
        &flattenedArrayCellLocalProcIndexId);

    /** @brief Creates a cell local index set map for flattened array
     *
     *  @param partitioner associated with the flattened array
     *  @param matrix_free_data object pointer associated with the matrix free data structure
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArrayMacroCellLocalProcIndexId macrocell's subcell local proc index map
     *  @return flattenedArrayCellLocalProcIndexId cell local proc index map
     */
    void
    computeCellLocalIndexSetMap(
      const std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
        &                                  partitioner,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      const unsigned int                   blockSize,
      std::vector<dealii::types::global_dof_index>
        &flattenedArrayCellLocalProcIndexId);


#ifdef USE_COMPLEX
    /** @brief Copies a single field component from a flattenedArray STL
     * vector containing multiple component fields to a 2-component field (real
     * and complex) parallel distributed vector.
     *
     *  @param[in] flattenedArray flattened STL vector with multiple component
     * fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] localProcDofIndicesReal local dof indices in the current
     * processor which correspond to component-1 of 2-component parallel
     * distributed array
     *  @param[in] localProcDofIndicesImag local dof indices in the current
     * processor which correspond to component-2 of 2-component parallel
     * distributed array
     *  @param[out] componentVectors vector of two component field parallel
     * distributed vectors with the values corresponding to fields of
     * componentIndexRange of flattenedArray. componentVectors is expected to be
     * of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the 2-component version of the same single component
     * partitioner used in the creation of the flattenedArray partitioner.
     */
    void
    copyFlattenedSTLVecToSingleCompVec(
      const std::complex<double> *                flattenedArray,
      const unsigned int                          totalNumberComponents,
      const unsigned int                          localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                     localProcDofIndicesImag,
      std::vector<distributedCPUVec<double>> &componentVectors);

    void
    copyFlattenedSTLVecToSingleCompVec(
      const std::complex<double> *                flattenedArray,
      const unsigned int                          totalNumberComponents,
      const unsigned int                          localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,

      std::vector<distributedCPUVec<double>> &componentVectors);

#else
    /** @brief Copies a single field component from a flattenedArray STL
     * vector containing multiple component fields to a single field parallel
     * distributed vector.
     *
     *  @param[in] flattenedArray flattened STL vector with multiple component
     * fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[out] componentVectors vector of parallel distributed vectors with
     * fields corresponding to componentIndexRange. componentVectors is expected
     * to be of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the same single component partitioner used in the
     * creation of the flattenedArray partitioner.
     */
    void
    copyFlattenedSTLVecToSingleCompVec(
      const double *                              flattenedArray,
      const unsigned int                          totalNumberComponents,
      const unsigned int                          localVectorSize,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      std::vector<distributedCPUVec<double>> &    componentVectors);

#endif

#ifdef USE_COMPLEX
    /** @brief Copies a single field component from a flattenedArray parallel distributed
     * vector containing multiple component fields to a 2-component field (real
     * and complex) parallel distributed vector.
     *
     *  @param[in] flattenedArray flattened parallel distributed vector with
     * multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] localProcDofIndicesReal local dof indices in the current
     * processor which correspond to component-1 of 2-component parallel
     * distributed array
     *  @param[in] localProcDofIndicesImag local dof indices in the current
     * processor which correspond to component-2 of 2-component parallel
     * distributed array
     *  @param[out] componentVectors vector of two component field parallel
     * distributed vectors with the values corresponding to fields of
     * componentIndexRange of flattenedArray. componentVectors is expected to be
     * of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the 2-component version of the same single component
     * partitioner used in the creation of the flattenedArray partitioner.
     *  @param[in] isFlattenedDealiiGhostValuesUpdated default is false. Use
     * true for optimization if update ghost values has already been called in
     * the flattened dealii vec.
     */
    void
    copyFlattenedDealiiVecToSingleCompVec(
      const distributedCPUVec<std::complex<double>> &flattenedArray,
      const unsigned int                             totalNumberComponents,
      const std::pair<unsigned int, unsigned int>    componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                     localProcDofIndicesImag,
      std::vector<distributedCPUVec<double>> &componentVectors,
      const bool isFlattenedDealiiGhostValuesUpdated = false);

#else
    /** @brief Copies a single field component from a flattenedArray parallel distributed
     * vector containing multiple component fields to a single field parallel
     * distributed vector.
     *
     *  @param[in] flattenedArray flattened parallel distributed vector with
     * multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[out] componentVectors vector of parallel distributed vectors with
     * fields corresponding to componentIndexRange. componentVectors is expected
     * to be of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the same single component partitioner used in the
     * creation of the flattenedArray partitioner.
     *  @param[in] isFlattenedDealiiGhostValuesUpdated default is false. Use
     * true for optimization if update ghost values has already been called in
     * the flattened dealii vec.
     */
    void
    copyFlattenedDealiiVecToSingleCompVec(
      const distributedCPUVec<double> &           flattenedArray,
      const unsigned int                          totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      std::vector<distributedCPUVec<double>> &    componentVectors,
      const bool isFlattenedDealiiGhostValuesUpdated = false);

#endif

#ifdef USE_COMPLEX
    /** @brief Copies to a flattenedArray parallel distributed
     * vector containing multiple component fields from a 2-component field
     * (real and complex) parallel distributed vector.
     *
     *  @param[out] flattenedArray flattened parallel distributed vector with
     * multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] localProcDofIndicesReal local dof indices in the current
     * processor which correspond to component-1 of 2-component parallel
     * distributed array
     *  @param[in] localProcDofIndicesImag local dof indices in the current
     * processor which correspond to component-2 of 2-component parallel
     * distributed array
     *  @param[in] componentVectors vector of two component field parallel
     * distributed vectors with the values corresponding to fields of
     * componentIndexRange of flattenedArray. componentVectors is expected to be
     * of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the 2-component version of the same single component
     * partitioner used in the creation of the flattenedArray partitioner.
     */
    void
    copySingleCompVecToFlattenedDealiiVec(
      distributedCPUVec<std::complex<double>> &   flattenedArray,
      const unsigned int                          totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                           localProcDofIndicesImag,
      const std::vector<distributedCPUVec<double>> &componentVectors);

#else
    /** @brief Copies to a flattenedArray parallel distributed
     * vector containing multiple component fields from a single field parallel
     * distributed vector.
     *
     *  @param[out] flattenedArray flattened parallel distributed vector with
     * multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] componentVectors vector of parallel distributed vectors with
     * fields corresponding to componentIndexRange. componentVectors is expected
     * to be of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the same single component partitioner used in the
     * creation of the flattenedArray partitioner.
     */
    void
    copySingleCompVecToFlattenedDealiiVec(
      distributedCPUVec<double> &                   flattenedArray,
      const unsigned int                            totalNumberComponents,
      const std::pair<unsigned int, unsigned int>   componentIndexRange,
      const std::vector<distributedCPUVec<double>> &componentVectors);

#endif

#ifdef USE_COMPLEX
    /** @brief Copies to a flattenedArray stl
     * vector containing multiple component fields from a 2-component field
     * (real and complex) parallel distributed vector.
     *
     *  @param[out] flattenedArray flattened stl vector with multiple component
     * fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] localProcDofIndicesReal local dof indices in the current
     * processor which correspond to component-1 of 2-component parallel
     * distributed array
     *  @param[in] localProcDofIndicesImag local dof indices in the current
     * processor which correspond to component-2 of 2-component parallel
     * distributed array
     *  @param[in] componentVectors vector of two component field parallel
     * distributed vectors with the values corresponding to fields of
     * componentIndexRange of flattenedArray. componentVectors is expected to be
     * of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the 2-component version of the same single component
     * partitioner used in the creation of the flattenedArray partitioner.
     */
    void
    copySingleCompVecToFlattenedSTLVec(
      std::vector<std::complex<double>> &         flattenedArray,
      const unsigned int                          totalNumberComponents,
      const std::pair<unsigned int, unsigned int> componentIndexRange,
      const std::vector<dealii::types::global_dof_index>
        &localProcDofIndicesReal,
      const std::vector<dealii::types::global_dof_index>
        &                                           localProcDofIndicesImag,
      const std::vector<distributedCPUVec<double>> &componentVectors);

#else
    /** @brief Copies to a flattenedArray stl
     * vector containing multiple component fields from a single field parallel
     * distributed vector.
     *
     *  @param[out] flattenedArray flattened stl vector with multiple component
     * fields
     *  @param[in] totalNumberComponents total number of component fiels in
     * flattenedArray
     *  @param[in] componentIndexRange desired range field components
     *  [componentIndexRange.first,componentIndexRange.second)
     *  @param[in] componentVectors vector of parallel distributed vectors with
     * fields corresponding to componentIndexRange. componentVectors is expected
     * to be of the size componentIndexRange.second-componentIndexRange.first.
     * Further, each entry of componentVectors is assumed to be already
     * initialized with the same single component partitioner used in the
     * creation of the flattenedArray partitioner.
     */
    void
    copySingleCompVecToFlattenedSTLVec(
      std::vector<double> &                         flattenedArray,
      const unsigned int                            totalNumberComponents,
      const std::pair<unsigned int, unsigned int>   componentIndexRange,
      const std::vector<distributedCPUVec<double>> &componentVectors);

#endif

    std::pair<dealii::Point<3>, dealii::Point<3>>
    createBoundingBoxTriaLocallyOwned(const dealii::DoFHandler<3> &dofHandler);


    void
    classifyInteriorSurfaceNodesInCell(
      const dealii::MatrixFree<3, double> &matrix_free_data,
      const unsigned int                   mfDofHandlerIndex,
      std::vector<unsigned int> &          nodesPerCellClassificationMap);


    void
    classifyInteriorSurfaceNodesInGlobalArray(
      const dealii::MatrixFree<3, double> &    matrix_free_data,
      const unsigned int                       mfDofHandlerIndex,
      const dealii::AffineConstraints<double> &constraintMatrix,
      std::vector<unsigned int> &              nodesPerCellClassificationMap,
      std::vector<unsigned int> &              globalArrayClassificationMap);



  } // namespace vectorTools
} // namespace dftfe
#endif
