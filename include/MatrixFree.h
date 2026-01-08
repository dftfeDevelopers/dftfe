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

/**
 * @author Gourab Panigrahi
 *
 */

#ifndef MatrixFree_H_
#define MatrixFree_H_
#include <type_traits>
#include <memory>
#include <dftfeDataTypes.h>
#include <MemorySpaceType.h>
#include <FEBasisOperations.h>
#include <MatrixFreeDevice.h>

#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif

namespace dftfe
{
  /**
   * @brief MatrixFree class template. template parameter nDofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   */
  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  class MatrixFree
  {
  public:
    static constexpr bool d_isComplex =
      std::is_same_v<dataTypes::number, std::complex<double>>;

    typedef std::conditional_t<d_isComplex, std::complex<T>, T> DataType;

    /// Constructor
    MatrixFree(const MPI_Comm                     &mpi_comm,
               std::shared_ptr<dftfe::basis::FEBasisOperations<
                 dataTypes::number,
                 double,
                 dftfe::utils::MemorySpace::HOST>> basisOperationsPtrHost,
               std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                   BLASWrapperPtr,
               const std::uint32_t operatorID,
               const std::uint32_t quadratureID,
               const std::uint32_t nVectors);

    /**
     * @brief Initialize data structures for MatrixFree class
     *
     */
    void
    init();

    /**
     * @brief Compute Laplace operator multipled by X
     *
     */
    inline void
    computeAX(T *dst, T *src);

    inline void
    constraintsDistribute(T *src);

    inline void
    constraintsDistributeTranspose(T *dst, T *src);

    inline void
    constraintsSetZero(T *src);

  private:
    /**
     * @brief Initialize optimized constraints
     *
     */
    void
    initConstraints();

    void
    setupConstraints(const dealii::IndexSet &indexSet);

    const std::uint32_t d_operatorID, d_quadratureID, d_nVectors, d_nBatch,
      d_nDofsPerCell, d_nQuadsPerCell;

    std::uint32_t d_nOwnedDofs, d_nRelaventDofs, d_nGhostDofs, d_nCells,
      d_localBlockSize, d_localSize, d_ghostBlockSize, d_ghostSize,
      d_nOMPThreads;

    static constexpr std::uint32_t d_quadODim = nQuadPointsPerDim / 2;
    static constexpr std::uint32_t d_quadEDim =
      nQuadPointsPerDim % 2 == 1 ? d_quadODim + 1 : d_quadODim;
    static constexpr std::uint32_t d_dofODim = nDofsPerDim / 2;
    static constexpr std::uint32_t d_dofEDim =
      nDofsPerDim % 2 == 1 ? d_dofODim + 1 : d_dofODim;

    std::array<T, d_quadEDim * d_dofEDim + d_quadODim * d_dofODim>
      nodalShapeFunctionValuesAtQuadPointsEO;
    std::array<T, 2 * d_quadODim * d_quadEDim>
                                     quadShapeFunctionGradientsAtQuadPointsEO;
    std::array<T, nQuadPointsPerDim> quadratureWeights;

    dftfe::utils::MemoryStorage<T, memorySpace>             d_jacobianFactor;
    dftfe::utils::MemoryStorage<std::uint32_t, memorySpace> d_map;

    // HOST only Data Structures
    std::vector<std::vector<std::uint32_t>> d_constrainingNodeBuckets,
      d_constrainedNodeBuckets;
    std::vector<std::vector<T>> d_weightMatrixList;
    std::vector<T>              d_inhomogenityList;

    // Device only Data Structures
#ifdef DFTFE_WITH_DEVICE
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::DEVICE>
      d_weightMatrixListDevice, d_inhomogenityListDevice;

    dftfe::utils::MemoryStorage<std::uint32_t,
                                dftfe::utils::MemorySpace::DEVICE>
      d_constrainingNodeBucketsDevice, d_constrainedNodeBucketsDevice,
      d_constrainingNodeOffsetDevice, d_constrainedNodeOffsetDevice,
      d_weightMatrixOffsetDevice;
#endif

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      d_singleVectorPartitioner, d_singleBatchPartitioner;

    dealii::ConditionalOStream pcout;
    const MPI_Comm             mpi_communicator;
    const std::uint32_t        n_mpi_processes;
    const std::uint32_t        this_mpi_process;
    std::vector<T>             tempGhostStorage, tempCompressStorage;
    std::vector<MPI_Request>   mpiRequestsGhost;
    std::vector<MPI_Request>   mpiRequestsCompress;
  };

  // List of operators
  enum operatorList
  {
    Laplace   = 1,
    Helmholtz = 2,
    LDA       = 3,
    GGA       = 4
  };

  // List of floating point representations
  enum floatingPointList
  {
    FP64 = 1,
    FP32 = 2
  };

} // namespace dftfe
#endif // MatrixFree_H_
