//
// -------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//
#if defined(DFTFE_WITH_DEVICE)
#  ifndef operatorDFTDeviceClass_h
#    define operatorDFTDeviceClass_h

#    include <constraintMatrixInfoDevice.h>
#    include <constraintMatrixInfo.h>
#    include <DeviceBlasWrapper.h>
#    include <MemoryStorage.h>
#    include <headers.h>
#    include "process_grid.h"
#    include "scalapackWrapper.h"

#    include <vector>

#    include "deviceDirectCCLWrapper.h"

namespace dftfe
{
  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   *
   * @author Phani Motamarri, Sambit Das
   */
  class operatorDFTDeviceClass
  {
    //
    // methods
    //
  public:
    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTDeviceClass() = 0;

    /**
     * @brief initialize operatorClass
     *
     */
    virtual void
    init() = 0;


    virtual void
    createDeviceBlasHandle() = 0;

    virtual void
    destroyDeviceBlasHandle() = 0;

    virtual dftfe::utils::deviceBlasHandle_t &
    getDeviceBlasHandle() = 0;

    virtual const double *
    getSqrtMassVec() = 0;

    virtual const double *
    getInvSqrtMassVec() = 0;

    virtual distributedCPUVec<dataTypes::number> &
    getProjectorKetTimesVectorSingle() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionGradientIntegral() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionGradientIntegralElectro() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionValues() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionValuesTransposed(const bool use2pPlusOneGLQuad = false) = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionValuesNLPTransposed() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionGradientValuesNLPTransposed() = 0;

    virtual dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getInverseJacobiansNLP() = 0;

    virtual dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getFlattenedArrayCellLocalProcIndexIdMap() = 0;

    virtual dftfe::utils::MemoryStorage<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getCellWaveFunctionMatrix() = 0;

    virtual distributedCPUVec<dataTypes::number> &
    getParallelVecSingleComponent() = 0;

    virtual distributedDeviceVec<dataTypes::number> &
    getParallelChebyBlockVectorDevice() = 0;

    virtual distributedDeviceVec<dataTypes::number> &
    getParallelChebyBlockVector2Device() = 0;

    virtual distributedDeviceVec<dataTypes::number> &
    getParallelProjectorKetTimesBlockVectorDevice() = 0;

    virtual dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE> &
    getLocallyOwnedProcBoundaryNodesVectorDevice() = 0;

    /**
     * @brief initializes parallel layouts and index maps for HX, XtHX and creates a flattened array format for X
     *
     * @param wavefunBlockSize number of wavefunction vector (block size of X).
     * @param flag controls the creation of flattened array format and index maps or only index maps
     *
     * @return X format to store a multi-vector array
     * in a flattened format with all the wavefunction values corresponding to a
     * given node being stored contiguously
     *
     */

    virtual void
    reinit(const unsigned int wavefunBlockSize, bool flag) = 0;

    /**
     * @brief sets the data member to appropriate kPoint and spin Index
     *
     * @param kPointIndex  k-point Index to set
     */
    virtual void
    reinitkPointSpinIndex(const unsigned int kPointIndex,
                          const unsigned int spinIndex) = 0;


    /**
     * @brief compute diagonal mass matrix
     *
     * @param dofHandler dofHandler associated with the current mesh
     * @param constraintMatrix constraints to be used
     * @param sqrtMassVec output the value of square root of diagonal mass matrix
     * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
     */
    virtual void
    computeMassVector(const dealii::DoFHandler<3> &            dofHandler,
                      const dealii::AffineConstraints<double> &constraintMatrix,
                      distributedCPUVec<double> &              sqrtMassVec,
                      distributedCPUVec<double> &invSqrtMassVec) = 0;


    /**
     * @brief Compute operator times multi-field vectors
     *
     * @param X Vector containing multi-wavefunction fields (though X does not
     * change inside the function it is scaled and rescaled back to
     * avoid duplication of memory and hence is not const)
     * @param numberComponents number of wavefunctions associated with a given node
     * @param Y Vector containing multi-component fields after operator times vectors product
     */
    virtual void
    HX(distributedDeviceVec<dataTypes::number> &X,
       distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
       const unsigned int                       localVectorSize,
       const unsigned int                       numberComponents,
       const bool                               scaleFlag,
       const double                             scalar,
       distributedDeviceVec<dataTypes::number> &Y,
       const bool                               doUnscalingX    = true,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    virtual void
    HXCheby(distributedDeviceVec<dataTypes::number> &    X,
            distributedDeviceVec<dataTypes::numberFP32> &XTemp,
            distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
            const unsigned int                       localVectorSize,
            const unsigned int                       numberComponents,
            distributedDeviceVec<dataTypes::number> &Y,
            bool                                     mixPrecFlag = false,
            bool returnBeforeCompressSkipUpdateSkipNonLocal      = false,
            bool returnBeforeCompressSkipUpdateSkipLocal         = false) = 0;

    /**
     * @brief implementation of non-local projector kets times psi product
     * using non-local discretized projectors at cell-level.
     * works for both complex and real data type
     * @param src Vector containing current values of source array with multi-vector array stored
     * in a flattened format with all the wavefunction value corresponding to a
     * given node is stored contiguously.
     * @param numberWaveFunctions Number of wavefunctions at a given node.
     */
    virtual void
    computeNonLocalProjectorKetTimesXTimesV(
      const dataTypes::number *                src,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                       numberWaveFunctions) = 0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given basis
     *
     * @param X Vector of Vectors containing all wavefunction vectors
     * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
     * @param HXb parallel distributed vector datastructure for handling H multiplied by block of
     * wavefunction vectors
     * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal
     * projector kets times block wavefunction vectors
     * @param M number of local dofs
     * @param N total number of wavefunction vectors
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHX(const dataTypes::number *                X,
         distributedDeviceVec<dataTypes::number> &Xb,
         distributedDeviceVec<dataTypes::number> &HXb,
         distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
         const unsigned int                       M,
         const unsigned int                       N,
         dftfe::utils::deviceBlasHandle_t &       handle,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         utils::DeviceCCLWrapper &devicecclMpiCommDomain,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given basis.
     * This routine also overlaps communication and computation.
     *
     * @param X Vector of Vectors containing all wavefunction vectors
     * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
     * @param HXb parallel distributed vector datastructure for handling H multiplied by block of
     * wavefunction vectors
     * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal
     * projector kets times block wavefunction vectors
     * @param M number of local dofs
     * @param N total number of wavefunction vectors
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHXOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given basis.
     * This routine uses a mixed precision algorithm
     * (https://doi.org/10.1016/j.cpc.2019.07.016) and further overlaps
     * communication and computation.
     *
     * @param X Vector of Vectors containing all wavefunction vectors
     * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
     * @param floatXb parallel distributed vector datastructure for handling block of wavefunction
     * vectors in single precision
     * @param HXb parallel distributed vector datastructure for handling H multiplied by block of
     * wavefunction vectors
     * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal
     * projector kets times block wavefunction vectors
     * @param M number of local dofs
     * @param N total number of wavefunction vectors
     * @param Noc number of fully occupied wavefunction vectors considered in the mixed precision algorithm
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHXMixedPrecOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::numberFP32> &    floatXb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;

    virtual void
    XtHXMixedPrecCommunOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    dftUtils::constraintMatrixInfo *
    getOverloadedConstraintMatrixHost() const;


    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    dftUtils::constraintMatrixInfoDevice *
    getOverloadedConstraintMatrix() const;


    /**
     * @brief Get matrix free data
     *
     * @return pointer to matrix free data
     */
    const dealii::MatrixFree<3, double> *
    getMatrixFreeData() const;


    /**
     * @brief Get relevant mpi communicator
     *
     * @return mpi communicator
     */
    const MPI_Comm &
    getMPICommunicator() const;


  protected:
    /**
     * @brief default Constructor.
     */
    operatorDFTDeviceClass();


    /**
     * @brief Constructor.
     */
    operatorDFTDeviceClass(
      const MPI_Comm &                      mpi_comm_replica,
      const dealii::MatrixFree<3, double> & matrix_free_data,
      dftUtils::constraintMatrixInfo &      constraintMatrixNone,
      dftUtils::constraintMatrixInfoDevice &constraintMatrixNoneDevice);

  protected:
    //
    // Get overloaded constraint matrix object constructed using 1-component FE
    // object
    //
    dftUtils::constraintMatrixInfo *d_constraintMatrixData;

    //
    // Get overloaded constraint matrix object constructed using 1-component FE
    // object
    //
    dftUtils::constraintMatrixInfoDevice *d_constraintMatrixDataDevice;
    //
    // matrix-free data
    //
    const dealii::MatrixFree<3, double> *d_matrix_free_data;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_cellShapeFunctionGradientIntegralFlattenedDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_cellShapeFunctionGradientIntegralFlattenedDeviceElectro;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_shapeFunctionValueDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_shapeFunctionValueTransposedDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_shapeFunctionValueNLPTransposedDevice;

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
    //   d_shapeFunctionGradientValueXTransposedDevice;

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
    //   d_shapeFunctionGradientValueYTransposedDevice;

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
    //   d_shapeFunctionGradientValueZTransposedDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_shapeFunctionGradientValueNLPTransposedDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_inverseJacobiansNLPDevice;

    /// 2p+1 Gauss Lobotta quadrature shape function values and shape function
    /// gradients
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_glShapeFunctionValueTransposedDevice;


    dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                dftfe::utils::MemorySpace::DEVICE>
      d_flattenedArrayCellLocalProcIndexIdMapDevice;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellWaveFunctionMatrix;

    distributedDeviceVec<dataTypes::number> d_parallelChebyBlockVectorDevice;

    distributedDeviceVec<dataTypes::number> d_parallelChebyBlockVector2Device;

    distributedDeviceVec<dataTypes::number>
      d_parallelProjectorKetTimesBlockVectorDevice;

    distributedCPUVec<dataTypes::number> d_parallelVecSingleComponent;

    //
    // mpi communicator
    //
    MPI_Comm d_mpi_communicator;
  };

} // namespace dftfe
#  endif
#endif
