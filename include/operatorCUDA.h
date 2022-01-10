//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
#if defined(DFTFE_WITH_GPU)
#  ifndef operatorDFTCUDAClass_h
#    define operatorDFTCUDAClass_h

#    include <constraintMatrixInfoCUDA.h>
#    include <constraintMatrixInfo.h>
#    include <cublas_v2.h>
#    include <headers.h>
#    include <thrust/device_vector.h>
#    include <thrust/host_vector.h>
#    include "process_grid.h"
#    include "scalapackWrapper.h"

#    include <vector>

#    include "gpuDirectCCLWrapper.h"
#    include "distributedMulticomponentVec.h"

namespace dftfe
{
  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   *
   * @author Phani Motamarri, Sambit Das
   */
  class operatorDFTCUDAClass
  {
    //
    // methods
    //
  public:
    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTCUDAClass() = 0;

    /**
     * @brief initialize operatorClass
     *
     */
    virtual void
    init() = 0;


    virtual void
    createCublasHandle() = 0;

    virtual void
    destroyCublasHandle() = 0;

    virtual cublasHandle_t &
    getCublasHandle() = 0;

    virtual const double *
    getSqrtMassVec() = 0;

    virtual const double *
    getInvSqrtMassVec() = 0;

    virtual thrust::device_vector<unsigned int> &
    getBoundaryIdToLocalIdMap() = 0;

    virtual distributedCPUVec<dataTypes::number> &
    getProjectorKetTimesVectorSingle() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientIntegral() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientIntegralElectro() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionValues() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionValuesInverted(const bool use2pPlusOneGLQuad = false) = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionValuesNLPInverted() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientValuesXInverted() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientValuesYInverted() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientValuesZInverted() = 0;

    virtual thrust::device_vector<double> &
    getShapeFunctionGradientValuesNLPInverted() = 0;

    virtual thrust::device_vector<double> &
    getInverseJacobiansNLP() = 0;

    virtual thrust::device_vector<dealii::types::global_dof_index> &
    getFlattenedArrayCellLocalProcIndexIdMap() = 0;

    virtual thrust::device_vector<dataTypes::numberThrustGPU> &
    getCellWaveFunctionMatrix() = 0;

    virtual distributedCPUVec<dataTypes::number> &
    getParallelVecSingleComponent() = 0;

    virtual distributedGPUVec<dataTypes::numberGPU> &
    getParallelChebyBlockVectorDevice() = 0;

    virtual distributedGPUVec<dataTypes::numberGPU> &
    getParallelChebyBlockVector2Device() = 0;

    virtual distributedGPUVec<dataTypes::numberGPU> &
    getParallelProjectorKetTimesBlockVectorDevice() = 0;

    virtual thrust::device_vector<unsigned int> &
    getLocallyOwnedProcBoundaryNodesVectorDevice() = 0;

    virtual thrust::device_vector<unsigned int> &
    getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice() = 0;

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
    HX(distributedGPUVec<dataTypes::numberGPU> &X,
       distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
       const unsigned int                       localVectorSize,
       const unsigned int                       numberComponents,
       const bool                               scaleFlag,
       const double                             scalar,
       distributedGPUVec<dataTypes::numberGPU> &Y,
       const bool                               doUnscalingX    = true,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    virtual void
    HXCheby(distributedGPUVec<dataTypes::numberGPU> &    X,
            distributedGPUVec<dataTypes::numberFP32GPU> &XTemp,
            distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
            const unsigned int                       localVectorSize,
            const unsigned int                       numberComponents,
            distributedGPUVec<dataTypes::numberGPU> &Y,
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
      const dataTypes::numberGPU *             src,
      distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
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
     * @param handle cublasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHX(const dataTypes::numberGPU *             X,
         distributedGPUVec<dataTypes::numberGPU> &Xb,
         distributedGPUVec<dataTypes::numberGPU> &HXb,
         distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
         const unsigned int                       M,
         const unsigned int                       N,
         cublasHandle_t &                         handle,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         GPUCCLWrapper &                                  gpucclMpiCommDomain,
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
     * @param handle cublasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHXOverlapComputeCommun(
      const dataTypes::numberGPU *                     X,
      distributedGPUVec<dataTypes::numberGPU> &        Xb,
      distributedGPUVec<dataTypes::numberGPU> &        HXb,
      distributedGPUVec<dataTypes::numberGPU> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
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
     * @param handle cublasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHXMixedPrecOverlapComputeCommun(
      const dataTypes::numberGPU *                     X,
      distributedGPUVec<dataTypes::numberGPU> &        Xb,
      distributedGPUVec<dataTypes::numberFP32GPU> &    floatXb,
      distributedGPUVec<dataTypes::numberGPU> &        HXb,
      distributedGPUVec<dataTypes::numberGPU> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
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
    dftUtils::constraintMatrixInfoCUDA *
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
    operatorDFTCUDAClass();


    /**
     * @brief Constructor.
     */
    operatorDFTCUDAClass(
      const MPI_Comm &                     mpi_comm_replica,
      const dealii::MatrixFree<3, double> &matrix_free_data,
      dftUtils::constraintMatrixInfo &     constraintMatrixNone,
      dftUtils::constraintMatrixInfoCUDA & constraintMatrixNoneCUDA);

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
    dftUtils::constraintMatrixInfoCUDA *d_constraintMatrixDataCUDA;
    //
    // matrix-free data
    //
    const dealii::MatrixFree<3, double> *d_matrix_free_data;

    thrust::device_vector<double>
      d_cellShapeFunctionGradientIntegralFlattenedDevice;

    thrust::device_vector<double>
      d_cellShapeFunctionGradientIntegralFlattenedDeviceElectro;

    thrust::device_vector<double> d_shapeFunctionValueDevice;

    thrust::device_vector<double> d_shapeFunctionValueInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionValueNLPInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueXInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueYInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueZInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueNLPInvertedDevice;

    thrust::device_vector<double> d_inverseJacobiansNLPDevice;

    /// 2p+1 Gauss Lobotta quadrature shape function values and shape function
    /// gradients
    thrust::device_vector<double> d_glShapeFunctionValueInvertedDevice;

    thrust::device_vector<unsigned int> d_boundaryIdToLocalIdMapDevice;



    thrust::device_vector<dealii::types::global_dof_index>
      d_flattenedArrayCellLocalProcIndexIdMapDevice;

    thrust::device_vector<dataTypes::numberThrustGPU> d_cellWaveFunctionMatrix;

    distributedGPUVec<dataTypes::numberGPU> d_parallelChebyBlockVectorDevice;

    distributedGPUVec<dataTypes::numberGPU> d_parallelChebyBlockVector2Device;

    distributedGPUVec<dataTypes::numberGPU>
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
