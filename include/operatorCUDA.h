//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#ifndef operatorDFTCUDAClass_h
#define operatorDFTCUDAClass_h

#include <vector>

#include <headers.h>
#include <constraintMatrixInfoCUDA.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>

namespace dftfe{

  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   *
   * @author Phani Motamarri, Sambit Das
   */
  class operatorDFTCUDAClass {

    //
    // methods
    //
  public:

    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTCUDAClass() = 0;

    unsigned int getScalapackBlockSize() const;

    unsigned int getScalapackBlockSizeValence() const;

    void processGridSetup(const unsigned int na,
                          const unsigned int nev);



    /**
     * @brief initialize operatorClass
     *
     */
    virtual void init() = 0;


    virtual void createCublasHandles() = 0;

    virtual void destroyCublasHandles() = 0;

    virtual cublasHandle_t & getCublasHandle() = 0;

    virtual const double * getSqrtMassVec() = 0;

    virtual const double * getInvSqrtMassVec() = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<dataTypes::number,dealii::MemorySpace::Host> &  getProjectorKetTimesVectorSingle()=0;

    //virtual cudaVectorType & getBlockCUDADealiiVector() = 0;

    //virtual cudaVectorType & getBlockCUDADealiiVector2() = 0;

    //virtual cudaVectorType & getBlockCUDADealiiVector3() = 0;
 
    //virtual thrust::device_vector<dataTypes::number> & getBlockCUDADealiiVector() = 0;

    //virtual thrust::device_vector<dataTypes::number> & getBlockCUDADealiiVector2() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientIntegral() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionValues() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionValuesInverted() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesX() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesY() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesZ() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesXInverted() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesYInverted() = 0;

    virtual thrust::device_vector<double> & getShapeFunctionGradientValuesZInverted() = 0;

    virtual thrust::device_vector<dealii::types::global_dof_index> & getFlattenedArrayCellLocalProcIndexIdMap()=0;

    virtual thrust::device_vector<dataTypes::number> & getCellWaveFunctionMatrix() = 0;

   /**
    * @brief initializes parallel layouts and index maps for HX, XtHX and creates a flattened array format for X
    *
    * @param wavefunBlockSize number of wavefunction vector (block size of X).
    * @param flag controls the creation of flattened array format and index maps or only index maps
    *
    * @return X format to store a multi-vector array
    * in a flattened format with all the wavefunction values corresponding to a given node being stored
    * contiguously
    *
    */
    virtual void reinit(const unsigned int wavefunBlockSize) = 0;

    virtual void reinit(const unsigned int wavefunBlockSize,
                        bool flag) = 0;

    /**
     * @brief compute diagonal mass matrix
     *
     * @param dofHandler dofHandler associated with the current mesh
     * @param constraintMatrix constraints to be used
     * @param sqrtMassVec output the value of square root of diagonal mass matrix
     * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
     */
    virtual void computeMassVector(const dealii::DoFHandler<3>    & dofHandler,
				   const dealii::AffineConstraints<double> & constraintMatrix,
				   vectorType                     & sqrtMassVec,
				   vectorType                     & invSqrtMassVec) = 0;


    /**
     * @brief Compute operator times vector or operator times bunch of vectors
     * @param X Vector of Vectors containing current values of X
     * @param Y Vector of Vectors containing operator times vectors product
     */
    virtual void HX(std::vector<vectorType> & X,
                    std::vector<vectorType> & Y) = 0;


    /**
     * @brief Compute operator times multi-field vectors
     *
     * @param X Vector containing multi-wavefunction fields (though X does not
     * change inside the function it is scaled and rescaled back to
     * avoid duplication of memory and hence is not const)
     * @param numberComponents number of wavefunctions associated with a given node
     * @param Y Vector containing multi-component fields after operator times vectors product
     */
    virtual void HX(cudaVectorType & X,
                    cudaVectorType & projectorKetTimesVector,
                    const unsigned int localVectorSize,
		    const unsigned int numberComponents,
		    const bool scaleFlag,
		    const double scalar,
		    cudaVectorType & Y) = 0;


    virtual void HXCheby(cudaVectorType & X,
                         cudaVectorTypeFloat &XTemp,
                         cudaVectorType & projectorKetTimesVector,
                         const unsigned int localVectorSize,
                         const unsigned int numberComponents,
                         cudaVectorType & Y,
                         bool mixPrecFlag=false) = 0;


      /**
       * @brief Compute projection of the operator into orthogonal basis
       *
       * @param X given orthogonal basis vectors
       * @return ProjMatrix projected small matrix
       */
     virtual void XtHX(const double *  X,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
                const unsigned int M,
		const unsigned int N,
                cublasHandle_t &handle,
		double* projHam,
                const bool isProjHamOnDevice=true)=0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
     virtual void XtHX(const double *  X,
                cudaVectorType & Xb,
                cudaVectorType & HXb,
                cudaVectorType & projectorKetTimesVector,
                const unsigned int M,
		const unsigned int N,
                cublasHandle_t &handle,
	        const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
	        dealii::ScaLAPACKMatrix<double> & projHamPar)=0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
     virtual void XtHXMixedPrec(const double *  X,
                cudaVectorType & Xb,
                cudaVectorType & HXb,
                cudaVectorType & projectorKetTimesVector,
                const unsigned int M,
		const unsigned int N,
                const unsigned int Noc,
                cublasHandle_t &handle,
	        const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
	        dealii::ScaLAPACKMatrix<double> & projHamPar)=0;
    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param  X Vector of Vectors containing the basis vectors spanning the subspace
     * @return ProjMatrix projected small matrix
     */
    /*virtual void XtHX(std::vector<vectorType> & X,
      std::vector<dataTypes::number> & ProjHam) = 0;*/



    /**
     * @brief Get local dof indices real
     *
     * @return pointer to local dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalDofIndicesReal() const;

    /**
     * @brief Get local dof indices imag
     *
     * @return pointer to local dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalDofIndicesImag() const;

    /**
     * @brief Get local proc dof indices real
     *
     * @return pointer to local proc dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalProcDofIndicesReal() const;


    /**
     * @brief Get local proc dof indices imag
     *
     * @return pointer to local proc dof indices imag
     */
    const std::vector<dealii::types::global_dof_index> * getLocalProcDofIndicesImag() const;

    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    const dealii::AffineConstraints<double> * getConstraintMatrixEigen() const;


    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    dftUtils::constraintMatrixInfoCUDA * getOverloadedConstraintMatrix() const;


    /**
     * @brief Get matrix free data
     *
     * @return pointer to matrix free data
     */
    const dealii::MatrixFree<3,double> * getMatrixFreeData() const;


    /**
     * @brief Get relevant mpi communicator
     *
     * @return mpi communicator
     */
    const MPI_Comm & getMPICommunicator() const;


  protected:

    /**
     * @brief default Constructor.
     */
    operatorDFTCUDAClass();


    /**
     * @brief Constructor.
     */
    operatorDFTCUDAClass(const MPI_Comm & mpi_comm_replica,
			 const dealii::MatrixFree<3,double> & matrix_free_data,
			 const std::vector<dealii::types::global_dof_index> & localDofIndicesReal,
			 const std::vector<dealii::types::global_dof_index> & localDofIndicesImag,
			 const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
			 const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
			 const dealii::AffineConstraints<double>  & constraintMatrixEigen,
			 dftUtils::constraintMatrixInfoCUDA & constraintMatrixNone);

  protected:


    //
    //global indices of degrees of freedom in the current processor which correspond to component-1 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localDofIndicesReal;

    //
    //global indices of degrees of freedom in the current processor which correspond to component-2 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localDofIndicesImag;

    //
    //local indices degrees of freedom in the current processor  which correspond to component-1 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localProcDofIndicesReal;

    //
    //local indices degrees of freedom in the current processor  which correspond to component-2 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localProcDofIndicesImag;

    //
    //constraint matrix used for the eigen problem (2-component FE Object for Periodic, 1-component FE object for non-periodic)
    //
    const dealii::AffineConstraints<double>  * d_constraintMatrixEigen;

    //
    //Get overloaded constraint matrix object constructed using 1-component FE object
    //
    dftUtils::constraintMatrixInfoCUDA * d_constraintMatrixData;
    //
    //matrix-free data
    //
    const dealii::MatrixFree<3,double> * d_matrix_free_data;

    thrust::device_vector<double> d_cellShapeFunctionGradientIntegralFlattenedDevice;


    thrust::device_vector<double> d_shapeFunctionValueDevice;
      
    thrust::device_vector<double> d_shapeFunctionValueInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueXDevice;
      
    thrust::device_vector<double> d_shapeFunctionGradientValueXInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueYDevice;
      
    thrust::device_vector<double> d_shapeFunctionGradientValueYInvertedDevice;

    thrust::device_vector<double> d_shapeFunctionGradientValueZDevice;
      
    thrust::device_vector<double> d_shapeFunctionGradientValueZInvertedDevice;

    thrust::device_vector<dealii::types::global_dof_index> d_flattenedArrayCellLocalProcIndexIdMapDevice;

    thrust::device_vector<dataTypes::number> d_cellWaveFunctionMatrix;  
    //
    //mpi communicator
    //
    MPI_Comm                          d_mpi_communicator;

    /// ScaLAPACK distributed format block size
    unsigned int d_scalapackBlockSize;
    
    /// ScaLAPACK distributed format block size for valence proj Ham
    unsigned int d_scalapackBlockSizeValence;
  };

/*--------------------- Inline functions --------------------------------*/
#  ifndef DOXYGEN
   inline unsigned int
   operatorDFTCUDAClass::getScalapackBlockSize() const
   {
     return d_scalapackBlockSize;
   }

   inline unsigned int
   operatorDFTCUDAClass::getScalapackBlockSizeValence() const
   {
     return d_scalapackBlockSizeValence;
   }
#  endif // ifndef DOXYGEN

}
#endif
