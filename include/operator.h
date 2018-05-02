//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri 
//
#ifndef operatorDFTClass_h
#define operatorDFTClass_h

#include <vector>

#include <headers.h>
#include <constraintMatrixInfo.h>

typedef dealii::parallel::distributed::Vector<double> vectorType;

namespace dftfe{
  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   */
  class operatorDFTClass {

    //
    // methods
    //
  public:

    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTClass() = 0;


    /**
     * @brief initialize operatorClass
     *
     */
    virtual void init() = 0;


    /**
     * @brief compute diagonal mass matrix
     *
     * @param dofHandler dofHandler associated with the current mesh
     * @param constraintMatrix constraints to be used
     * @param sqrtMassVec output the value of square root of diagonal mass matrix 
     * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
     */
    virtual void computeMassVector(const dealii::DoFHandler<3>    & dofHandler,
				   const dealii::ConstraintMatrix & constraintMatrix,
				   vectorType                     & sqrtMassVec,
				   vectorType                     & invSqrtMassVec) = 0;


    /**
     * @brief Compute operator times vector or operator times bunch of vectors
     *
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
     * @param macroCellMap precomputed cell-local index id map of the multi-wavefuncton field
     * @param cellMap precomputed cell-local index id map of the multi-wavefunction field
     * @param Y Vector containing multi-component fields after operator times vectors product
     */
#ifdef ENABLE_PERIODIC_BC
    virtual void HX(dealii::parallel::distributed::Vector<std::complex<double> > & X,
		    const unsigned int numberComponents,
		    const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
		    const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
		    dealii::parallel::distributed::Vector<std::complex<double> > & Y) = 0;
#else
    virtual void HX(dealii::parallel::distributed::Vector<double> & X,
		    const unsigned int numberComponents,
		    const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
		    const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
		    dealii::parallel::distributed::Vector<double> & Y) = 0;
#endif


    /**
     * @brief Compute projection of the operator into orthogonal basis
     *
     * @param X given orthogonal basis vectors
     * @return ProjMatrix projected small matrix 
     */
#ifdef ENABLE_PERIODIC_BC
    virtual void XtHX(std::vector<vectorType> & X,
		      std::vector<std::complex<double> > & ProjHam) = 0;
#else
    virtual void XtHX(std::vector<vectorType> & X,
		      std::vector<double> & ProjHam) = 0;
#endif


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
    const dealii::ConstraintMatrix * getConstraintMatrixEigen() const;


    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    dftUtils::constraintMatrixInfo * getOverloadedConstraintMatrix() const;

    
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
    operatorDFTClass();


    /**
     * @brief Constructor.
     */
    operatorDFTClass(const MPI_Comm & mpi_comm_replica,
		     const dealii::MatrixFree<3,double> & matrix_free_data,
		     const std::vector<dealii::types::global_dof_index> & localDofIndicesReal,
		     const std::vector<dealii::types::global_dof_index> & localDofIndicesImag,
		     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
		     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
		     const dealii::ConstraintMatrix  & constraintMatrixEigen,
		     dftUtils::constraintMatrixInfo & constraintMatrixNone);

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
    const dealii::ConstraintMatrix  * d_constraintMatrixEigen;

    //
    //Get overloaded constraint matrix object constructed using 1-component FE object
    //
    dftUtils::constraintMatrixInfo * d_constraintMatrixData;
    //
    //matrix-free data
    //
    const dealii::MatrixFree<3,double> * d_matrix_free_data;

    //
    //mpi communicator
    //
    MPI_Comm                          d_mpi_communicator;
  };

}
#endif
