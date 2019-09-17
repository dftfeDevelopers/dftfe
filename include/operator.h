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
// @author Phani Motamarri
//
#ifndef operatorDFTClass_h
#define operatorDFTClass_h

#include <vector>

#include <headers.h>
#include <constraintMatrixInfo.h>
#ifdef DFTFE_WITH_ELPA
extern "C"
{
#include <elpa/elpa.h>
}
#endif

namespace dftfe{

  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   *
   * @author Phani Motamarri
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

#ifdef DEAL_II_WITH_SCALAPACK
    unsigned int getScalapackBlockSize() const;

    unsigned int getScalapackBlockSizeValence() const;

    void processGridOptionalELPASetup(const unsigned int na,
    		                      const unsigned int nev);

#ifdef DFTFE_WITH_ELPA
    void elpaDeallocateHandles(const unsigned int na,
		    const unsigned int nev);

    elpa_t & getElpaHandle();

    elpa_t & getElpaHandlePartialEigenVec();

    elpa_t & getElpaHandleValence();

    elpa_autotune_t & getElpaAutoTuneHandle();
#endif

#endif

    /**
     * @brief initialize operatorClass
     *
     */
    virtual void init() = 0;

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
    virtual void reinit(const unsigned int wavefunBlockSize,
			dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & X,
			bool flag) = 0;

    virtual void reinit(const unsigned int wavefunBlockSize) = 0;

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
     * @param Y Vector containing multi-component fields after operator times vectors product
     */
    virtual void HX(dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & X,
		    const unsigned int numberComponents,
		    const bool scaleFlag,
		    const double scalar,
		    dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & Y) = 0;


    virtual void MX(dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & X,
		    const unsigned int numberComponents,
		    dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & Y) = 0;

    
    virtual void HX(dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & X,
		    const unsigned int numberComponents,
		    dealii::LinearAlgebra::distributed::Vector<dataTypes::number> & Y) = 0;


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param ProjMatrix projected small matrix
     */
    virtual void XtHX(const std::vector<dataTypes::number> & X,
		      const unsigned int numberComponents,
		      std::vector<dataTypes::number> & ProjHam) = 0;

#ifdef DEAL_II_WITH_SCALAPACK
    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void XtHX(const std::vector<dataTypes::number> & X,
		      const unsigned int numberComponents,
		      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		      dealii::ScaLAPACKMatrix<dataTypes::number> & projHamPar,
		      bool origHFlag=false) = 0;

    
    virtual void XtMX(const std::vector<dataTypes::number> & X,
		      const unsigned int numberComponents,
		      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		      dealii::ScaLAPACKMatrix<dataTypes::number> & projMassPar) = 0;

    


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param totalNumberComponents number of wavefunctions associated with a given node
     * @param singlePrecComponents number of wavecfuntions starting from the first for
     * which the project Hamiltionian block will be computed in single procession. However
     * the cross blocks will still be computed in double precision.
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void XtHXMixedPrec
	             (const std::vector<dataTypes::number> & X,
		      const unsigned int totalNumberComponents,
		      const unsigned int singlePrecComponents,
		      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		      dealii::ScaLAPACKMatrix<dataTypes::number> & projHamPar,
		      bool origHFlag=false) = 0;


#endif
    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis
     *
     * @param  X Vector of Vectors containing the basis vectors spanning the subspace
     * @return ProjMatrix projected small matrix
     */
    virtual void XtHX(std::vector<vectorType> & X,
		      std::vector<dataTypes::number> & ProjHam) = 0;


    void setInvSqrtMassVector(vectorType & X);
    vectorType & getInvSqrtMassVector();

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
    //inv sqrt mass vector
    //
    vectorType d_invSqrtMassVector;

    //
    //mpi communicator
    //
    MPI_Comm                          d_mpi_communicator;

#ifdef DEAL_II_WITH_SCALAPACK
#ifdef DFTFE_WITH_ELPA
    /// ELPA handle
    elpa_t d_elpaHandle;

    /// ELPA handle for valence proj Ham
    elpa_t d_elpaHandleValence;

    /// ELPA handle for partial eigenvectors of full proj ham
    elpa_t d_elpaHandlePartialEigenVec;

    /// ELPA autotune handle
    elpa_autotune_t d_elpaAutoTuneHandle;

    /// processGrid mpi communicator
    MPI_Comm d_processGridCommunicatorActive;

    MPI_Comm d_processGridCommunicatorActivePartial;
     
    MPI_Comm d_processGridCommunicatorActiveValence;
#endif

    /// ScaLAPACK distributed format block size
    unsigned int d_scalapackBlockSize;

    /// ScaLAPACK distributed format block size for valence proj Ham
    unsigned int d_scalapackBlockSizeValence;
#endif
  };

/*--------------------- Inline functions --------------------------------*/

#  ifndef DOXYGEN
#ifdef DEAL_II_WITH_SCALAPACK
   inline unsigned int
   operatorDFTClass::getScalapackBlockSize() const
   {
     return d_scalapackBlockSize;
   }

   inline unsigned int
   operatorDFTClass::getScalapackBlockSizeValence() const
   {
     return d_scalapackBlockSizeValence;
   }
#ifdef DFTFE_WITH_ELPA
  inline
  elpa_t & operatorDFTClass::getElpaHandle()
  {
       return d_elpaHandle;
  }

  inline
  elpa_t & operatorDFTClass::getElpaHandleValence()
  {
       return d_elpaHandleValence;
  }

  inline
  elpa_t & operatorDFTClass::getElpaHandlePartialEigenVec()
  {
       return d_elpaHandlePartialEigenVec;
  }


  inline
  elpa_autotune_t & operatorDFTClass::getElpaAutoTuneHandle()
  {
       return d_elpaAutoTuneHandle;
  }
#endif
#endif
#  endif // ifndef DOXYGEN

}
#endif
