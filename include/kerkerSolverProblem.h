// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020  The Regents of the University of Michigan and DFT-FE authors.
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


#include <dealiiLinearSolverProblem.h>
#include <triangulationManager.h>

#ifndef kerkerSolverProblem_H_
#define kerkerSolverProblem_H_

namespace dftfe {

  /**
   * @brief poisson solver problem class template. template parameter FEOrder
   * is the finite element polynomial order
   *
   * @author Phani Motamarri
   */
  template<unsigned int FEOrder>
    class kerkerSolverProblem: public dealiiLinearSolverProblem {

  public:

    /// Constructor
    kerkerSolverProblem(const  MPI_Comm &mpi_comm);


    

    /**
     * @brief initialize the matrix-free data structures
     *
     * @param matrixFreeData structure to hold quadrature rule, constraints vector and appropriate dofHandler
     * @param constraintMatrix to hold constraints in the given problem     
     * @param x vector to be initialized using matrix-free object
     *
     */
    void init(dealii::MatrixFree<3,double> & matrixFreeData,
	      dealii::ConstraintMatrix & constraintMatrix,
	      distributedCPUVec<double> & x,
	      double kerkerMixingParameter);


    /**
     * @brief reinitialize data structures .
     *
     * @param x vector to store initial guess and solution
     * @param gradResidualValues stores the gradient of difference of input electron-density and output electron-density
     * @param kerkerMixingParameter used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k_{TF}**2).
     *
     */
    void reinit(distributedCPUVec<double> & x,
		const std::map<dealii::CellId,std::vector<double> > & gradResidualValues);


    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedCPUVec<double> & getX();

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void vmult(distributedCPUVec<double> &Ax,
	       const distributedCPUVec<double> &x) const;

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void computeRhs(distributedCPUVec<double> & rhs);

    /**
     * @brief Jacobi preconditioning.
     *
     */
    void precondition_Jacobi(distributedCPUVec<double>& dst,
			     const distributedCPUVec<double>& src,
			     const double omega) const;

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    void distributeX();


    /// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
    void subscribe (std::atomic< bool > *const validity, const std::string &identifier="") const{};

    /// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
    void unsubscribe (std::atomic< bool > *const validity, const std::string &identifier="") const{};

    /// function needed by dealii to mimic SparseMatrix
    bool operator!= (double val) const {return true;};

  private:

    /**
     * @brief required for the cell_loop operation in dealii's MatrixFree class
     *
     */
    void AX (const dealii::MatrixFree<3,double> & matrixFreeData,
	     distributedCPUVec<double> &dst,
	     const distributedCPUVec<double> &src,
	     const std::pair<unsigned int,unsigned int> &cell_range) const;


    /**
     * @brief Compute the diagonal of A.
     *
     */
    void computeDiagonalA();


    /// storage for diagonal of the A matrix
    distributedCPUVec<double> d_diagonalA;

   
    /// pointer to the x vector being solved for
    distributedCPUVec<double> * d_xPtr;

    //kerker mixing constant
    double d_gamma;

    /// pointer to electron density cell and grad residual data
    const std::map<dealii::CellId,std::vector<double> >* d_quadGradResidualValuesPtr;
    const dealii::DoFHandler<3> * d_dofHandlerPRefinedPtr;
    const dealii::ConstraintMatrix * d_constraintMatrixPRefinedPtr;
    const dealii::MatrixFree<3,double> * d_matrixFreeDataPRefinedPtr;

    const MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    dealii::ConditionalOStream   pcout;
  };

}
#endif // kerkerSolverProblem_H_
