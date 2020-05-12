// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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

#ifndef poissonSolverProblem_H_
#define poissonSolverProblem_H_

namespace dftfe {

 /**
  * @brief poisson solver problem class template. template parameter FEOrder
  * is the finite element polynomial order
  *
  * @author Shiva Rudraraju, Phani Motamarri, Sambit Das
  */
  template<unsigned int FEOrder>
  class poissonSolverProblem: public dealiiLinearSolverProblem {

    public:

	/// Constructor
	poissonSolverProblem(const  MPI_Comm &mpi_comm);


	/**
	 * @brief reinitialize data structures for total electrostatic potential solve.
	 *
	 * For Hartree electrostatic potential solve give an empty map to the atoms parameter.
	 *
	 */
	 void reinit(const dealii::MatrixFree<3,double> & matrixFreeData,
		     distributedCPUVec<double> & x,
		     const dealii::ConstraintMatrix & constraintMatrix,
		     const unsigned int matrixFreeVectorComponent,
	             const std::map<dealii::types::global_dof_index, double> & atoms,
		     const std::map<dealii::CellId,std::vector<double> > & rhoValues,
		     const bool isComputeDiagonalA=true,
                     const bool isComputeMeanValueConstraints=false);

	/**
	 * @brief reinitialize data structures for nuclear electrostatic potential solve
	 *
	 */
	 void reinit(const dealii::MatrixFree<3,double> & matrixFreeData,
		     distributedCPUVec<double> & x,
		     const dealii::ConstraintMatrix & constraintMatrix,
		     const unsigned int matrixFreeVectorComponent,
	             const std::map<dealii::types::global_dof_index, double> & atoms,
		     const bool isComputeDiagonalA=true,
                     const bool isPrecomputeShapeGradIntegral=false);

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
        void AX (const dealii::MatrixFree<3,double>  &matrixFreeData,
	                  distributedCPUVec<double> &dst,
		          const distributedCPUVec<double> &src,
		          const std::pair<unsigned int,unsigned int> &cell_range) const;


	/**
	 * @brief Compute the diagonal of A.
	 *
	 */
	void computeDiagonalA();

	/**
	 * @brief Compute mean value constraint which is required in case of fully periodic
	 * boundary conditions.
	 *
	 */
	void computeMeanValueConstraint();


	/**
	 * @brief Mean value constraint distibute
	 *
	 */
	void meanValueConstraintDistribute(distributedCPUVec<double>& vec) const;

	/**
	 * @brief Mean value constraint distibute slave to master
	 *
	 */
	void meanValueConstraintDistributeSlaveToMaster(distributedCPUVec<double>& vec) const;


	/**
	 * @brief Mean value constraint set zero
	 *
	 */
	void meanValueConstraintSetZero(distributedCPUVec<double>& vec) const;

	/**
	 * @brief precompute shape function gradient integral.
	 *
	 */
	void precomputeShapeFunctionGradientIntegral();


	/// storage for diagonal of the A matrix
	distributedCPUVec<double> d_diagonalA;

	/// pointer to dealii MatrixFree object
        const dealii::MatrixFree<3,double>  * d_matrixFreeDataPtr;

	/// pointer to the x vector being solved for
        distributedCPUVec<double> * d_xPtr;

	/// pointer to dealii ConstraintMatrix object
        const dealii::ConstraintMatrix * d_constraintMatrixPtr;

	/// matrix free index required to access the DofHandler and ConstraintMatrix objects corresponding to the
	/// problem
        unsigned int d_matrixFreeVectorComponent;

	/// pointer to electron density cell quadrature data
	const std::map<dealii::CellId,std::vector<double> >* d_rhoValuesPtr;

	/// pointer to map between global dof index in current processor and the atomic charge on that dof
	const std::map<dealii::types::global_dof_index, double> * d_atomsPtr;

        /// shape function gradient integral storage
        std::vector<double> d_cellShapeFunctionGradientIntegralFlattened;

        bool d_isShapeGradIntegralPrecomputed;

	/// storage for mean value constraint vector
	distributedCPUVec<double> d_meanValueConstraintVec;

	/// boolean flag to query if mean value constraint datastructures are precomputed
	bool d_isMeanValueConstraintComputed;

	/// mean constrained nodeid
	dealii::types::global_dof_index d_meanValueConstraintNodeId;

        const MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
        dealii::ConditionalOStream   pcout;
  };

}
#endif // poissonSolverProblem_H_
