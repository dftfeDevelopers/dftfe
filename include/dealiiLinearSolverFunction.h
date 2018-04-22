// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//

/**
* @brief Abstract class for linear solver functions.
*
* @author Sambit Das
*/

#include <headers.h>

#ifndef dealiiLinearSolverFunction_H_
#define dealiiLinearSolverFunction_H_

namespace dftfe {

  typedef dealii::parallel::distributed::Vector<double> vectorType;

  class dealiiLinearSolverFunction {

     public:
        /**
         * @brief Constructor.
         */
        dealiiLinearSolverFunction();

	/**
	 * @brief get the reference to x field
	 *
	 * @return reference to x field. Assumes x field data structure is already initialized
	 */
	virtual vectorType & getX() = 0;

	/**
	 * @brief Compute A matrix multipled by x.
	 *
	 */
	virtual void vmult(vectorType &Ax,
			   const vectorType &x) const= 0;

	/**
	 * @brief Compute right hand side vector for the problem Ax = rhs.
	 *
	 * @param rhs vector for the right hand side values
	 */
	virtual void computeRhs(vectorType & rhs) = 0;

	/**
	 * @brief Jacobi preconditioning function.
	 *
	 */
        virtual void precondition_Jacobi(vectorType& dst,
		                         const vectorType& src,
				         const double omega) const=0;

	/**
	 * @brief distribute x to the constrained nodes.
	 *
	 */
	virtual void distributeX() = 0;

	// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
        virtual void subscribe (const char *identifier=0) const=0;

	// function needed by dealii to mimic SparseMatrix for Jacobi preconditioning
        virtual void unsubscribe (const char *identifier=0) const=0;

	// function needed by dealii to mimic SparseMatrix
        virtual bool operator!= (double val) const =0;

     protected:

	 // typedef declaration needed by dealii
	 typedef dealii::types::global_dof_index size_type;
    };

}
#endif // dealiiLinearSolverFunction_H_
