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

#if defined(DFTFE_WITH_GPU)
#  ifndef linearSolverProblemCUDA_H_
#    define linearSolverProblemCUDA_H_

#    include <headers.h>
#    include <thrust/device_vector.h>
#    include <thrust/host_vector.h>

namespace dftfe
{
  /**
   * @brief Abstract class for linear solver problems to be used with the linearSolverCGCUDA interface.
   *
   * @author Phani Motamarri, Sambit Das, Gourab Panigrahi
   */
  class linearSolverProblemCUDA
  {
  public:
    /**
     * @brief Constructor.
     */
    linearSolverProblemCUDA();

    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    virtual distributedGPUVec<double> &
    getX() = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    computeAX(distributedGPUVec<double> &dst,
              distributedGPUVec<double> &src) = 0;

    /**
     * @brief Combines the precondition_Jacobi and various functions in CG
     *
     */
    virtual double
    cg(double *hvec, double *gvec) = 0;

    /**
     * @brief Combines the precondition_Jacobi and various functions in CG
     *
     */
    virtual double
    cg2(double *hvec, double *gvec, double *dvec) = 0;

    /**
     * @brief Combines the precondition_Jacobi and various functions in CG
     *
     */
    virtual double
    cg3(double *hvec, double *gvec, double *dvec, double &alpha) = 0;

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    virtual void
    computeRhs(distributedCPUVec<double> &rhs) = 0;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    virtual void
    precondition_Jacobi(distributedGPUVec<double> &      dst,
                        const distributedGPUVec<double> &src) const = 0;


    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    virtual void
    setX() = 0;

    virtual void
    distributeX() = 0;

    virtual void
    copyCUDAToHost() = 0;

    // protected:

    /// typedef declaration needed by dealii
    typedef dealii::types::global_dof_index size_type;
  };

} // namespace dftfe
#  endif // linearSolverProblemCUDA_H_
#endif
