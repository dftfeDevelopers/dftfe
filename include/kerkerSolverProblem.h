// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020  The Regents of the University of Michigan and DFT-FE
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


#include <dealiiLinearSolverProblem.h>
#include <triangulationManager.h>
#include <FEBasisOperations.h>
#ifndef kerkerSolverProblem_H_
#  define kerkerSolverProblem_H_

namespace dftfe
{
  /**
   * @brief poisson solver problem class template. template parameter FEOrderElectro
   * is the finite element polynomial order for electrostatics
   *
   * @author Phani Motamarri
   */
  template <unsigned int FEOrderElectro>
  class kerkerSolverProblem : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    kerkerSolverProblem(const MPI_Comm &mpi_comm_parent,
                        const MPI_Comm &mpi_comm_domain);



    /**
     * @brief initialize the matrix-free data structures
     *
     * @param matrixFreeData structure to hold quadrature rule, constraints vector and appropriate dofHandler
     * @param constraintMatrix to hold constraints in the given problem
     * @param x vector to be initialized using matrix-free object
     *
     */
    void
    init(std::shared_ptr<
           dftfe::basis::
             FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
           &                                basisOperationsPtr,
         dealii::AffineConstraints<double> &constraintMatrix,
         distributedCPUVec<double> &        x,
         double                             kerkerMixingParameter,
         const unsigned int                 matrixFreeVectorComponent,
         const unsigned int                 matrixFreeQuadratureComponent);



    /**
     * @brief reinitialize data structures .
     *
     * @param x vector to store initial guess and solution
     * @param gradResidualValues stores the gradient of difference of input electron-density and output electron-density
     * @param kerkerMixingParameter used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k_{TF}**2).
     *
     */
    void
    reinit(
      distributedCPUVec<double> &x,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadPointValues);


    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedCPUVec<double> &
    getX();

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x);

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief Jacobi preconditioning.
     *
     */
    void
    precondition_Jacobi(distributedCPUVec<double> &      dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const;

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    void
    distributeX();


    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    void
    subscribe(std::atomic<bool> *const validity,
              const std::string &      identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    void
    unsubscribe(std::atomic<bool> *const validity,
                const std::string &      identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix
    bool
    operator!=(double val) const
    {
      return true;
    };

  private:
    /**
     * @brief required for the cell_loop operation in dealii's MatrixFree class
     *
     */
    void
    AX(const dealii::MatrixFree<3, double> &        matrixFreeData,
       distributedCPUVec<double> &                  dst,
       const distributedCPUVec<double> &            src,
       const std::pair<unsigned int, unsigned int> &cell_range) const;


    /**
     * @brief Compute the diagonal of A.
     *
     */
    void
    computeDiagonalA();


    /// storage for diagonal of the A matrix
    distributedCPUVec<double> d_diagonalA;


    /// pointer to the x vector being solved for
    distributedCPUVec<double> *d_xPtr;

    // kerker mixing constant
    double d_gamma;

    /// matrix free index required to access the DofHandler and
    /// dealii::AffineConstraints<double> objects corresponding to the problem
    unsigned int d_matrixFreeVectorComponent;

    /// matrix free quadrature index
    unsigned int d_matrixFreeQuadratureComponent;


    /// pointer to electron density cell and grad residual data
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      *                                      d_residualQuadValuesPtr;
    const dealii::DoFHandler<3> *            d_dofHandlerPRefinedPtr;
    const dealii::AffineConstraints<double> *d_constraintMatrixPRefinedPtr;
    const dealii::MatrixFree<3, double> *    d_matrixFreeDataPRefinedPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtr;


    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // kerkerSolverProblem_H_
