// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_MULTIVECTORPOISSONLINEARSOLVERPROBLEM_H
#define DFTFE_MULTIVECTORPOISSONLINEARSOLVERPROBLEM_H

#include "MultiVectorLinearSolverProblem.h"
#include "headers.h"
#include "BLASWrapper.h"
#include "FEBasisOperations.h"
namespace dftfe
{
  /*
   * @brief Extension of the poissonSolverProblem to multi Vector
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class MultiVectorPoissonLinearSolverProblem
    : public MultiVectorLinearSolverProblem<memorySpace>
  {
  public:
    // Constructor
    MultiVectorPoissonLinearSolverProblem(const MPI_Comm &mpi_comm_parent,
                                          const MPI_Comm &mpi_comm_domain);

    // Destructor
    ~MultiVectorPoissonLinearSolverProblem();

    /*
     * @brief reinit function to set up the internal variables
     * this function calls the computeDiagonal() and preComputeShapeFunction()
     */
    void
    reinit(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
             BLASWrapperPtr,
           std::shared_ptr<
             dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                                    basisOperationsPtr,
           const dealii::AffineConstraints<double> &constraintMatrix,
           const unsigned int                       matrixFreeVectorComponent,
           const unsigned int matrixFreeQuadratureComponentRhs,
           const unsigned int matrixFreeQuadratureComponentAX,
           bool               isComputeMeanValueConstraint);
    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &
    computeRhs(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &NDBCVec,
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &outputVec,
      unsigned int blockSizeInput) override;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    vmult(dftfe::linearAlgebra::MultiVector<double, memorySpace> &Ax,
          dftfe::linearAlgebra::MultiVector<double, memorySpace> &x,
          unsigned int blockSize) override;

    /**
     * @brief Apply the constraints to the solution vector.
     *
     */
    void
    distributeX() override;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    void
    precondition_Jacobi(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
      const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
      const double omega) const override;


    /**
     * @brief Apply square-root of the Jacobi preconditioner function.
     *
     */
    void
    precondition_JacobiSqrt(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
      const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
      const double omega) const override;

    /**
     * @brief function to set data for Rhs Vec.
     *  @param[in] inputQuadData the value of the right hand side at the quad
     * points
     */
    void
    setDataForRhsVec(
      dftfe::utils::MemoryStorage<double, memorySpace> &inputQuadData);


    void
    clear();

  private:
    void
    tempRhsVecCalc(dftfe::linearAlgebra::MultiVector<double, memorySpace> &rhs);

    void
    preComputeShapeFunction();

    void
    computeDiagonalA();

    void
    computeMeanValueConstraint();

    bool d_isComputeDiagonalA;

    bool d_isMeanValueConstraintComputed;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    /// the vector that stores the output obtained by solving the poisson
    /// problem
    dftfe::linearAlgebra::MultiVector<double, memorySpace> *d_blockedXPtr,
      *d_blockedNDBCPtr, d_rhsVec;

    unsigned int d_matrixFreeQuadratureComponentRhs;
    unsigned int d_matrixFreeVectorComponent;
    unsigned int d_blockSize;

    dftfe::utils::MemoryStorage<double, memorySpace> d_diagonalA,
      d_diagonalSqrtA;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>>
      d_basisOperationsPtr;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    const dftfe::utils::MemoryStorage<double, memorySpace>
      *d_cellStiffnessMatrixPtr;

    dftUtils::constraintMatrixInfo<memorySpace> d_constraintsInfo;

    /// data members for the mpi implementation
    const MPI_Comm             mpi_communicator, d_mpi_parent;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
    size_type                  d_locallyOwnedSize;

    size_type d_numberDofsPerElement;
    size_type d_numCells;

    size_type d_inc;
    double    d_negScalarCoeffAlpha;
    double    d_scalarCoeffAlpha;
    double    d_beta;
    double    d_alpha;
    char      d_transA;
    char      d_transB;


    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapNodeIdToProcId, d_mapQuadIdToProcId;

    dftfe::utils::MemoryStorage<double, memorySpace> d_xCellLLevelNodalData,
      d_AxCellLLevelNodalData;


    dftfe::utils::MemoryStorage<double, memorySpace> *d_rhsQuadDataPtr;

    unsigned int d_matrixFreeQuadratureComponentAX, d_nQuadsPerCell;

    /// pointer to the dealii::DofHandler object. This is already part of the
    /// matrixFreeData object.
    const dealii::DoFHandler<3> *d_dofHandler;

    /**
     * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(\nabla N_i \cdot \nabla N_j))
     * with first dimension traversing the macro cell id
     * and second dimension storing the matrix of size numberNodesPerElement x
     * numberNodesPerElement in a flattened 1D dealii Vectorized array
     */
    std::vector<double> d_cellShapeFunctionGradientIntegral;

    /**
     * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(\nabla N_i ))
     * with first dimension traversing the macro cell id
     * and second dimension storing the matrix of size numberNodesPerElement in
     * a flattened 1D dealii Vectorized array
     */
    std::vector<double> d_cellShapeFunctionJxW;

    /// storage for shapefunctions
    std::vector<double> d_shapeFunctionValue;

    unsigned int d_cellBlockSize;
  };

} // namespace dftfe


#endif // DFTFE_MULTIVECTORPOISSONLINEARSOLVERPROBLEM_H
