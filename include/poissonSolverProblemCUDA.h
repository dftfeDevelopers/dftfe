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
#  ifndef poissonSolverProblemCUDA_H_
#    define poissonSolverProblemCUDA_H_

#    include <linearSolverProblemCUDA.h>
#    include <constraintMatrixInfoCUDA.h>
#    include <constraintMatrixInfo.h>
#    include <constants.h>
#    include <cudaHelpers.h>
#    include <dftUtils.h>
#    include <headers.h>

namespace dftfe
{
  /**
   * @brief poisson solver problem cuda class template. template parameter FEOrderElectro
   * is the finite element polynomial order. FEOrder template parameter is used
   * in conjunction with FEOrderElectro to determine the order of the Gauss
   * quadrature rule. The class should not be used with FLOATING NUCLEAR
   * CHARGES = false or POINT WISE DIRICHLET CONSTRAINT = true
   *
   * @author Gourab Panigrahi
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class poissonSolverProblemCUDA : public linearSolverProblemCUDA
  {
  public:
    /// Constructor
    poissonSolverProblemCUDA(const MPI_Comm &mpi_comm);


    /**
     * @brief clears all datamembers and reset to original state.
     *
     *
     */
    void
    clear();


    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(
      const dealii::MatrixFree<3, double> &    matrixFreeData,
      distributedCPUVec<double> &              x,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const unsigned int                       matrixFreeVectorComponent,
      const unsigned int matrixFreeQuadratureComponentRhsDensity,
      const unsigned int matrixFreeQuadratureComponentAX,
      const std::map<dealii::types::global_dof_index, double> &atoms,
      const std::map<dealii::CellId, std::vector<double>> &smearedChargeValues,
      const unsigned int smearedChargeQuadratureId,
      const std::map<dealii::CellId, std::vector<double>> &rhoValues,
      cublasHandle_t &                                     cublasHandle,
      const bool         isComputeDiagonalA               = true,
      const bool         isComputeMeanValueConstraints    = false,
      const bool         smearedNuclearCharges            = false,
      const bool         isRhoValues                      = true,
      const bool         isGradSmearedChargeRhs           = false,
      const unsigned int smearedChargeGradientComponentId = 0,
      const bool         storeSmearedChargeRhs            = false,
      const bool         reuseSmearedChargeRhs            = false,
      const bool         reinitializeFastConstraints      = false);


    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedGPUVec<double> &
    getX();

    /**
     * @brief get the reference to Preconditioner
     *
     * @return reference to Preconditioner
     */
    distributedGPUVec<double> &
    getPreconditioner();

    /**
     * @brief Sets up the matrixfree shapefunction, gradient, weights, jacobian and map for matrixfree computeAX
     *
     */
    void
    setupMatrixFree();

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(distributedGPUVec<double> &Ax, distributedGPUVec<double> &x);

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

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

    void
    setX();

    /**
     * @brief Copies x from Device to Host
     *
     */
    void
    copyXfromDeviceToHost();

  private:
    /**
     * @brief Compute the diagonal of A.
     *
     */
    void
    computeDiagonalA();

    /**
     * @brief Compute mean value constraint which is required in case of fully periodic
     * boundary conditions.
     *
     */
    void
    computeMeanValueConstraint();

    /**
     * @brief Mean value constraint distibute
     *
     */
    void
    meanValueConstraintDistribute(distributedGPUVec<double> &vec) const;

    /**
     * @brief Mean value constraint distibute slave to master
     *
     */
    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedGPUVec<double> &vec) const;

    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedCPUVec<double> &vec) const;

    /**
     * @brief Mean value constraint set zero
     *
     */
    void
    meanValueConstraintSetZero(distributedGPUVec<double> &vec) const;

    void
    meanValueConstraintSetZero(distributedCPUVec<double> &vec) const;

    /// storage for diagonal of the A matrix
    distributedCPUVec<double> d_diagonalA;
    distributedGPUVec<double> d_diagonalAdevice;

    /// storage for smeared charge rhs in case of total potential solve (doesn't
    /// change every scf)
    distributedCPUVec<double> d_rhsSmearedCharge;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to the x vector being solved for
    distributedCPUVec<double> *d_xPtr;
    distributedGPUVec<double>  d_xDevice;

    // shape function value, gradient, weights, jacobian and map for matrixfree
    thrust::device_vector<double> d_shapeFunctionValue, d_shapeFunctionGradient,
      d_jacobianAction;
    thrust::device_vector<int> d_map;

    // cuBLAS handle for cuBLAS operations
    cublasHandle_t *d_cublasHandlePtr;

    // constraints
    dftUtils::constraintMatrixInfoCUDA constraintsTotalPotentialInfo;

    // number of cells local to each mpi task, number of degrees of freedom
    // locally owned and total degrees of freedom including ghost
    int d_nLocalCells, d_xLenLocalDof, d_xLen;

    // Pointers to shape function value, gradient, weights, jacobian and map for
    // matrixfree on device
    double *shapeFunctionValuePtr;
    double *shapeFunctionGradientPtr;
    double *jacobianActionPtr;
    int *   mapPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    /// matrix free index required to access the DofHandler and
    /// dealii::AffineConstraints<double> objects corresponding to the problem
    unsigned int d_matrixFreeVectorComponent;

    /// matrix free quadrature index
    unsigned int d_matrixFreeQuadratureComponentRhsDensity;

    /// matrix free quadrature index
    unsigned int d_matrixFreeQuadratureComponentAX;

    /// pointer to electron density cell quadrature data
    const std::map<dealii::CellId, std::vector<double>> *d_rhoValuesPtr;

    /// pointer to smeared charge cell quadrature data
    const std::map<dealii::CellId, std::vector<double>>
      *d_smearedChargeValuesPtr;

    ///
    unsigned int d_smearedChargeQuadratureId;

    /// pointer to map between global dof index in current processor and the
    /// atomic charge on that dof
    const std::map<dealii::types::global_dof_index, double> *d_atomsPtr;

    /// shape function gradient integral storage
    std::vector<double> d_cellShapeFunctionGradientIntegralFlattened;

    /// storage for mean value constraint vector
    distributedCPUVec<double> d_meanValueConstraintVec;

    /// storage for mean value constraint gpu vector
    distributedGPUVec<double> d_meanValueConstraintGPUVec;

    /// boolean flag to query if mean value constraint datastructures are
    /// precomputed
    bool d_isMeanValueConstraintComputed;

    ///
    bool d_isGradSmearedChargeRhs;

    ///
    bool d_isStoreSmearedChargeRhs;

    ///
    bool d_isReuseSmearedChargeRhs;

    ///
    unsigned int d_smearedChargeGradientComponentId;

    /// mean value constraints: mean value constrained node
    dealii::types::global_dof_index d_meanValueConstraintNodeId;

    /// mean value constrained node local id
    dealii::types::global_dof_index d_meanValueConstraintNodeIdLocal;

    /// mean value constraints: constrained proc id containing the mean value
    /// constrained node
    unsigned int d_meanValueConstraintProcId;

    /// duplicate constraints object with flattened maps for faster access
    dftUtils::constraintMatrixInfo d_constraintsInfo;

    ///
    bool d_isFastConstraintsInitialized;

    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#  endif // poissonSolverProblemCUDA_H_
#endif