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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef poissonSolverProblemDevice_H_
#    define poissonSolverProblemDevice_H_

#    include <linearSolverProblemDevice.h>
#    include <constraintMatrixInfoDevice.h>
#    include <constraintMatrixInfo.h>
#    include <constants.h>
#    include <dftUtils.h>
#    include <headers.h>
#    include "FEBasisOperations.h"
#    include "BLASWrapper.h"

namespace dftfe
{
  /**
   * @brief poisson solver problem device class template. template parameter FEOrderElectro
   * is the finite element polynomial order. FEOrder template parameter is used
   * in conjunction with FEOrderElectro to determine the order of the Gauss
   * quadrature rule. The class should not be used with FLOATING NUCLEAR
   * CHARGES = false or POINT WISE DIRICHLET CONSTRAINT = true
   *
   * @author Gourab Panigrahi
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class poissonSolverProblemDevice : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    poissonSolverProblemDevice(const MPI_Comm &mpi_comm);

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
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                                      basisOperationsPtr,
      distributedCPUVec<double> &              x,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const unsigned int                       matrixFreeVectorComponent,
      const unsigned int matrixFreeQuadratureComponentRhsDensity,
      const unsigned int matrixFreeQuadratureComponentAX,
      const std::map<dealii::types::global_dof_index, double> &atoms,
      const std::map<dealii::CellId, std::vector<double>> &smearedChargeValues,
      const unsigned int smearedChargeQuadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoValues,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                         BLASWrapperPtr,
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
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x);

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    distributedDeviceVec<double> &
    getX();

    /**
     * @brief get the reference to Preconditioner
     *
     * @return reference to Preconditioner
     */
    distributedDeviceVec<double> &
    getPreconditioner();

    /**
     * @brief Copies x from Device to Host
     *
     */
    void
    copyXfromDeviceToHost();

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


  private:
    /**
     * @brief Sets up the matrixfree shapefunction, gradient, jacobian and map for matrixfree computeAX
     *
     */
    void
    setupMatrixFree();

    /**
     * @brief Sets up the constraints matrix
     *
     */
    void
    setupconstraints();

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
    meanValueConstraintDistribute(distributedDeviceVec<double> &vec) const;

    /**
     * @brief Mean value constraint distibute slave to master
     *
     */
    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedDeviceVec<double> &vec) const;

    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedCPUVec<double> &vec) const;

    /**
     * @brief Mean value constraint set zero
     *
     */
    void
    meanValueConstraintSetZero(distributedDeviceVec<double> &vec) const;

    void
    meanValueConstraintSetZero(distributedCPUVec<double> &vec) const;

    /// storage for diagonal of the A matrix
    distributedCPUVec<double>    d_diagonalA;
    distributedDeviceVec<double> d_diagonalAdevice;

    /// storage for smeared charge rhs in case of total potential solve (doesn't
    /// change every scf)
    distributedCPUVec<double> d_rhsSmearedCharge;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to the x vector being solved for
    distributedCPUVec<double> *  d_xPtr;
    distributedDeviceVec<double> d_xDevice;

    // number of cells local to each mpi task, number of degrees of freedom
    // locally owned and total degrees of freedom including ghost
    int d_nLocalCells, d_xLocalDof, d_xLen;

    // shape function value, gradient, jacobian and map for matrixfree
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                                                                        d_shapeFunction, d_jacobianFactor;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE> d_map;

    // Pointers to shape function value, gradient, jacobian and map for
    // matrixfree
    double *d_shapeFunctionPtr;
    double *d_jacobianFactorPtr;
    int *   d_mapPtr;


    // constraints
    dftUtils::constraintMatrixInfoDevice d_constraintsTotalPotentialInfo;

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
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      *d_rhoValuesPtr;

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

    /// storage for mean value constraint device vector
    distributedDeviceVec<double> d_meanValueConstraintDeviceVec;

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
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtr;
    ///
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
         d_BLASWrapperPtr;
    bool d_isFastConstraintsInitialized;

    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#  endif // poissonSolverProblemDevice_H_
#endif
