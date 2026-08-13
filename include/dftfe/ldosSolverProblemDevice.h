#if defined(DFTFE_WITH_DEVICE)

#  ifndef ldosSolverProblemDevice_H_
#    define ldosSolverProblemDevice_H_

#    include <dftfe/linearSolverProblemDevice.h>
#    include <dftfe/triangulationManager.h>
#    include <dftfe/constraintMatrixInfo.h>
#    include <dftfe/MemoryStorage.h>
#    include <dftfe/dftUtils.h>
#    include <dftfe/FEBasisOperations.h>
#    include "dftfe/BLASWrapper.h"
#    include <dftfe/DeviceAPICalls.h>
#    include "dftfe/MatrixFreeWrapper.h"

namespace dftfe
{
  template <dftfe::uInt FEOrderElectro>
  class ldosSolverProblemDevice : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    ldosSolverProblemDevice(const MPI_Comm &mpi_comm_parent,
                            const MPI_Comm &mpi_comm_domain);
    void
    init(std::shared_ptr<
           dftfe::basis::
             FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                           &basisOperationsPtr,
         dealii::AffineConstraints<double> &constraintMatrix,
         distributedCPUVec<double>         &x,
         const dftfe::uInt                  matrixFreeVectorComponent,
         const dftfe::uInt                  matrixFreeQuadratureComponent,
         const dftfe::uInt                  matrixFreeAxQuadratureComponent,
         const bool                         isComputeMeanValueConstraint);

    /**
     * @brief Reinitialize for a new SCF iteration.
     */
    void
    reinit(
      distributedCPUVec<double> &x,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &residualQuadValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &ldosAxQuadValues);

    /**
     * @brief Return the device x vector.
     */
    distributedDeviceVec<double> &
    getX();

    /**
     * @brief Return the Jacobi preconditioner on device.
     */
    distributedDeviceVec<double> &
    getPreconditioner();

    /**
     * @brief Apply the LDOS operator on device.
     */
    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x);

    /**
     * @brief Not implemented.
     */
    void
    setX();

    /**
     * @brief Assemble the RHS on HOST (-residual projected to nodes).
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief Distribute x to constrained nodes (device).
     */
    void
    distributeX();

    /**
     * @brief Copy device x back to HOST x.
     */
    void
    copyXfromDeviceToHost();

    /**
     * @brief Set BLASWrapper pointer
     */
    void
    setBLASWrapperPtr(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                        dftfe::utils::MemorySpace::DEVICE>> blasWrapperPtr)
    {
      d_BLASWrapperPtr = blasWrapperPtr;
    }

    /**
     * @brief Compute integral of D_loc * phi (HOST dot product).
     *        Uses d_dlocMassVector which is set in reinit().
     */
    double
    computeDlocIntegral(const distributedCPUVec<double> &phi) const
    {
      return d_dlocMassVector * phi;
    }

    double
    getTotalDOS() const
    {
      return d_totalDOS;
    }

  private:
    void
    computeMeanValueConstraint();

    void
    meanValueConstraintDistribute(distributedDeviceVec<double> &vec) const;

    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedDeviceVec<double> &vec) const;

    void
    meanValueConstraintDistributeSlaveToMaster(
      distributedCPUVec<double> &vec) const;

    void
    meanValueConstraintSetZero(distributedCPUVec<double> &vec) const;

    /**
     * @brief Project quad-point values to nodal field (HOST).
     */
    void
    computeProjectedQuadToNodalField(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                &quadValues,
      distributedCPUVec<double> &nodalField);

    /**
     * @brief Set up device-side constraint info.
     */
    void
    setupConstraints();

    /**
     * @brief Compute the Jacobi preconditioner diagonal (HOST + device transfer).
     */
    void
    computeDiagonalA();

    /// HOST diagonal , device diagonal
    distributedCPUVec<double>    d_diagonalA;
    distributedDeviceVec<double> d_diagonalAdevice;

    distributedCPUVec<double>    d_dlocMassVector;
    distributedDeviceVec<double> d_dlocMassVectorDevice;
    distributedCPUVec<double>    d_meanValueConstraintVec;
    distributedDeviceVec<double> d_meanValueConstraintDeviceVec;

    bool                            d_isMeanValueConstraintComputed;
    dealii::types::global_dof_index d_meanValueConstraintNodeId;
    dealii::types::global_dof_index d_meanValueConstraintNodeIdLocal;
    dftfe::uInt                     d_meanValueConstraintProcId;

    /// Pointer to HOST x vector
    distributedCPUVec<double> *d_xPtr;

    /// Device x vector
    distributedDeviceVec<double> d_xDevice;

    /// Integral of the discrete LDOS field used in the Helmholtz operator.
    double d_totalDOS;

    dftfe::Int d_nLocalCells, d_xLocalDof, d_xLen;

    // constraints
    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
      d_constraintsTotalPotentialInfo;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperPtr;

    /// Matrix free indices
    dftfe::uInt d_matrixFreeVectorComponent;
    dftfe::uInt d_matrixFreeQuadratureComponent;
    dftfe::uInt d_matrixFreeAxQuadratureComponent;

    /// Pointers to HOST quad data
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      *d_residualQuadValuesPtr;
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      *d_ldosAxQuadValuesPtr;

    const dealii::DoFHandler<3>             *d_dofHandlerPRefinedPtr;
    const dealii::AffineConstraints<double> *d_constraintMatrixPRefinedPtr;
    const dealii::MatrixFree<3, double>     *d_matrixFreeDataPRefinedPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtr;

    // Matrix free wrapper object — uses the Helmholtz operator with a
    // per-(cell, quadrature-point) coefficient pointer to apply the LDOS
    // operator (Helmholtz with spatially varying coefficient).
    std::unique_ptr<
      dftfe::MatrixFreeWrapperClass<double,
                                    dftfe::operatorList::Helmholtz,
                                    dftfe::utils::MemorySpace::DEVICE,
                                    false>>
      d_matrixFreeWrapperDevice;

    /// Device buffer for per-quad LDOS coefficients in MatrixFree
    /// cell-reordered layout. Stores 4π·n_loc(r_q).
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_ldosCoeffQuadDevice;

    std::vector<dftfe::uInt> d_cellIndexToMacroCellSubCellIndexMap;

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#  endif // ldosSolverProblemDevice_H_
#endif   // DFTFE_WITH_DEVICE
