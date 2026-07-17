#include <dftfe/dealiiLinearSolverProblem.h>
#include <dftfe/triangulationManager.h>
#include <dftfe/FEBasisOperations.h>

#ifndef ldosSolverProblem_H_
#  define ldosSolverProblem_H_

namespace dftfe
{
  template <dftfe::uInt FEOrderElectro>
  class ldosSolverProblem : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    ldosSolverProblem(const MPI_Comm &mpi_comm_parent,
                      const MPI_Comm &mpi_comm_domain);

    /**
     * @brief initialize the matrix-free data structures
     *
     */
    void
    init(std::shared_ptr<
           dftfe::basis::
             FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                           &basisOperationsPtr,
         dealii::AffineConstraints<double> &constraintMatrix,
         distributedCPUVec<double>         &x,
         const dftfe::uInt                  matrixFreeVectorComponent,
         const dftfe::uInt                  matrixFreeQuadratureComponent,
         const dftfe::uInt                  matrixFreeAxQuadratureComponent);

    /**
     * @brief reinitialize data structures .
     *
     */
    void
    reinit(
      distributedCPUVec<double> &x,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &residualQuadValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &ldosAxQuadValues);

    /**
     * @brief get the reference to x field
     *
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
     */
    void
    computeRhs(distributedCPUVec<double> &rhs);

    /**
     * @brief Jacobi preconditioning.
     *
     */
    void
    precondition_Jacobi(distributedCPUVec<double>       &dst,
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
              const std::string       &identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    void
    unsubscribe(std::atomic<bool> *const validity,
                const std::string       &identifier = "") const {};

    /// function needed by dealii to mimic SparseMatrix
    bool
    operator!=(double val) const
    {
      return true;
    };

    const distributedCPUVec<double> &
    getDlocMassVector() const;

    double
    computeDlocIntegral(const distributedCPUVec<double> &phi) const;

    double
    getTotalDOS() const;

  private:
    /**
     * @brief required for the cell_loop operation in dealii's MatrixFree class
     *
     */
    void
    AX(const dealii::MatrixFree<3, double>       &matrixFreeData,
       distributedCPUVec<double>                 &dst,
       const distributedCPUVec<double>           &src,
       const std::pair<dftfe::uInt, dftfe::uInt> &cell_range) const;

    /**
     * @brief Compute the diagonal of A.
     *
     */
    void
    computeDiagonalA();

    void
    computeProjectedQuadToNodalField(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                &quadValues,
      distributedCPUVec<double> &nodalField);

    /// storage for diagonal of the A matrix
    distributedCPUVec<double> d_diagonalA;
    distributedCPUVec<double> d_dlocMassVector;

    /// pointer to the x vector being solved for
    distributedCPUVec<double> *d_xPtr;

    /// Integral of the discrete LDOS field used in the Helmholtz operator.
    double d_totalDOS;

    /// matrix free index required to access the DofHandler and
    /// dealii::AffineConstraints<double> objects corresponding to the problem
    dftfe::uInt d_matrixFreeVectorComponent;

    /// matrix free quadrature index
    dftfe::uInt d_matrixFreeQuadratureComponent;
    dftfe::uInt d_matrixFreeAxQuadratureComponent;

    /// pointer to electron density cell and grad residual data
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

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // ldosSolverProblem_H_
