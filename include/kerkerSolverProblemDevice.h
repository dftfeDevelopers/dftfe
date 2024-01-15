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



#if defined(DFTFE_WITH_DEVICE)

#  ifndef kerkerSolverProblemDevice_H_
#    define kerkerSolverProblemDevice_H_

#    include <linearSolverProblemDevice.h>
#    include <triangulationManager.h>
#    include <constraintMatrixInfoDevice.h>
#    include <deviceKernelsGeneric.h>
#    include <MemoryStorage.h>
#    include <dftUtils.h>
#    include <FEBasisOperations.h>


namespace dftfe
{
  /**
   * @brief helmholtz solver problem class template. template parameter FEOrderElectro
   * is the finite element polynomial order for electrostatics
   *
   * @author Gourab Panigrahi
   */
  template <unsigned int FEOrderElectro>
  class kerkerSolverProblemDevice : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    kerkerSolverProblemDevice(const MPI_Comm &mpi_comm_parent,
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
    init(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
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
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x);


    void
    setX();


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


    /**
     * @brief Copies x from Device to Host
     *
     */
    void
    copyXfromDeviceToHost();


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


    /// storage for diagonal of the A matrix
    distributedCPUVec<double>    d_diagonalA;
    distributedDeviceVec<double> d_diagonalAdevice;

    /// pointer to the x vector being solved for
    distributedCPUVec<double> *d_xPtr;

    /// Device x vector being solved for
    distributedDeviceVec<double> d_xDevice;

    // number of cells local to each mpi task, number of degrees of freedom
    // locally owned and total degrees of freedom including ghost
    int d_nLocalCells, d_xLocalDof, d_xLen;

    // kerker mixing constant
    double d_gamma;

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
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
      d_basisOperationsPtr;



    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#  endif // kerkerSolverProblemDevice_H_
#endif
