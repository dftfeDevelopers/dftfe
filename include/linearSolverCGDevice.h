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
#  ifndef linearSolverCGDevice_H_
#    define linearSolverCGDevice_H_

#    include <linearSolverDevice.h>
#    include <linearSolverProblemDevice.h>
#    include <MemoryStorage.h>
#    include <BLASWrapper.h>
namespace dftfe
{
  /**
   * @brief conjugate gradient device linear solver class wrapper
   *
   * @author Gourab Panigrahi
   */
  class linearSolverCGDevice : public linearSolverDevice
  {
  public:
    enum solverType
    {
      CG = 0,
      GMRES
    };

    /**
     * @brief Constructor
     *
     * @param mpi_comm_parent parent mpi communicato
     * @param mpi_comm_domain domain mpi communicator
     * @param type enum specifying the choice of the linear solver
     */
    linearSolverCGDevice(
      const MPI_Comm & mpi_comm_parent,
      const MPI_Comm & mpi_comm_domain,
      const solverType type,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtr);

    /**
     * @brief Solve linear system, A*x=Rhs
     *
     * @param problem linearSolverProblemDevice object (functor) to compute Rhs and A*x, and preconditioning
     * @param relTolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     */
    void
    solve(linearSolverProblemDevice &problem,
          const double               absTolerance,
          const unsigned int         maxNumberIterations,
          const int                  debugLevel     = 0,
          bool                       distributeFlag = true);

  private:
    /// enum denoting the choice of the linear solver
    const solverType d_type;

    /// define some temporary vectors
    distributedDeviceVec<double> d_qvec, d_rvec, d_dvec;

    int     d_xLocalDof;
    double *d_devSumPtr;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_devSum;

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperPtr;

    /**
     * @brief Combines precondition and dot product
     *
     */
    double
    applyPreconditionAndComputeDotProduct(const double *jacobi);

    /**
     * @brief Combines precondition, sadd and dot product
     *
     */
    double
    applyPreconditionComputeDotProductAndSadd(const double *jacobi);

    /**
     * @brief Combines scaling and norm
     *
     */
    double
    scaleXRandComputeNorm(double *x, const double &alpha);
  };

} // namespace dftfe
#  endif // linearSolverCGDevice_H_
#endif
