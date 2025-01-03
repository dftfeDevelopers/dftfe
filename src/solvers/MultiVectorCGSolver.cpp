// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian
//

#include "MultiVectorCGSolver.h"
#include "MemoryTransfer.h"
namespace dftfe
{
  // constructor
  MultiVectorCGSolver::MultiVectorCGSolver(const MPI_Comm &mpi_comm_parent,
                                           const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorCGSolver::solve(
    MultiVectorLinearSolverProblem<memorySpace> &problem,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                            BLASWrapperPtr,
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &x,
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &NDBCVec,
    unsigned int                                            locallyOwned,
    unsigned int                                            blockSize,
    const double                                            absTolerance,
    const unsigned int                                      maxNumberIterations,
    const unsigned int                                      debugLevel,
    bool                                                    distributeFlag)

  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);

    dealii::TimerOutput computing_timer(mpi_communicator,
                                        pcout,
                                        dealii::TimerOutput::summary,
                                        dealii::TimerOutput::wall_times);

    bool   iterate = true;
    double omega   = 0.3;
    computing_timer.enter_subsection("Compute Rhs MPI");
    dftfe::linearAlgebra::MultiVector<double, memorySpace>  rhs_one;
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &rhs =
      problem.computeRhs(NDBCVec, x, blockSize);
    computing_timer.leave_subsection("Compute Rhs MPI");

    computing_timer.enter_subsection("CG solver MPI");

    dftfe::linearAlgebra::MultiVector<double, memorySpace> g, d, h;

    dftfe::linearAlgebra::MultiVector<double, memorySpace> tempVec(x, 0);



    int                                              it = 0;
    dftfe::utils::MemoryStorage<double, memorySpace> resMemSpace, alphaMemSpace,
      initial_resMemSpace;
    std::vector<double> resHost, alphaHost, initial_resHost;


    dftfe::utils::MemoryStorage<double, memorySpace> d_onesMemSpace;
    d_onesMemSpace.resize(locallyOwned);
    d_onesMemSpace.setValue(1.0);


    resMemSpace.resize(blockSize);
    alphaMemSpace.resize(blockSize);
    initial_resMemSpace.resize(blockSize);

    resHost.resize(blockSize);
    alphaHost.resize(blockSize);
    initial_resHost.resize(blockSize);


    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    g.reinit(x);
    d.reinit(x);
    h.reinit(x);


    // These should be array of size blockSize

    dftfe::utils::MemoryStorage<double, memorySpace> ghMemSpace, betaMemSpace;

    ghMemSpace.resize(blockSize);
    betaMemSpace.resize(blockSize);


    std::vector<double> ghHost, betaHost;
    ghHost.resize(blockSize);
    betaHost.resize(blockSize);

    problem.vmult(g, x, blockSize);

    BLASWrapperPtr->MultiVectorXDot(blockSize,
                                    locallyOwned,
                                    g.data(),
                                    g.data(),
                                    d_onesMemSpace.data(),
                                    tempVec.data(),
                                    resMemSpace.data(),
                                    mpi_communicator,
                                    resHost.data());

    pcout << "initial residuals = \n";
    for (unsigned int i = 0; i < blockSize; i++)
      {
        initial_resHost[i] = resHost[i];
        pcout << initial_resHost[i] << "\n";
      }
    pcout << "\n";
    // g = g - rhs;
    BLASWrapperPtr->axpby(
      locallyOwned * blockSize, -1.0, rhs.data(), 1.0, g.data());


    BLASWrapperPtr->MultiVectorXDot(blockSize,
                                    locallyOwned,
                                    g.data(),
                                    g.data(),
                                    d_onesMemSpace.data(),
                                    tempVec.data(),
                                    resMemSpace.data(),
                                    mpi_communicator,
                                    resHost.data());

    pcout << "initial residuals = \n";
    for (unsigned int i = 0; i < blockSize; i++)
      {
        resHost[i]         = std::sqrt(resHost[i]);
        initial_resHost[i] = resHost[i];
        pcout << initial_resHost[i] << "\n";
      }
    pcout << "\n";


    problem.precondition_Jacobi(h, g, omega);
    //    d.equ(-1., h);
    d.setValue(0.0);
    // d = d - h;
    BLASWrapperPtr->axpby(
      locallyOwned * blockSize, -1.0, h.data(), 0.0, d.data());

    BLASWrapperPtr->MultiVectorXDot(blockSize,
                                    locallyOwned,
                                    g.data(),
                                    h.data(),
                                    d_onesMemSpace.data(),
                                    tempVec.data(),
                                    ghMemSpace.data(),
                                    mpi_communicator,
                                    ghHost.data());
    while (iterate)
      {
        it++;
        problem.vmult(h, d, blockSize);

        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        h.data(),
                                        d.data(),
                                        d_onesMemSpace.data(),
                                        tempVec.data(),
                                        alphaMemSpace.data(),
                                        mpi_communicator,
                                        alphaHost.data());
        for (unsigned int i = 0; i < blockSize; i++)
          {
            alphaHost[i] = ghHost[i] / alphaHost[i];
          }

        alphaMemSpace.copyFrom(alphaHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         alphaMemSpace.begin(),
        //                                                                                         .begin());
        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
          blockSize, locallyOwned, d.data(), alphaMemSpace.data(), x.data());
        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
          blockSize, locallyOwned, h.data(), alphaMemSpace.data(), g.data());

        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        g.data(),
                                        g.data(),
                                        d_onesMemSpace.data(),
                                        tempVec.data(),
                                        resMemSpace.data(),
                                        mpi_communicator,
                                        resHost.data());

        for (unsigned int i = 0; i < blockSize; i++)
          {
            resHost[i] = std::sqrt(resHost[i]);
          }
        problem.precondition_Jacobi(h, g, omega);
        betaHost = ghHost;

        //        ghMemSpace.copyFrom(ghHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         ghMemSpace.begin(),
        //                                                                                         ghHost.begin());


        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        g.data(),
                                        h.data(),
                                        d_onesMemSpace.data(),
                                        tempVec.data(),
                                        ghMemSpace.data(),
                                        mpi_communicator,
                                        ghHost.data());

        for (unsigned int i = 0; i < blockSize; i++)
          {
            betaHost[i] = (ghHost[i] / betaHost[i]);
          }

        betaMemSpace.copyFrom(betaHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         betaMemSpace.begin(),
        //                                                                                         betaHost.begin());

        BLASWrapperPtr->stridedBlockScaleColumnWise(blockSize,
                                                    locallyOwned,
                                                    betaMemSpace.data(),
                                                    d.data());

        // d = d - h;
        BLASWrapperPtr->axpby(
          locallyOwned * blockSize, -1.0, h.data(), 1.0, d.data());

        bool convergeStat = true;
        for (unsigned int id = 0; id < blockSize; id++)
          {
            if (std::abs(resHost[id]) > absTolerance)
              convergeStat = false;
          }

        if ((convergeStat) || (it > maxNumberIterations))
          iterate = false;
      }

    problem.distributeX();
    if (it > maxNumberIterations)
      {
        pcout
          << "MultiVector Poisson Solve did not converge. Try increasing the number of iterations or check the input\n";
        pcout << "initial abs. residual: " << initial_resHost[0]
              << " , current abs. residual: " << resHost[0]
              << " , nsteps: " << it
              << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
      }

    else
      pcout << "initial abs. residual: " << initial_resHost[0]
            << " , current abs. residual: " << resHost[0] << " , nsteps: " << it
            << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
    computing_timer.leave_subsection("CG solver MPI");
  }

  template void
  MultiVectorCGSolver::solve<dftfe::utils::MemorySpace::HOST>(
    MultiVectorLinearSolverProblem<dftfe::utils::MemorySpace::HOST> &problem,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtr,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      &x,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      &                NDBCVec,
    unsigned int       locallyOwned,
    unsigned int       blockSize,
    const double       absTolerance,
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    bool               distributeFlag);
} // namespace dftfe
