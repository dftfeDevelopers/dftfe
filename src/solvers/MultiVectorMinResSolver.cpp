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

#include "MultiVectorMinResSolver.h"
#include "DeviceAPICalls.h"
namespace dftfe
{
  // constructor
  MultiVectorMinResSolver::MultiVectorMinResSolver(
    const MPI_Comm &mpi_comm_parent,
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
  MultiVectorMinResSolver::solve(
    MultiVectorLinearSolverProblem<memorySpace> &problem,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                            BLASWrapperPtr,
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &xMemSpace,
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
                                        debugLevel > 2 ?
                                          dealii::TimerOutput::summary :
                                          dealii::TimerOutput::never,
                                        dealii::TimerOutput::wall_times);

    bool   iterate = true;
    double omega   = 1.0;

    dftfe::utils::MemoryStorage<double, memorySpace> d_onesMemSpace;
    d_onesMemSpace.resize(locallyOwned);
    d_onesMemSpace.setValue(1.0);

    dftfe::linearAlgebra::MultiVector<double, memorySpace> tempVec(xMemSpace,
                                                                   0.0);

    // TODO use dft parameters to get this
    const double rhsNormTolForZero = 1e-15;
    computing_timer.enter_subsection("Compute Rhs MPI");
    dftfe::linearAlgebra::MultiVector<double, memorySpace> &bMemSpace =
      problem.computeRhs(NDBCVec, xMemSpace, blockSize);

    computing_timer.leave_subsection("Compute Rhs MPI");

    computing_timer.enter_subsection("MINRES solver MPI");
    computing_timer.enter_subsection("MINRES initial MPI");


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      negOneHost(blockSize, -1.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      beta1Host(blockSize, 0.0);


    dftfe::utils::MemoryStorage<double, memorySpace> betaMemSpace(blockSize,
                                                                  0.0);
    dftfe::utils::MemoryStorage<double, memorySpace> beta1MemSpace(blockSize,
                                                                   0.0);
    dftfe::utils::MemoryStorage<double, memorySpace> alphaMemSpace(blockSize,
                                                                   0.0);
    dftfe::utils::MemoryStorage<double, memorySpace> negOneMemSpace(blockSize,
                                                                    -1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> oneMemSpace(blockSize,
                                                                 1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> sMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> negBetaByBetaOldMemSpace(
      blockSize, 1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> negAlphaByBetaMemSpace(
      blockSize, 1.0);


    dftfe::utils::MemoryStorage<double, memorySpace> negOldepsMemSpace(
      blockSize, 1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> negDeltaMemSpace(blockSize,
                                                                      1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> denomMemSpace(blockSize,
                                                                   1.0);
    dftfe::utils::MemoryStorage<double, memorySpace> phiMemSpace(blockSize,
                                                                 1.0);

    dftfe::utils::MemoryStorage<double, memorySpace> coeffForXMemInMemSpace(
      blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, memorySpace> coeffForXTmpMemSpace(
      blockSize, 0.0);

    // allocate the vectors
    dftfe::linearAlgebra::MultiVector<double, memorySpace> xTmpMemSpace,
      yMemSpace, vMemSpace, r1MemSpace, r2MemSpace, wMemSpace, w1MemSpace,
      w2MemSpace;
    xTmpMemSpace.reinit(bMemSpace);
    yMemSpace.reinit(bMemSpace);
    vMemSpace.reinit(bMemSpace);
    r1MemSpace.reinit(bMemSpace);
    r2MemSpace.reinit(bMemSpace);
    wMemSpace.reinit(bMemSpace);
    w1MemSpace.reinit(bMemSpace);
    w2MemSpace.reinit(bMemSpace);

    BLASWrapperPtr->axpby(locallyOwned * blockSize,
                          1.0,
                          xMemSpace.begin(),
                          0.0,
                          xTmpMemSpace.begin());

    problem.vmult(r1MemSpace, xTmpMemSpace, blockSize);


    BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,
                                                      locallyOwned,
                                                      bMemSpace.data(),
                                                      negOneMemSpace.data(),
                                                      r1MemSpace.begin());



    BLASWrapperPtr->axpby(locallyOwned * blockSize,
                          0.0,
                          xTmpMemSpace.begin(),
                          -1.0,
                          r1MemSpace.begin());

    problem.precondition_Jacobi(yMemSpace, r1MemSpace, omega);

    // pcout<<" first MultiVectorXDot : beta1MemSpace =
    // "<<beta1MemSpace.size()<<" beta1Host = "<<beta1Host.size();

    BLASWrapperPtr->MultiVectorXDot(blockSize,
                                    locallyOwned,
                                    r1MemSpace.begin(),
                                    yMemSpace.begin(),
                                    d_onesMemSpace.begin(),
                                    tempVec.begin(),
                                    beta1MemSpace.begin(),
                                    mpi_communicator,
                                    beta1Host.begin());

    bool notPosDef = std::any_of(beta1Host.begin(),
                                 beta1Host.end(),
                                 [](double val) { return val <= 0.0; });

    for (unsigned int i = 0; i < blockSize; ++i)
      beta1Host[i] = std::sqrt(beta1Host[i]) + rhsNormTolForZero;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    epsHost(blockSize, std::numeric_limits<double>::epsilon());
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      oldbHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      betaHost(beta1Host);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      dbarHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      epslnHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      oldepsHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      qrnormHost(beta1Host);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      phiHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                                                         phibarHost(beta1Host);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> csHost(
      blockSize, -1.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> snHost(
      blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      alphaHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gammaHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      deltaHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gbarHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rnormHost(blockSize, 0.0);



    BLASWrapperPtr->axpby(locallyOwned * blockSize,
                          1.0,
                          r1MemSpace.begin(),
                          0.0,
                          r2MemSpace.begin());

    bool hasAllConverged = false;
    dftfe::utils::MemoryStorage<bool, dftfe::utils::MemorySpace::HOST>
                                                                         hasConvergedHost(blockSize, false);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> sHost(
      blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      negBetaByBetaOldHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      negAlphaByBetaHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      denomHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      negOldepsHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      negDeltaHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::HOST>
                 lanczosSizeHost(blockSize, 0);
    unsigned int iter = 0;
    computing_timer.leave_subsection("MINRES initial MPI");
    while (iter < maxNumberIterations && hasAllConverged == false)
      {
        // pcout << "iter  = " << iter << "\n";
        computing_timer.enter_subsection("MINRES vmult MPI");
        for (unsigned int i = 0; i < blockSize; ++i)
          sHost[i] = (1.0 / betaHost[i]) - 1.0;

        BLASWrapperPtr->axpby(locallyOwned * blockSize,
                              1.0,
                              yMemSpace.begin(),
                              0.0,
                              vMemSpace.begin());

        sMemSpace.copyFrom(sHost);


        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,
                                                          locallyOwned,
                                                          vMemSpace.data(),
                                                          sMemSpace.data(),
                                                          vMemSpace.data());

        problem.vmult(yMemSpace, vMemSpace, blockSize);

        if (iter > 0)
          {
            for (unsigned int i = 0; i < blockSize; ++i)
              negBetaByBetaOldHost[i] = -betaHost[i] / oldbHost[i];

            negBetaByBetaOldMemSpace.copyFrom(negBetaByBetaOldHost);
            //            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
            //                                                                                             negBetaByBetaOldMemSpace.begin(),
            //                                                                                             negBetaByBetaOldHost.begin());

            BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
              blockSize,
              locallyOwned,
              r1MemSpace.data(),
              negBetaByBetaOldMemSpace.data(),
              yMemSpace.data());
          }
        computing_timer.leave_subsection("MINRES vmult MPI");
        computing_timer.enter_subsection("MINRES linalg MPI");

        // pcout<<" second MultiVectorXDot: alphaMemSpace =
        // "<<alphaMemSpace.size()<<" alphaHost = "<<alphaHost.size()<<"\n";
        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        vMemSpace.begin(),
                                        yMemSpace.begin(),
                                        d_onesMemSpace.begin(),
                                        tempVec.begin(),
                                        alphaMemSpace.begin(),
                                        mpi_communicator,
                                        alphaHost.begin());

        for (unsigned int i = 0; i < blockSize; ++i)
          negAlphaByBetaHost[i] = -alphaHost[i] / betaHost[i];

        negAlphaByBetaMemSpace.copyFrom(negAlphaByBetaHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         negAlphaByBetaMemSpace.begin(),
        //                                                                                         negAlphaByBetaHost.begin());

        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
          blockSize,
          locallyOwned,
          r2MemSpace.data(),
          negAlphaByBetaMemSpace.data(),
          yMemSpace.data());

        BLASWrapperPtr->axpby(locallyOwned * blockSize,
                              1.0,
                              r2MemSpace.begin(),
                              0.0,
                              r1MemSpace.begin());

        BLASWrapperPtr->axpby(locallyOwned * blockSize,
                              1.0,
                              yMemSpace.begin(),
                              0.0,
                              r2MemSpace.begin());

        problem.precondition_Jacobi(yMemSpace, r2MemSpace, omega);

        oldbHost = betaHost;

        // pcout<<" third MultiVectorXDot: betaMemSpace =
        // "<<betaMemSpace.size()<<" betaHost = "<<betaHost.size()<<"\n";

        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        r2MemSpace.begin(),
                                        yMemSpace.begin(),
                                        d_onesMemSpace.begin(),
                                        tempVec.begin(),
                                        betaMemSpace.begin(),
                                        mpi_communicator,
                                        betaHost.begin());

        notPosDef = std::any_of(betaHost.begin(),
                                betaHost.end(),
                                [](double val) { return val <= 0.0; });

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            betaHost[i] = std::sqrt(betaHost[i]) + rhsNormTolForZero;
          }

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            oldepsHost[i] = epslnHost[i];
            deltaHost[i]  = csHost[i] * dbarHost[i] + snHost[i] * alphaHost[i];
            gbarHost[i]   = snHost[i] * dbarHost[i] - csHost[i] * alphaHost[i];
            epslnHost[i]  = snHost[i] * betaHost[i];
            dbarHost[i]   = -csHost[i] * betaHost[i];

            // Compute next plane rotation Q_k
            gammaHost[i]     = sqrt(gbarHost[i] * gbarHost[i] +
                                betaHost[i] * betaHost[i]); // gamma_k
            gammaHost[i]     = std::max(gammaHost[i], epsHost[i]);
            csHost[i]        = gbarHost[i] / gammaHost[i]; // c_k
            snHost[i]        = betaHost[i] / gammaHost[i]; // s_k
            phiHost[i]       = csHost[i] * phibarHost[i];  // phi_k
            phibarHost[i]    = snHost[i] * phibarHost[i];  // phibar_{k+1}
            denomHost[i]     = 1.0 / gammaHost[i];
            negOldepsHost[i] = -oldepsHost[i];
            negDeltaHost[i]  = -deltaHost[i];
          }

        BLASWrapperPtr->axpby(locallyOwned * blockSize,
                              1.0,
                              w2MemSpace.begin(),
                              0.0,
                              w1MemSpace.begin());

        BLASWrapperPtr->axpby(locallyOwned * blockSize,
                              1.0,
                              wMemSpace.begin(),
                              0.0,
                              w2MemSpace.begin());

        negOldepsMemSpace.copyFrom(negOldepsHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         negOldepsMemSpace.begin(),
        //                                                                                         negOldepsHost.begin());

        negDeltaMemSpace.copyFrom(negDeltaHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         negDeltaMemSpace.begin(),
        //                                                                                         negDeltaHost.begin());

        denomMemSpace.copyFrom(denomHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         denomMemSpace.begin(),
        //                                                                                         denomHost.begin());

        phiMemSpace.copyFrom(phiHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         phiMemSpace.begin(),
        //                                                                                         phiHost.begin());
        BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
          blockSize,
          locallyOwned,
          w1MemSpace.data(),
          negOldepsMemSpace.data(),
          w2MemSpace.data(),
          negDeltaMemSpace.data(),
          wMemSpace.data());

        BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
          blockSize,
          locallyOwned,
          vMemSpace.data(),
          denomMemSpace.data(),
          wMemSpace.data(),
          denomMemSpace.data(),
          wMemSpace.data());

        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,
                                                          locallyOwned,
                                                          wMemSpace.data(),
                                                          phiMemSpace.data(),
                                                          xTmpMemSpace.data());

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            qrnormHost[i] = phibarHost[i];
            rnormHost[i]  = qrnormHost[i];

            //            std::cout<<" res = "<<rnormHost[i]<<"\n";
          }

        // pcout << " iter = " << iter << "\n";
        bool updateFlag = false;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          coeffForXMemHost(blockSize, 1.0);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          coeffForXTmpHost(blockSize, 0.0);
        for (unsigned int i = 0; i < blockSize; ++i)
          {
            // pcout << " res = " << rnormHost[i] << "\n";
            if (rnormHost[i] < absTolerance && hasConvergedHost[i] == false)
              {
                updateFlag          = true;
                hasConvergedHost[i] = true;
                lanczosSizeHost[i]  = iter + 1;
                coeffForXMemHost[i] = 0.0;
                coeffForXTmpHost[i] = 1.0;
              }
          }
        if (updateFlag)
          {
            coeffForXMemInMemSpace.copyFrom(coeffForXMemHost);
            //            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
            //                                                                                             coeffForXMemInMemSpace.begin(),
            //                                                                                             coeffForXMemHost.begin());

            coeffForXTmpMemSpace.copyFrom(coeffForXTmpHost);
            //            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
            //                                                                                             coeffForXTmpMemSpace.begin(),
            //                                                                                             coeffForXTmpHost.begin());

            BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
              blockSize,
              locallyOwned,
              xMemSpace.data(),
              coeffForXMemInMemSpace.data(),
              xTmpMemSpace.data(),
              coeffForXTmpMemSpace.data(),
              xMemSpace.data());
          }

        if (std::all_of(hasConvergedHost.begin(),
                        hasConvergedHost.end(),
                        [](bool boolVal) { return boolVal; }))
          {
            hasAllConverged = true;
          }

        computing_timer.leave_subsection("MINRES linalg MPI");
        iter++;
      }

    computing_timer.enter_subsection("MINRES dist MPI");

    bool updateUncovergedFlag = false;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      coeffForXMemHost(blockSize, 1.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      coeffForXTmpHost(blockSize, 0.0);
    for (unsigned int i = 0; i < blockSize; ++i)
      {
        if (hasConvergedHost[i] == false)
          {
            pcout << " MINRES SOlVER not converging for iBlock = " << i << "\n";
            updateUncovergedFlag = true;
            coeffForXMemHost[i]  = 0.0;
            coeffForXTmpHost[i]  = 1.0;
          }
      }

    if (updateUncovergedFlag)
      {
        coeffForXMemInMemSpace.copyFrom(coeffForXMemHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         coeffForXMemInMemSpace.begin(),
        //                                                                                         coeffForXMemHost.begin());

        coeffForXTmpMemSpace.copyFrom(coeffForXTmpHost);
        //        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
        //                                                                                         coeffForXTmpMemSpace.begin(),
        //                                                                                         coeffForXTmpHost.begin());

        BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
          blockSize,
          locallyOwned,
          xMemSpace.data(),
          coeffForXMemInMemSpace.data(),
          xTmpMemSpace.data(),
          coeffForXTmpMemSpace.data(),
          xMemSpace.data());
      }


    std::vector<double> l2NormVec(blockSize, 0.0);

    // xMemSpace.l2Norm(&l2NormVec[0]);
    /*
        pcout<<" xMemSpace before dist = \n";
        for(unsigned int iB = 0; iB  < blockSize ; iB++)
        {
                pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
        }
    */
    problem.distributeX();

    // xMemSpace.l2Norm(&l2NormVec[0]);
/*
    pcout<<" xMemSpace after dist = \n";
    for(unsigned int iB = 0; iB  < blockSize ; iB++)
    {
            pcout<<" iB = "<<iB<<" norm = "<<l2NormVec[iB]<<"\n";
    }
    */
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    computing_timer.leave_subsection("MINRES dist MPI");
  }

  template void
  MultiVectorMinResSolver::solve<dftfe::utils::MemorySpace::HOST>(
    MultiVectorLinearSolverProblem<dftfe::utils::MemorySpace::HOST> &problem,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtr,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      &xMemSpace,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      &                NDBCVec,
    unsigned int       locallyOwned,
    unsigned int       blockSize,
    const double       absTolerance,
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    bool               distributeFlag);


#ifdef DFTFE_WITH_DEVICE

  template void
  MultiVectorMinResSolver::solve<dftfe::utils::MemorySpace::DEVICE>(
    MultiVectorLinearSolverProblem<dftfe::utils::MemorySpace::DEVICE> &problem,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::DEVICE>
      &xMemSpace,
    dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::DEVICE>
      &                NDBCVec,
    unsigned int       locallyOwned,
    unsigned int       blockSize,
    const double       absTolerance,
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    bool               distributeFlag);

#endif
} // namespace dftfe
