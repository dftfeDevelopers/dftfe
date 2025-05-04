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
// @author Sambit Das
//

#ifdef DFTFE_WITH_DEVICE
#  include <solveVselfInBinsDevice.h>
#  include <vectorUtilities.h>
#  include <MemoryStorage.h>
#  include "solveVselfInBinsDeviceKernels.h"

namespace dftfe
{
  namespace poissonDevice
  {
    namespace
    {
      void
      computeAX(
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
          &BLASWrapperPtr,
        dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
                                     &constraintsMatrixDataInfoDevice,
        distributedDeviceVec<double> &src,
        distributedDeviceVec<double> &temp,
        const dftfe::uInt             totalLocallyOwnedCells,
        const dftfe::uInt             numberNodesPerElement,
        const dftfe::uInt             numberVectors,
        const dftfe::uInt             localSize,
        const dftfe::uInt             ghostSize,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &poissonCellStiffnessMatricesD,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &inhomoIdsColoredVecFlattenedD,
        const dftfe::utils::MemoryStorage<dftfe::uInt,
                                          dftfe::utils::MemorySpace::DEVICE>
                                     &cellLocalProcIndexIdMapD,
        distributedDeviceVec<double> &dst,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &cellNodalVectorD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &cellStiffnessMatrixTimesVectorD)
      {
        // const dftfe::uInt numberVectors = 1;
        dst.setValue(0);

        // distributedDeviceVec<double> temp;
        // temp.reinit(src);
        // temp=src;
        dftfe::utils::deviceMemcpyD2D(temp.begin(),
                                      src.begin(),
                                      localSize * numberVectors *
                                        sizeof(double));

        // src.update_ghost_values();
        // constraintsMatrixDataInfoDevice.distribute(src,numberVectors);
        temp.updateGhostValues();
        constraintsMatrixDataInfoDevice.distribute(temp);

        if ((localSize + ghostSize) > 0)
          scale(numberVectors * (localSize + ghostSize),
                temp.begin(),
                inhomoIdsColoredVecFlattenedD.begin());
        //
        // elemental matrix-multiplication
        //
        const double scalarCoeffAlpha = 1.0 / (4.0 * M_PI),
                     scalarCoeffBeta  = 0.0;

        if (totalLocallyOwnedCells > 0)
          BLASWrapperPtr->stridedCopyToBlock(numberVectors,
                                             totalLocallyOwnedCells *
                                               numberNodesPerElement,
                                             temp.begin(), // src.begin(),
                                             cellNodalVectorD.begin(),
                                             cellLocalProcIndexIdMapD.data());



        const dftfe::uInt strideA = numberNodesPerElement * numberVectors;
        const dftfe::uInt strideB =
          numberNodesPerElement * numberNodesPerElement;
        const dftfe::uInt strideC = numberNodesPerElement * numberVectors;

        //
        // do matrix-matrix multiplication
        //
        BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          numberVectors,
          numberNodesPerElement,
          numberNodesPerElement,
          &scalarCoeffAlpha,
          cellNodalVectorD.begin(),
          numberVectors,
          strideA,
          poissonCellStiffnessMatricesD.begin(),
          numberNodesPerElement,
          strideB,
          &scalarCoeffBeta,
          cellStiffnessMatrixTimesVectorD.begin(),
          numberVectors,
          strideC,
          totalLocallyOwnedCells);

        if (totalLocallyOwnedCells > 0)
          BLASWrapperPtr->axpyStridedBlockAtomicAdd(
            numberVectors,
            totalLocallyOwnedCells * numberNodesPerElement,
            cellStiffnessMatrixTimesVectorD.begin(),
            dst.begin(),
            cellLocalProcIndexIdMapD.begin());


        // think dirichlet hanging node linked to two master solved nodes
        if ((localSize + ghostSize) > 0)
          scale(numberVectors * (localSize + ghostSize),
                dst.begin(),
                inhomoIdsColoredVecFlattenedD.begin());


        constraintsMatrixDataInfoDevice.distribute_slave_to_master(dst);

        dst.accumulateAddLocallyOwned();
        temp.setValue(0);

        if (localSize > 0)
          scale(numberVectors * localSize,
                dst.begin(),
                inhomoIdsColoredVecFlattenedD.begin());

        // src.zero_out_ghost_values();
        // constraintsMatrixDataInfoDevice.set_zero(src,numberVectors);
      }

      void
      precondition_Jacobi(const double     *src,
                          const double     *diagonalA,
                          const dftfe::uInt numberVectors,
                          const dftfe::uInt localSize,
                          double           *dst)
      {
        if (localSize > 0)
          diagScale(numberVectors, localSize, src, diagonalA, dst);
      }

      void
      computeResidualSq(const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                          dftfe::utils::MemorySpace::DEVICE>> &BLASWrapperPtr,
                        const double                          *vec1,
                        const double                          *vec2,
                        double                                *vecTemp,
                        const double                          *onesVec,
                        const dftfe::uInt                      numberVectors,
                        const dftfe::uInt                      localSize,
                        double                                *residualNormSq)
      {
        if (localSize > 0)
          dotProductContributionBlocked(numberVectors * localSize,
                                        vec1,
                                        vec2,
                                        vecTemp);

        const double alpha = 1.0, beta = 0.0;
        BLASWrapperPtr->xgemm('N',
                              'T',
                              1,
                              numberVectors,
                              localSize,
                              &alpha,
                              onesVec,
                              1,
                              vecTemp,
                              numberVectors,
                              &beta,
                              residualNormSq,
                              1);
      }
    } // namespace

    void
    solveVselfInBins(
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellGradNIGradNJIntergralDevice,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                              &BLASWrapperPtr,
      const dealii::MatrixFree<3, double>     &matrixFreeData,
      const dftfe::uInt                        mfDofHandlerIndex,
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const double                            *bH,
      const double                            *diagonalAH,
      const double                            *inhomoIdsColoredVecFlattenedH,
      const dftfe::uInt                        localSize,
      const dftfe::uInt                        ghostSize,
      const dftfe::uInt                        numberBins,
      const MPI_Comm                          &mpiCommParent,
      const MPI_Comm                          &mpiCommDomain,
      double                                  *xH,
      const dftfe::Int                         verbosity,
      const dftfe::uInt                        maxLinearSolverIterations,
      const double                             absLinearSolverTolerance,
      const bool isElectroFEOrderDifferentFromFEOrder)
    {
      int this_process;
      MPI_Comm_rank(mpiCommParent, &this_process);

      const dftfe::uInt blockSize = numberBins;
      const dftfe::uInt totalLocallyOwnedCells =
        matrixFreeData.n_physical_cells();
      const dftfe::uInt numberNodesPerElement =
        matrixFreeData.get_dofs_per_cell(mfDofHandlerIndex);

      distributedDeviceVec<double> xD;

      MPI_Barrier(mpiCommParent);
      double time = MPI_Wtime();

      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
        blockSize,
        xD);

      xD.setValue(0);
      dftfe::utils::deviceMemcpyH2D(xD.begin(),
                                    xH,
                                    localSize * numberBins * sizeof(double));

      MPI_Barrier(mpiCommParent);
      time = MPI_Wtime() - time;
      if (verbosity >= 2 && this_process == 0)
        std::cout << " poissonDevice::solveVselfInBins: time for creating xD: "
                  << time << std::endl;

      distributedCPUMultiVec<double> flattenedArray;
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
        blockSize,
        flattenedArray);
      std::vector<dftfe::uInt> cellLocalProcIndexIdMapH;

      vectorTools::computeCellLocalIndexSetMap(
        flattenedArray.getMPIPatternP2P(),
        matrixFreeData,
        mfDofHandlerIndex,
        blockSize,
        cellLocalProcIndexIdMapH);

      dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
        constraintsMatrixDataInfoDevice;
      constraintsMatrixDataInfoDevice.initialize(
        matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
        hangingPeriodicConstraintMatrix);

      constraintsMatrixDataInfoDevice.set_zero(xD);

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      time = MPI_Wtime();

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> bD(
        localSize * numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        diagonalAD(localSize, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        inhomoIdsColoredVecFlattenedD((localSize + ghostSize) * numberBins,
                                      0.0);
      dftfe::utils::MemoryStorage<dftfe::uInt,
                                  dftfe::utils::MemorySpace::DEVICE>
        cellLocalProcIndexIdMapD(totalLocallyOwnedCells *
                                 numberNodesPerElement);

      dftfe::utils::deviceMemcpyH2D(bD.begin(),
                                    bH,
                                    localSize * numberBins * sizeof(double));

      dftfe::utils::deviceMemcpyH2D(diagonalAD.begin(),
                                    diagonalAH,
                                    localSize * sizeof(double));

      dftfe::utils::deviceMemcpyH2D(inhomoIdsColoredVecFlattenedD.begin(),
                                    inhomoIdsColoredVecFlattenedH,
                                    (localSize + ghostSize) * numberBins *
                                      sizeof(double));


      dftfe::utils::deviceMemcpyH2D(cellLocalProcIndexIdMapD.begin(),
                                    &cellLocalProcIndexIdMapH[0],
                                    totalLocallyOwnedCells *
                                      numberNodesPerElement *
                                      sizeof(dftfe::uInt));

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      time = MPI_Wtime() - time;
      if (verbosity >= 2 && this_process == 0)
        std::cout
          << " poissonDevice::solveVselfInBins: time for mem allocation: "
          << time << std::endl;

      cgSolver(BLASWrapperPtr,
               constraintsMatrixDataInfoDevice,
               bD.begin(),
               diagonalAD.begin(),
               cellGradNIGradNJIntergralDevice,
               inhomoIdsColoredVecFlattenedD,
               cellLocalProcIndexIdMapD,
               localSize,
               ghostSize,
               numberBins,
               totalLocallyOwnedCells,
               numberNodesPerElement,
               verbosity,
               maxLinearSolverIterations,
               absLinearSolverTolerance,
               mpiCommParent,
               mpiCommDomain,
               xD);

      dftfe::utils::deviceMemcpyD2H(xH,
                                    xD.begin(),
                                    localSize * numberBins * sizeof(double));
    }

    void
    cgSolver(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
                   &constraintsMatrixDataInfoDevice,
      const double *bD,
      const double *diagonalAD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &poissonCellStiffnessMatricesD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &inhomoIdsColoredVecFlattenedD,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
                                   &cellLocalProcIndexIdMapD,
      const dftfe::uInt             localSize,
      const dftfe::uInt             ghostSize,
      const dftfe::uInt             numberBins,
      const dftfe::uInt             totalLocallyOwnedCells,
      const dftfe::uInt             numberNodesPerElement,
      const dftfe::Int              debugLevel,
      const dftfe::uInt             maxIter,
      const double                  absTol,
      const MPI_Comm               &mpiCommParent,
      const MPI_Comm               &mpiCommDomain,
      distributedDeviceVec<double> &x)
    {
      int this_process;
      MPI_Comm_rank(mpiCommParent, &this_process);

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      double start_time = MPI_Wtime();

      // initialize certain variables
      const double negOne = -1.0;
      // const double posOne = 1.0;
      const dftfe::uInt inc = 1;

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        delta_newD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        delta_oldD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        delta_0D(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        alphaD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        betaD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        scalarD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        residualNormSqD(numberBins, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        negOneD(numberBins, -1.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        posOneD(numberBins, 1.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        vecTempD(localSize * numberBins, 1.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        onesVecD(localSize, 1.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        cellNodalVectorD(totalLocallyOwnedCells * numberNodesPerElement *
                         numberBins);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        cellStiffnessMatrixTimesVectorD(totalLocallyOwnedCells *
                                        numberNodesPerElement * numberBins);

      std::vector<double> delta_newH(numberBins, 0.0);
      std::vector<double> delta_oldH(numberBins, 0.0);
      std::vector<double> alphaH(numberBins, 0.0);
      std::vector<double> betaH(numberBins, 0.0);
      std::vector<double> scalarH(numberBins, 0.0);
      std::vector<double> residualNormSqH(numberBins, 0.0);

      // compute RHS b
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // b;

      // double start_timeRhs = MPI_Wtime();
      // problem.computeRhs(b);
      // double end_timeRhs = MPI_Wtime() - start_timeRhs;

      // if(debugLevel >= 2)
      // std::cout<<" Time for Poisson problem compute rhs:
      // "<<end_timeRhs<<std::endl;

      // get size of vectors
      // dftfe::uInt localSize = b.size();


      // get access to initial guess for solving Ax=b
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> &
      // x = problem.getX(); x.update_ghost_values();


      // compute Ax
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // Ax; Ax.resize(localSize,0.0);
      distributedDeviceVec<double> Ax;
      Ax.reinit(x);
      // computeAX(x,Ax);

      distributedDeviceVec<double> r;
      r.reinit(x);

      distributedDeviceVec<double> q, s;
      q.reinit(x);
      s.reinit(x);

      distributedDeviceVec<double> d, temp;
      d.reinit(x);
      temp.reinit(x);

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      double device_time = MPI_Wtime() - start_time;
      if (debugLevel >= 2 && this_process == 0)
        std::cout
          << " poissonDevice::solveVselfInBins: time for Device CG solver memory allocation: "
          << device_time << std::endl;

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      start_time = MPI_Wtime();

      computeAX(BLASWrapperPtr,
                constraintsMatrixDataInfoDevice,
                x,
                temp,
                totalLocallyOwnedCells,
                numberNodesPerElement,
                numberBins,
                localSize,
                ghostSize,
                poissonCellStiffnessMatricesD,
                inhomoIdsColoredVecFlattenedD,
                cellLocalProcIndexIdMapD,
                Ax,
                cellNodalVectorD,
                cellStiffnessMatrixTimesVectorD);


      // compute residue r = b - Ax
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // r; r.resize(localSize,0.0);

      // r = b
      BLASWrapperPtr->xcopy(localSize * numberBins, bD, inc, r.begin(), inc);


      // r = b - Ax i.e r - Ax
      BLASWrapperPtr->xaxpy(
        localSize * numberBins, &negOne, Ax.begin(), inc, r.begin(), inc);


      // precondition r
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // d; d.resize(localSize,0.0);

      // precondition_Jacobi(r,d);
      precondition_Jacobi(
        r.begin(), diagonalAD, numberBins, localSize, d.begin());


      computeResidualSq(BLASWrapperPtr,
                        r.begin(),
                        d.begin(),
                        vecTempD.begin(),
                        onesVecD.begin(),
                        numberBins,
                        localSize,
                        delta_newD.begin());

      dftfe::utils::deviceMemcpyD2H(&delta_newH[0],
                                    delta_newD.begin(),
                                    numberBins * sizeof(double));


      MPI_Allreduce(MPI_IN_PLACE,
                    &delta_newH[0],
                    numberBins,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpiCommDomain);

      dftfe::utils::deviceMemcpyH2D(delta_newD.begin(),
                                    &delta_newH[0],
                                    numberBins * sizeof(double));

      // assign delta0 to delta_new
      delta_0D = delta_newD;

      // allocate memory for q
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // q,s; q.resize(localSize,0.0); s.resize(localSize,0.0);

      dftfe::uInt iterationNumber = 0;

      computeResidualSq(BLASWrapperPtr,
                        r.begin(),
                        r.begin(),
                        vecTempD.begin(),
                        onesVecD.begin(),
                        numberBins,
                        localSize,
                        residualNormSqD.begin());

      dftfe::utils::deviceMemcpyD2H(&residualNormSqH[0],
                                    residualNormSqD.begin(),
                                    numberBins * sizeof(double));


      MPI_Allreduce(MPI_IN_PLACE,
                    &residualNormSqH[0],
                    numberBins,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpiCommDomain);

      if (debugLevel >= 2 && this_process == 0)
        {
          for (dftfe::uInt i = 0; i < numberBins; i++)
            std::cout
              << "Device based Linear Conjugate Gradient solver for bin: " << i
              << " started with residual norm squared: " << residualNormSqH[i]
              << std::endl;
        }

      for (dftfe::uInt iter = 0; iter < maxIter; ++iter)
        {
          // q = Ad
          // computeAX(d,q);


          computeAX(BLASWrapperPtr,
                    constraintsMatrixDataInfoDevice,
                    d,
                    temp,
                    totalLocallyOwnedCells,
                    numberNodesPerElement,
                    numberBins,
                    localSize,
                    ghostSize,
                    poissonCellStiffnessMatricesD,
                    inhomoIdsColoredVecFlattenedD,
                    cellLocalProcIndexIdMapD,
                    q,
                    cellNodalVectorD,
                    cellStiffnessMatrixTimesVectorD);

          // compute alpha
          computeResidualSq(BLASWrapperPtr,
                            d.begin(),
                            q.begin(),
                            vecTempD.begin(),
                            onesVecD.begin(),
                            numberBins,
                            localSize,
                            scalarD.begin());

          dftfe::utils::deviceMemcpyD2H(&scalarH[0],
                                        scalarD.begin(),
                                        numberBins * sizeof(double));


          MPI_Allreduce(MPI_IN_PLACE,
                        &scalarH[0],
                        numberBins,
                        MPI_DOUBLE,
                        MPI_SUM,
                        mpiCommDomain);

          // for (dftfe::uInt i=0;i <numberBins; i++)
          //   std::cout<< "scalar "<<scalarH[i]<<std::endl;

          for (dftfe::uInt i = 0; i < numberBins; i++)
            alphaH[i] = delta_newH[i] / scalarH[i];

          // for (dftfe::uInt i=0;i <numberBins; i++)
          //   std::cout<< "alpha "<<alphaH[i]<<std::endl;

          dftfe::utils::deviceMemcpyH2D(alphaD.begin(),
                                        &alphaH[0],
                                        numberBins * sizeof(double));

          // update x; x = x + alpha*d
          if (localSize > 0)
            daxpyBlocked(
              numberBins, localSize, d.begin(), alphaD.begin(), x.begin());

          if (iter % 50 == 0)
            {
              // r = b
              BLASWrapperPtr->xcopy(
                localSize * numberBins, bD, inc, r.begin(), inc);

              // computeAX(x,Ax);

              computeAX(BLASWrapperPtr,
                        constraintsMatrixDataInfoDevice,
                        x,
                        temp,
                        totalLocallyOwnedCells,
                        numberNodesPerElement,
                        numberBins,
                        localSize,
                        ghostSize,
                        poissonCellStiffnessMatricesD,
                        inhomoIdsColoredVecFlattenedD,
                        cellLocalProcIndexIdMapD,
                        Ax,
                        cellNodalVectorD,
                        cellStiffnessMatrixTimesVectorD);

              if (localSize > 0)
                daxpyBlocked(numberBins,
                             localSize,
                             Ax.begin(),
                             negOneD.begin(),
                             r.begin());
            }
          else
            {
              // negAlphaD = -alpha;
              if (localSize > 0)
                dmaxpyBlocked(
                  numberBins, localSize, q.begin(), alphaD.begin(), r.begin());
            }

          // precondition_Jacobi(r,s);
          precondition_Jacobi(
            r.begin(), diagonalAD, numberBins, localSize, s.begin());

          delta_oldD = delta_newD;

          dftfe::utils::deviceMemcpyD2H(&delta_oldH[0],
                                        delta_oldD.begin(),
                                        numberBins * sizeof(double));


          // delta_new = r*s;
          computeResidualSq(BLASWrapperPtr,
                            r.begin(),
                            s.begin(),
                            vecTempD.begin(),
                            onesVecD.begin(),
                            numberBins,
                            localSize,
                            delta_newD.begin());

          // beta = delta_new/delta_old;


          dftfe::utils::deviceMemcpyD2H(&delta_newH[0],
                                        delta_newD.begin(),
                                        numberBins * sizeof(double));


          MPI_Allreduce(MPI_IN_PLACE,
                        &delta_newH[0],
                        numberBins,
                        MPI_DOUBLE,
                        MPI_SUM,
                        mpiCommDomain);


          // for (dftfe::uInt i=0;i <numberBins; i++)
          //   std::cout<< "delta_new "<<delta_newH[i]<<std::endl;

          for (dftfe::uInt i = 0; i < numberBins; i++)
            betaH[i] = delta_newH[i] / delta_oldH[i];

          dftfe::utils::deviceMemcpyH2D(betaD.begin(),
                                        &betaH[0],
                                        numberBins * sizeof(double));

          dftfe::utils::deviceMemcpyH2D(delta_newD.begin(),
                                        &delta_newH[0],
                                        numberBins * sizeof(double));

          // d *= beta;
          if (localSize > 0)
            scaleBlocked(numberBins, localSize, d.begin(), betaD.begin());

          // d.add(1.0,s);
          if (localSize > 0)
            daxpyBlocked(
              numberBins, localSize, s.begin(), posOneD.begin(), d.begin());
          dftfe::uInt isBreak = 1;
          // if(delta_new < relTolerance*relTolerance*delta_0)
          //  isBreak = 1;

          for (dftfe::uInt i = 0; i < numberBins; i++)
            if (delta_newH[i] > absTol * absTol)
              isBreak = 0;

          if (isBreak == 1)
            break;

          iterationNumber += 1;
        }



      // compute residual norm at end
      computeResidualSq(BLASWrapperPtr,
                        r.begin(),
                        r.begin(),
                        vecTempD.begin(),
                        onesVecD.begin(),
                        numberBins,
                        localSize,
                        residualNormSqD.begin());

      dftfe::utils::deviceMemcpyD2H(&residualNormSqH[0],
                                    residualNormSqD.begin(),
                                    numberBins * sizeof(double));

      MPI_Allreduce(MPI_IN_PLACE,
                    &residualNormSqH[0],
                    numberBins,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpiCommDomain);

      // residualNorm = std::sqrt(residualNorm);

      //
      // set error condition
      //
      dftfe::uInt solveStatus = 1;

      if (iterationNumber == maxIter)
        solveStatus = 0;


      if (debugLevel >= 2 && this_process == 0)
        {
          if (solveStatus == 1)
            {
              for (dftfe::uInt i = 0; i < numberBins; i++)
                std::cout << "Linear Conjugate Gradient solver for bin: " << i
                          << " converged after " << iterationNumber + 1
                          << " iterations. with residual norm squared "
                          << residualNormSqH[i] << std::endl;
            }
          else
            {
              for (dftfe::uInt i = 0; i < numberBins; i++)
                std::cout << "Linear Conjugate Gradient solver for bin: " << i
                          << " failed to converge after " << iterationNumber
                          << " iterations. with residual norm squared "
                          << residualNormSqH[i] << std::endl;
            }
        }


      // problem.setX();
      x.updateGhostValues();
      constraintsMatrixDataInfoDevice.distribute(x);
      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      device_time = MPI_Wtime() - start_time;
      if (debugLevel >= 2 && this_process == 0)
        std::cout
          << " poissonDevice::solveVselfInBins: time for Poisson problem iterations: "
          << device_time << std::endl;
    }

  } // namespace poissonDevice
} // namespace dftfe
#endif
