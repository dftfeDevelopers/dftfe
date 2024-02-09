// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
#  include <deviceKernelsGeneric.h>
#  include <MemoryStorage.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceBlasWrapper.h>

namespace dftfe
{
  namespace poissonDevice
  {
    namespace
    {
      __global__ void
      diagScaleKernel(const unsigned int blockSize,
                      const unsigned int numContiguousBlocks,
                      const double *     srcArray,
                      const double *     scalingVector,
                      double *           dstArray)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex = index / blockSize;
            *(dstArray + index) =
              *(srcArray + index) * (*(scalingVector + blockIndex));
          }
      }

      __global__ void
      dotProductContributionBlockedKernel(const unsigned int numEntries,
                                          const double *     vec1,
                                          const double *     vec2,
                                          double *           vecTemp)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numEntries;
             index += blockDim.x * gridDim.x)
          {
            vecTemp[index] = vec1[index] * vec2[index];
          }
      }

      __global__ void
      scaleBlockedKernel(const unsigned int blockSize,
                         const unsigned int numContiguousBlocks,
                         double *           xArray,
                         const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int intraBlockIndex = index % blockSize;
            *(xArray + index) *= (*(scalingVector + intraBlockIndex));
          }
      }

      __global__ void
      scaleKernel(const unsigned int numEntries,
                  double *           xArray,
                  const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numEntries;
             index += blockDim.x * gridDim.x)
          {
            xArray[index] *= scalingVector[index];
          }
      }

      // y=alpha*x+y
      __global__ void
      daxpyBlockedKernel(const unsigned int blockSize,
                         const unsigned int numContiguousBlocks,
                         const double *     x,
                         const double *     alpha,
                         double *           y)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / blockSize;
            const unsigned int intraBlockIndex = index - blockIndex * blockSize;
            y[index] += alpha[intraBlockIndex] * x[index];
          }
      }


      // y=-alpha*x+y
      __global__ void
      dmaxpyBlockedKernel(const unsigned int blockSize,
                          const unsigned int numContiguousBlocks,
                          const double *     x,
                          const double *     alpha,
                          double *           y)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / blockSize;
            const unsigned int intraBlockIndex = index - blockIndex * blockSize;
            y[index] += -alpha[intraBlockIndex] * x[index];
          }
      }



      void
      computeAX(
        dftfe::utils::deviceBlasHandle_t &    handle,
        dftUtils::constraintMatrixInfoDevice &constraintsMatrixDataInfoDevice,
        distributedDeviceVec<double> &        src,
        distributedDeviceVec<double> &        temp,
        const unsigned int                    totalLocallyOwnedCells,
        const unsigned int                    numberNodesPerElement,
        const unsigned int                    numberVectors,
        const unsigned int                    localSize,
        const unsigned int                    ghostSize,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &poissonCellStiffnessMatricesD,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &inhomoIdsColoredVecFlattenedD,
        const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                           cellLocalProcIndexIdMapD,
        distributedDeviceVec<double> &dst,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &cellNodalVectorD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &cellStiffnessMatrixTimesVectorD)
      {
        // const unsigned int numberVectors = 1;
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
        constraintsMatrixDataInfoDevice.distribute(temp, numberVectors);

        if ((localSize + ghostSize) > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          scaleKernel<<<
            (numberVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE *(localSize + ghostSize),
            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberVectors * (localSize + ghostSize),
            temp.begin(),
            inhomoIdsColoredVecFlattenedD.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(scaleKernel,
                             (numberVectors +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE *
                               (localSize + ghostSize),
                             dftfe::utils::DEVICE_BLOCK_SIZE,
                             0,
                             0,
                             numberVectors * (localSize + ghostSize),
                             temp.begin(),
                             inhomoIdsColoredVecFlattenedD.begin());
#  endif
        //
        // elemental matrix-multiplication
        //
        const double scalarCoeffAlpha = 1.0 / (4.0 * M_PI),
                     scalarCoeffBeta  = 0.0;

        if (totalLocallyOwnedCells > 0)
          dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
            numberVectors,
            totalLocallyOwnedCells * numberNodesPerElement,
            temp.begin(), // src.begin(),
            cellNodalVectorD.begin(),
            cellLocalProcIndexIdMapD.begin());



        const unsigned int strideA = numberNodesPerElement * numberVectors;
        const unsigned int strideB =
          numberNodesPerElement * numberNodesPerElement;
        const unsigned int strideC = numberNodesPerElement * numberVectors;

        //
        // do matrix-matrix multiplication
        //
        dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
          handle,
          dftfe::utils::DEVICEBLAS_OP_N,
          dftfe::utils::DEVICEBLAS_OP_N,
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
          dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
            numberVectors,
            totalLocallyOwnedCells * numberNodesPerElement,
            cellStiffnessMatrixTimesVectorD.begin(),
            dst.begin(),
            cellLocalProcIndexIdMapD.begin());


        // think dirichlet hanging node linked to two master solved nodes
        if ((localSize + ghostSize) > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          scaleKernel<<<
            (numberVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE *(localSize + ghostSize),
            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberVectors * (localSize + ghostSize),
            dst.begin(),
            inhomoIdsColoredVecFlattenedD.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(scaleKernel,
                             (numberVectors +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE *
                               (localSize + ghostSize),
                             dftfe::utils::DEVICE_BLOCK_SIZE,
                             0,
                             0,
                             numberVectors * (localSize + ghostSize),
                             dst.begin(),
                             inhomoIdsColoredVecFlattenedD.begin());
#  endif


        constraintsMatrixDataInfoDevice.distribute_slave_to_master(
          dst, numberVectors);

        dst.accumulateAddLocallyOwned();

        if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          scaleKernel<<<(numberVectors +
                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberVectors * localSize,
            dst.begin(),
            inhomoIdsColoredVecFlattenedD.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(scaleKernel,
                             (numberVectors +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                             dftfe::utils::DEVICE_BLOCK_SIZE,
                             0,
                             0,
                             numberVectors * localSize,
                             dst.begin(),
                             inhomoIdsColoredVecFlattenedD.begin());
#  endif

        // src.zero_out_ghosts();
        // constraintsMatrixDataInfoDevice.set_zero(src,numberVectors);
      }

      void
      precondition_Jacobi(const double *     src,
                          const double *     diagonalA,
                          const unsigned int numberVectors,
                          const unsigned int localSize,
                          double *           dst)
      {
        if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          diagScaleKernel<<<(numberVectors +
                             (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                              dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberVectors, localSize, src, diagonalA, dst);
#  elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(diagScaleKernel,
                             (numberVectors +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                             dftfe::utils::DEVICE_BLOCK_SIZE,
                             0,
                             0,
                             numberVectors,
                             localSize,
                             src,
                             diagonalA,
                             dst);
#  endif
      }

      void
      computeResidualSq(dftfe::utils::deviceBlasHandle_t &handle,
                        const double *                    vec1,
                        const double *                    vec2,
                        double *                          vecTemp,
                        const double *                    onesVec,
                        const unsigned int                numberVectors,
                        const unsigned int                localSize,
                        double *                          residualNormSq)
      {
        if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          dotProductContributionBlockedKernel<<<
            (numberVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
            dftfe::utils::DEVICE_BLOCK_SIZE>>>(numberVectors * localSize,
                                               vec1,
                                               vec2,
                                               vecTemp);
#  elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(dotProductContributionBlockedKernel,
                             (numberVectors +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                             dftfe::utils::DEVICE_BLOCK_SIZE,
                             0,
                             0,
                             numberVectors * localSize,
                             vec1,
                             vec2,
                             vecTemp);
#  endif

        const double alpha = 1.0, beta = 0.0;
        dftfe::utils::deviceBlasWrapper::gemm(handle,
                                              dftfe::utils::DEVICEBLAS_OP_N,
                                              dftfe::utils::DEVICEBLAS_OP_T,
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
        &                                      BLASWrapperPtr,
      const dealii::MatrixFree<3, double> &    matrixFreeData,
      const unsigned int                       mfDofHandlerIndex,
      const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
      const double *                           bH,
      const double *                           diagonalAH,
      const double *                           inhomoIdsColoredVecFlattenedH,
      const unsigned int                       localSize,
      const unsigned int                       ghostSize,
      const unsigned int                       numberBins,
      const MPI_Comm &                         mpiCommParent,
      const MPI_Comm &                         mpiCommDomain,
      double *                                 xH,
      const int                                verbosity,
      const unsigned int                       maxLinearSolverIterations,
      const double                             absLinearSolverTolerance,
      const bool isElectroFEOrderDifferentFromFEOrder)
    {
      int this_process;
      MPI_Comm_rank(mpiCommParent, &this_process);

      const unsigned int blockSize = numberBins;
      const unsigned int totalLocallyOwnedCells =
        matrixFreeData.n_physical_cells();
      const unsigned int numberNodesPerElement =
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
      ////////
      /*
      std::cout<<"flattened array:
      "<<flattenedArray.get_partitioner()->n_ghost_indices()<<std::endl;

      std::cout<<"Multivector: "<<xD.ghostSize()*xD.numVectors()<<std::endl;

      dftfe::utils::MemoryManager<double,dftfe::utils::MemorySpace::DEVICE>::set(xD.localSize()*xD.numVectors(),
      xD.begin(), 1.0); xD.accumulateAddLocallyOwned(); std::vector<double>
      xHVec(xD.localSize()*xD.numVectors());
      dftfe::utils::deviceMemcpyD2H(&xHVec[0],
                                    xD.begin(),
                                    xD.localSize()*xD.numVectors() *
      sizeof(double));

      for (unsigned int i=0;i<xD.localSize()*xD.numVectors();i++)
        *(flattenedArray.begin()+i)=1.0;

      flattenedArray.compress(dealii::VectorOperation::add);


      //for (unsigned int i=0;i<xD.locallyOwnedSize()*xD.numVectors();i++)
      //  std::cout<<*(flattenedArray.begin()+i)<<" "<<xHVec[i]<<std::endl;

      dftfe::linearAlgebra::MultiVector<double,
                                        dftfe::utils::MemorySpace::HOST> xHPar;
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
      blockSize,
      xHPar);

      dftfe::utils::MemoryManager<double,dftfe::utils::MemorySpace::HOST>::set(xHPar.localSize()*xHPar.numVectors(),
      xHPar.begin(), 1.0); xHPar.accumulateAddLocallyOwned();

      //for (unsigned int i=0;i<xD.locallyOwnedSize()*xD.numVectors();i++)
      //  std::cout<<"this process: "<<this_process <<"
      "<<*(flattenedArray.begin()+i)<<" "<<*(xHPar.begin()+i)<<std::endl;

      for (unsigned int i=0;i<xD.locallyOwnedSize()*xD.numVectors();i++)
        std::cout<<"this process: "<<this_process <<" "<<xHVec[i]<<"
      "<<*(xHPar.begin()+i)<<std::endl;

      exit(0);
      */
      /////////
      std::vector<dealii::types::global_dof_index> cellLocalProcIndexIdMapH;

      vectorTools::computeCellLocalIndexSetMap(
        flattenedArray.getMPIPatternP2P(),
        matrixFreeData,
        mfDofHandlerIndex,
        blockSize,
        cellLocalProcIndexIdMapH);

      dftUtils::constraintMatrixInfoDevice constraintsMatrixDataInfoDevice;
      constraintsMatrixDataInfoDevice.initialize(
        matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
        hangingPeriodicConstraintMatrix);

      constraintsMatrixDataInfoDevice.set_zero(xD, blockSize);

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
      dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
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
                                      sizeof(dealii::types::global_dof_index));

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      time = MPI_Wtime() - time;
      if (verbosity >= 2 && this_process == 0)
        std::cout
          << " poissonDevice::solveVselfInBins: time for mem allocation: "
          << time << std::endl;

      cgSolver(BLASWrapperPtr->getDeviceBlasHandle(),
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
      dftfe::utils::deviceBlasHandle_t &    handle,
      dftUtils::constraintMatrixInfoDevice &constraintsMatrixDataInfoDevice,
      const double *                        bD,
      const double *                        diagonalAD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &poissonCellStiffnessMatricesD,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &inhomoIdsColoredVecFlattenedD,
      const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                           cellLocalProcIndexIdMapD,
      const unsigned int            localSize,
      const unsigned int            ghostSize,
      const unsigned int            numberBins,
      const unsigned int            totalLocallyOwnedCells,
      const unsigned int            numberNodesPerElement,
      const int                     debugLevel,
      const unsigned int            maxIter,
      const double                  absTol,
      const MPI_Comm &              mpiCommParent,
      const MPI_Comm &              mpiCommDomain,
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
      const unsigned int inc = 1;

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
      // unsigned int localSize = b.size();


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

      computeAX(handle,
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
      dftfe::utils::deviceBlasWrapper::copy(
        handle, localSize * numberBins, bD, inc, r.begin(), inc);


      // r = b - Ax i.e r - Ax
      dftfe::utils::deviceBlasWrapper::axpy(handle,
                                            localSize * numberBins,
                                            &negOne,
                                            Ax.begin(),
                                            inc,
                                            r.begin(),
                                            inc);


      // precondition r
      // dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE>
      // d; d.resize(localSize,0.0);

      // precondition_Jacobi(r,d);
      precondition_Jacobi(
        r.begin(), diagonalAD, numberBins, localSize, d.begin());


      computeResidualSq(handle,
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

      unsigned int iterationNumber = 0;

      computeResidualSq(handle,
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
          for (unsigned int i = 0; i < numberBins; i++)
            std::cout
              << "Device based Linear Conjugate Gradient solver for bin: " << i
              << " started with residual norm squared: " << residualNormSqH[i]
              << std::endl;
        }

      for (unsigned int iter = 0; iter < maxIter; ++iter)
        {
          // q = Ad
          // computeAX(d,q);


          computeAX(handle,
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
          computeResidualSq(handle,
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

          // for (unsigned int i=0;i <numberBins; i++)
          //   std::cout<< "scalar "<<scalarH[i]<<std::endl;

          for (unsigned int i = 0; i < numberBins; i++)
            alphaH[i] = delta_newH[i] / scalarH[i];

          // for (unsigned int i=0;i <numberBins; i++)
          //   std::cout<< "alpha "<<alphaH[i]<<std::endl;

          dftfe::utils::deviceMemcpyH2D(alphaD.begin(),
                                        &alphaH[0],
                                        numberBins * sizeof(double));

          // update x; x = x + alpha*d
          if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            daxpyBlockedKernel<<<(numberBins +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
              numberBins, localSize, d.begin(), alphaD.begin(), x.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
            hipLaunchKernelGGL(daxpyBlockedKernel,
                               (numberBins +
                                (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                 dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                               0,
                               0,
                               numberBins,
                               localSize,
                               d.begin(),
                               alphaD.begin(),
                               x.begin());
#  endif

          if (iter % 50 == 0)
            {
              // r = b
              dftfe::utils::deviceBlasWrapper::copy(
                handle, localSize * numberBins, bD, inc, r.begin(), inc);

              // computeAX(x,Ax);

              computeAX(handle,
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
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                daxpyBlockedKernel<<<
                  (numberBins + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(numberBins,
                                                     localSize,
                                                     Ax.begin(),
                                                     negOneD.begin(),
                                                     r.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
                hipLaunchKernelGGL(daxpyBlockedKernel,
                                   (numberBins +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE *
                                     localSize,
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                   0,
                                   0,
                                   numberBins,
                                   localSize,
                                   Ax.begin(),
                                   negOneD.begin(),
                                   r.begin());
#  endif
            }
          else
            {
              // negAlphaD = -alpha;
              if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                dmaxpyBlockedKernel<<<
                  (numberBins + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                  numberBins, localSize, q.begin(), alphaD.begin(), r.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
                hipLaunchKernelGGL(dmaxpyBlockedKernel,
                                   (numberBins +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE *
                                     localSize,
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                   0,
                                   0,
                                   numberBins,
                                   localSize,
                                   q.begin(),
                                   alphaD.begin(),
                                   r.begin());
#  endif
            }

          // precondition_Jacobi(r,s);
          precondition_Jacobi(
            r.begin(), diagonalAD, numberBins, localSize, s.begin());

          delta_oldD = delta_newD;

          dftfe::utils::deviceMemcpyD2H(&delta_oldH[0],
                                        delta_oldD.begin(),
                                        numberBins * sizeof(double));


          // delta_new = r*s;
          computeResidualSq(handle,
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


          // for (unsigned int i=0;i <numberBins; i++)
          //   std::cout<< "delta_new "<<delta_newH[i]<<std::endl;

          for (unsigned int i = 0; i < numberBins; i++)
            betaH[i] = delta_newH[i] / delta_oldH[i];

          dftfe::utils::deviceMemcpyH2D(betaD.begin(),
                                        &betaH[0],
                                        numberBins * sizeof(double));

          dftfe::utils::deviceMemcpyH2D(delta_newD.begin(),
                                        &delta_newH[0],
                                        numberBins * sizeof(double));

          // d *= beta;
          if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            scaleBlockedKernel<<<(numberBins +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
              numberBins, localSize, d.begin(), betaD.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
            hipLaunchKernelGGL(scaleBlockedKernel,
                               (numberBins +
                                (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                 dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                               0,
                               0,
                               numberBins,
                               localSize,
                               d.begin(),
                               betaD.begin());
#  endif

          // d.add(1.0,s);
          if (localSize > 0)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            daxpyBlockedKernel<<<(numberBins +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
              numberBins, localSize, s.begin(), posOneD.begin(), d.begin());
#  elif DFTFE_WITH_DEVICE_LANG_HIP
            hipLaunchKernelGGL(daxpyBlockedKernel,
                               (numberBins +
                                (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                 dftfe::utils::DEVICE_BLOCK_SIZE * localSize,
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                               0,
                               0,
                               numberBins,
                               localSize,
                               s.begin(),
                               posOneD.begin(),
                               d.begin());
#  endif
          unsigned int isBreak = 1;
          // if(delta_new < relTolerance*relTolerance*delta_0)
          //  isBreak = 1;

          for (unsigned int i = 0; i < numberBins; i++)
            if (delta_newH[i] > absTol * absTol)
              isBreak = 0;

          if (isBreak == 1)
            break;

          iterationNumber += 1;
        }



      // compute residual norm at end
      computeResidualSq(handle,
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
      unsigned int solveStatus = 1;

      if (iterationNumber == maxIter)
        solveStatus = 0;


      if (debugLevel >= 2 && this_process == 0)
        {
          if (solveStatus == 1)
            {
              for (unsigned int i = 0; i < numberBins; i++)
                std::cout << "Linear Conjugate Gradient solver for bin: " << i
                          << " converged after " << iterationNumber + 1
                          << " iterations. with residual norm squared "
                          << residualNormSqH[i] << std::endl;
            }
          else
            {
              for (unsigned int i = 0; i < numberBins; i++)
                std::cout << "Linear Conjugate Gradient solver for bin: " << i
                          << " failed to converge after " << iterationNumber
                          << " iterations. with residual norm squared "
                          << residualNormSqH[i] << std::endl;
            }
        }


      // problem.setX();
      x.updateGhostValues();
      constraintsMatrixDataInfoDevice.distribute(x, numberBins);
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
