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
// @author Gourab Panigrahi
//

#include <linearSolverCGDevice.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryTransfer.h>

namespace dftfe
{
  template <typename Type, int blockSize>
  __global__ void
  applyPreconditionAndComputeDotProductKernel(Type *      d_dvec,
                                              Type *      d_devSum,
                                              const Type *d_rvec,
                                              const Type *d_jacobi,
                                              const int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type jacobi = d_jacobi[idx];
        Type r      = d_rvec[idx];

        localSum    = jacobi * r * r;
        d_dvec[idx] = jacobi * r;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type jacobi = d_jacobi[idx + blockSize];
        Type r      = d_rvec[idx + blockSize];
        localSum += jacobi * r * r;
        d_dvec[idx + blockSize] = jacobi * r;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }


  template <typename Type, int blockSize>
  __global__ void
  applyPreconditionComputeDotProductAndSaddKernel(Type *      d_qvec,
                                                  Type *      d_devSum,
                                                  const Type *d_rvec,
                                                  const Type *d_jacobi,
                                                  const int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type jacobi = d_jacobi[idx];
        Type r      = d_rvec[idx];

        localSum    = jacobi * r * r;
        d_qvec[idx] = -1 * jacobi * r;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type jacobi = d_jacobi[idx + blockSize];
        Type r      = d_rvec[idx + blockSize];
        localSum += jacobi * r * r;
        d_qvec[idx + blockSize] = -1 * jacobi * r;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];
        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }


  template <typename Type, int blockSize>
  __global__ void
  scaleXRandComputeNormKernel(Type *      x,
                              Type *      d_rvec,
                              Type *      d_devSum,
                              const Type *d_qvec,
                              const Type *d_dvec,
                              const Type  alpha,
                              const int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type rNew;
        Type rOld = d_rvec[idx];
        x[idx] += alpha * d_qvec[idx];
        rNew        = rOld + alpha * d_dvec[idx];
        localSum    = rNew * rNew;
        d_rvec[idx] = rNew;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type rNew;
        Type rOld = d_rvec[idx + blockSize];
        x[idx + blockSize] += alpha * d_qvec[idx + blockSize];
        rNew = rOld + alpha * d_dvec[idx + blockSize];
        localSum += rNew * rNew;
        d_rvec[idx + blockSize] = rNew;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }

  // constructor
  linearSolverCGDevice::linearSolverCGDevice(
    const MPI_Comm & mpi_comm_parent,
    const MPI_Comm & mpi_comm_domain,
    const solverType type,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , d_type(type)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_BLASWrapperPtr(BLASWrapperPtr)
  {}


  // solve
  void
  linearSolverCGDevice::solve(linearSolverProblemDevice &problem,
                              const double               absTolerance,
                              const unsigned int         maxNumberIterations,
                              const int                  debugLevel,
                              bool                       distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double start_time = MPI_Wtime();
    double time;

    // compute RHS
    distributedCPUVec<double> rhsHost;
    problem.computeRhs(rhsHost);

    distributedDeviceVec<double> &x = problem.getX();
    distributedDeviceVec<double>  rhsDevice;
    rhsDevice.reinit(x);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(rhsDevice.locallyOwnedSize() *
                                               rhsDevice.numVectors(),
                                             rhsDevice.begin(),
                                             rhsHost.begin());


    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (debugLevel >= 4)
      pcout << "Time for compute rhsHost and copy to Device: "
            << time - start_time << std::endl;


    distributedDeviceVec<double> &d_Jacobi = problem.getPreconditioner();

    d_devSum.resize(1);
    d_devSumPtr = d_devSum.data();
    d_xLocalDof = x.locallyOwnedSize() * x.numVectors();

    double res = 0.0, initial_res = 0.0;
    bool   conv = false;
    int    it   = 0;

    try
      {
        x.updateGhostValues();

        if (d_type == CG)
          {
            // resize the vectors, but do not set the values since they'd be
            // overwritten soon anyway.
            d_qvec.reinit(x);
            d_rvec.reinit(x);
            d_dvec.reinit(x);

            d_qvec.zeroOutGhosts();
            d_rvec.zeroOutGhosts();
            d_dvec.zeroOutGhosts();

            double alpha = 0.0;
            double beta  = 0.0;
            double delta = 0.0;
            // r = Ax
            problem.computeAX(d_rvec, x);

            // r = Ax - rhs
            // dftfe::utils::deviceKernelsGeneric::add(d_rvec.begin(),
            //                                         rhsDevice.begin(),
            //                                         -1.,
            //                                         d_xLocalDof,
            //                                         deviceBlasHandle);
            d_BLASWrapperPtr->add(d_rvec.begin(),
                                  rhsDevice.begin(),
                                  -1,
                                  d_xLocalDof);
            // res = r.r
            // res = dftfe::utils::deviceKernelsGeneric::l2_norm(d_rvec.begin(),
            //                                                   d_xLocalDof,
            //                                                   mpi_communicator,
            //                                                   deviceBlasHandle);

            d_BLASWrapperPtr->xnrm2(
              d_xLocalDof, d_rvec.begin(), 1, mpi_communicator, &res);
            initial_res = res;

            if (res < absTolerance)
              conv = true;
            if (conv)
              return;

            while ((!conv) && (it < maxNumberIterations))
              {
                it++;

                if (it > 1)
                  {
                    beta = delta;
                    AssertThrow(std::abs(beta) != 0.,
                                dealii::ExcMessage("Division by zero\n"));

                    // d = M^(-1) * r
                    // delta = d.r
                    delta =
                      applyPreconditionAndComputeDotProduct(d_Jacobi.begin());

                    beta = delta / beta;

                    // q = beta * q - d
                    dftfe::utils::deviceKernelsGeneric::sadd<double>(
                      d_qvec.begin(), d_dvec.begin(), beta, d_xLocalDof);
                  }
                else
                  {
                    // delta = r.(M^(-1) * r)
                    // q = -M^(-1) * r
                    delta = applyPreconditionComputeDotProductAndSadd(
                      d_Jacobi.begin());
                  }

                // d = Aq
                problem.computeAX(d_dvec, d_qvec);

                // alpha = q.d
                // alpha =
                //   dftfe::utils::deviceKernelsGeneric::dot(d_qvec.begin(),
                //                                           d_dvec.begin(),
                //                                           d_xLocalDof,
                //                                           mpi_communicator,
                //                                           deviceBlasHandle);

                d_BLASWrapperPtr->xdot(d_xLocalDof,
                                       d_qvec.begin(),
                                       1,
                                       d_dvec.begin(),
                                       1,
                                       mpi_communicator,
                                       &alpha);

                AssertThrow(std::abs(alpha) != 0.,
                            dealii::ExcMessage("Division by zero\n"));
                alpha = delta / alpha;

                // res = r.r
                // r += alpha * d
                // x += alpha * q
                res = scaleXRandComputeNorm(x.begin(), alpha);

                if (res < absTolerance)
                  conv = true;
              }

            if (!conv)
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "DFT-FE Error: Solver did not converge\n"));
              }
          }
        else if (d_type == GMRES)
          {
            AssertThrow(false,
                        dealii::ExcMessage("DFT-FE Error: Not implemented"));
          }

        x.updateGhostValues();

        if (distributeFlag)
          problem.distributeX();

        problem.copyXfromDeviceToHost();
      }

    catch (...)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Poisson solver did not converge as per set tolerances. consider increasing MAXIMUM ITERATIONS in Poisson problem parameters. In rare cases for all-electron problems this can also occur due to a known parallel constraints issue in dealii library. Try using set CONSTRAINTS FROM SERIAL DOFHANDLER=true under the Boundary conditions subsection."));
        pcout
          << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
        pcout << "Current abs. residual in Device: " << res << std::endl;
      }

    if (debugLevel >= 2)
      {
        pcout << std::endl;
        pcout << "initial abs. residual in Device: " << initial_res
              << " , current abs. residual in Device: " << res
              << " , nsteps: " << it
              << " , abs. tolerance criterion in Device:  " << absTolerance
              << "\n\n";
      }

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime() - time;

    if (debugLevel >= 4)
      pcout << "Time for Device Poisson/Helmholtz problem CG iterations: "
            << time << std::endl;
  }


  double
  linearSolverCGDevice::applyPreconditionAndComputeDotProduct(
    const double *d_jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks =
      (d_xLocalDof + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
      (dftfe::utils::DEVICE_BLOCK_SIZE * 2);

    dftfe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    applyPreconditionAndComputeDotProductKernel<double,
                                                dftfe::utils::DEVICE_BLOCK_SIZE>
      <<<blocks, dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        d_dvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         applyPreconditionAndComputeDotProductKernel<
                           double,
                           dftfe::utils::DEVICE_BLOCK_SIZE>),
                       blocks,
                       dftfe::utils::DEVICE_BLOCK_SIZE,
                       0,
                       0,
                       d_dvec.begin(),
                       d_devSumPtr,
                       d_rvec.begin(),
                       d_jacobi,
                       d_xLocalDof);
#endif

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGDevice::applyPreconditionComputeDotProductAndSadd(
    const double *d_jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks =
      (d_xLocalDof + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
      (dftfe::utils::DEVICE_BLOCK_SIZE * 2);

    dftfe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    applyPreconditionComputeDotProductAndSaddKernel<
      double,
      dftfe::utils::DEVICE_BLOCK_SIZE>
      <<<blocks, dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        d_qvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         applyPreconditionComputeDotProductAndSaddKernel<
                           double,
                           dftfe::utils::DEVICE_BLOCK_SIZE>),
                       blocks,
                       dftfe::utils::DEVICE_BLOCK_SIZE,
                       0,
                       0,
                       d_qvec.begin(),
                       d_devSumPtr,
                       d_rvec.begin(),
                       d_jacobi,
                       d_xLocalDof);
#endif

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGDevice::scaleXRandComputeNorm(double *x, const double &alpha)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks =
      (d_xLocalDof + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
      (dftfe::utils::DEVICE_BLOCK_SIZE * 2);

    dftfe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>
      <<<blocks, dftfe::utils::DEVICE_BLOCK_SIZE>>>(x,
                                                    d_rvec.begin(),
                                                    d_devSumPtr,
                                                    d_qvec.begin(),
                                                    d_dvec.begin(),
                                                    alpha,
                                                    d_xLocalDof);
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
        scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      x,
      d_rvec.begin(),
      d_devSumPtr,
      d_qvec.begin(),
      d_dvec.begin(),
      alpha,
      d_xLocalDof);
#endif

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return std::sqrt(sum);
  }

} // namespace dftfe
