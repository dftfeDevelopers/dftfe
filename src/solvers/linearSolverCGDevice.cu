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
#include <deviceHelpers.h>

namespace dftfe
{
  // constructor
  linearSolverCGDevice::linearSolverCGDevice(const MPI_Comm & mpi_comm_parent,
                                             const MPI_Comm & mpi_comm_domain,
                                             const solverType type)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , d_type(type)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}


  // solve
  void
  linearSolverCGDevice::solve(linearSolverProblemDevice &problem,
                              const double               absTolerance,
                              const unsigned int         maxNumberIterations,
                              cublasHandle_t &           cublasHandle,
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

    distributedDeviceVec<double> rhsDevice;
    rhsDevice.reinit(rhsHost.get_partitioner(), 1);

    deviceUtils::copyHostVecToDeviceVec<double>(
      rhsHost.begin(), rhsDevice.begin(), rhsDevice.locallyOwnedDofsSize());

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (debugLevel >= 4)
      pcout << "Time for compute rhsHost and copy to Device: "
            << time - start_time << std::endl;


    distributedDeviceVec<double> &x        = problem.getX();
    distributedDeviceVec<double> &d_Jacobi = problem.getPreconditioner();

    d_devSum.resize(1);
    d_devSumPtr = thrust::raw_pointer_cast(d_devSum.data());
    d_xLocalDof = x.locallyOwnedDofsSize();

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
            deviceUtils::add(d_rvec.begin(),
                             rhsDevice.begin(),
                             -1.,
                             d_xLocalDof,
                             cublasHandle);

            // res = r.r
            res = deviceUtils::l2_norm(d_rvec.begin(),
                                       d_xLocalDof,
                                       mpi_communicator,
                                       cublasHandle);

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
                    deviceUtils::sadd<double>(d_qvec.begin(),
                                              d_dvec.begin(),
                                              beta,
                                              d_xLocalDof);
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
                alpha = deviceUtils::dot(d_qvec.begin(),
                                         d_dvec.begin(),
                                         d_xLocalDof,
                                         mpi_communicator,
                                         cublasHandle);

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
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

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
    cooperative_groups::sync(block);

    for (int size = MAXBLOCKSIZE / 2; size >= 4 * WARPSIZE; size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        cooperative_groups::sync(block);
      }

    cooperative_groups::thread_block_tile<WARPSIZE> tileWarp =
      cooperative_groups::tiled_partition<WARPSIZE>(block);

    if (block.thread_rank() < WARPSIZE)
      {
        if (blockSize >= 2 * WARPSIZE)
          localSum += smem[tid + WARPSIZE];

        for (int offset = tileWarp.size() / 2; offset > 0; offset /= 2)
          localSum += tileWarp.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
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
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

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
    cooperative_groups::sync(block);

    for (int size = MAXBLOCKSIZE / 2; size >= 4 * WARPSIZE; size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        cooperative_groups::sync(block);
      }

    cooperative_groups::thread_block_tile<WARPSIZE> tileWarp =
      cooperative_groups::tiled_partition<WARPSIZE>(block);

    if (block.thread_rank() < WARPSIZE)
      {
        if (blockSize >= 2 * WARPSIZE)
          localSum += smem[tid + WARPSIZE];

        for (int offset = tileWarp.size() / 2; offset > 0; offset /= 2)
          localSum += tileWarp.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
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
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

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
    cooperative_groups::sync(block);

    for (int size = MAXBLOCKSIZE / 2; size >= 4 * WARPSIZE; size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        cooperative_groups::sync(block);
      }

    cooperative_groups::thread_block_tile<WARPSIZE> tileWarp =
      cooperative_groups::tiled_partition<WARPSIZE>(block);

    if (block.thread_rank() < WARPSIZE)
      {
        if (blockSize >= 2 * WARPSIZE)
          localSum += smem[tid + WARPSIZE];

        for (int offset = tileWarp.size() / 2; offset > 0; offset /= 2)
          localSum += tileWarp.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&d_devSum[0], localSum);
  }


  double
  linearSolverCGDevice::applyPreconditionAndComputeDotProduct(
    const double *d_jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    cudaMemset(d_devSumPtr, 0, sizeof(double));

    applyPreconditionAndComputeDotProductKernel<double,
                                                cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        d_dvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);

    local_sum = d_devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGDevice::applyPreconditionComputeDotProductAndSadd(
    const double *d_jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    cudaMemset(d_devSumPtr, 0, sizeof(double));

    applyPreconditionComputeDotProductAndSaddKernel<double,
                                                    cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        d_qvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);

    local_sum = d_devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGDevice::scaleXRandComputeNorm(double *x, const double &alpha)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    cudaMemset(d_devSumPtr, 0, sizeof(double));

    scaleXRandComputeNormKernel<double, cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(x,
                                             d_rvec.begin(),
                                             d_devSumPtr,
                                             d_qvec.begin(),
                                             d_dvec.begin(),
                                             alpha,
                                             d_xLocalDof);

    local_sum = d_devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return std::sqrt(sum);
  }

} // namespace dftfe
