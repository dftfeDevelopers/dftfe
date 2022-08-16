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

#include <linearSolverCGCUDA.h>
#include <cudaHelpers.h>

namespace dftfe
{
  // constructor
  linearSolverCGCUDA::linearSolverCGCUDA(const MPI_Comm & mpi_comm_parent,
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
  linearSolverCGCUDA::solve(linearSolverProblemCUDA &problem,
                            const double             absTolerance,
                            const unsigned int       maxNumberIterations,
                            cublasHandle_t &         cublasHandle,
                            const unsigned int       debugLevel,
                            bool                     distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double start_time = MPI_Wtime();
    double time;

    // compute RHS
    distributedCPUVec<double> rhs_host;
    problem.computeRhs(rhs_host);

    distributedGPUVec<double> rhs_device;
    rhs_device.reinit(rhs_host.get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(rhs_host.begin(),
                                            rhs_device.begin(),
                                            rhs_device.locallyOwnedDofsSize());

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (debugLevel >= 4)
      pcout << "Time for compute rhs_host: " << time - start_time << std::endl;

    bool conv = false;

    distributedGPUVec<double> &x        = problem.getX();
    distributedGPUVec<double> &d_Jacobi = problem.getPreconditioner();

    devSum.resize(1);
    devSumPtr = thrust::raw_pointer_cast(devSum.data());

    d_xLenLocalDof = x.locallyOwnedDofsSize();

    double res = 0.0, initial_res = 0.0;
    int    it = 0;

    try
      {
        x.updateGhostValues();

        if (d_type == CG)
          {
            // resize the vectors, but do not set the values since they'd be
            // overwritten soon anyway.
            q.reinit(x);
            r.reinit(x);
            d.reinit(x);

            q.zeroOutGhosts();
            r.zeroOutGhosts();
            d.zeroOutGhosts();

            double alpha = 0.0;
            double beta  = 0.0;
            double delta = 0.0;

            // r = Ax
            problem.computeAX(r, x);

            // r = Ax - rhs
            cudaUtils::add<double>(
              r.begin(), rhs_device.begin(), -1., d_xLenLocalDof, cublasHandle);

            // res = l2_norm(r)
            res = cudaUtils::l2_norm<double>(r.begin(),
                                             d_xLenLocalDof,
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

                    // d = M^(-1)r
                    // delta = r * d
                    delta =
                      applyPreconditionAndComputeDotProduct(d_Jacobi.begin());

                    beta = delta / beta;

                    // q = beta * q - d
                    cudaUtils::sadd<double>(q.begin(),
                                            d.begin(),
                                            beta,
                                            d_xLenLocalDof);
                  }
                else
                  {
                    // delta = r * M^(-1)r
                    // q = -M^(-1)r
                    delta = applyPreconditionComputeDotProductAndSadd(
                      d_Jacobi.begin());
                  }

                // d = Aq
                problem.computeAX(d, q);

                // alpha = q * d
                alpha = cudaUtils::dot<double>(q.begin(),
                                               d.begin(),
                                               d_xLenLocalDof,
                                               mpi_communicator,
                                               cublasHandle);

                AssertThrow(std::abs(alpha) != 0.,
                            dealii::ExcMessage("Division by zero\n"));
                alpha = delta / alpha;

                // x = x + alpha * q
                // r + alpha * d
                // res = l2_norm(r)
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

        // x.updateGhostValues();

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
        pcout << "Current abs. residual: " << res << std::endl;
      }

    if (debugLevel >= 2)
      {
        pcout << std::endl;
        pcout << "initial abs. residual: " << initial_res
              << " , current abs. residual: " << res << " , nsteps: " << it
              << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
      }

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime() - time;

    if (debugLevel >= 4)
      pcout << "Time for Poisson/Helmholtz problem CG iterations: " << time
            << std::endl;
  }


  template <typename Type, int blockSize>
  __global__ void
  applyPreconditionAndComputeDotProductKernel(Type *d,
                                              Type *r,
                                              Type *jacobi,
                                              Type *devSum,
                                              int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_jacobi = jacobi[idx];
        Type d_r      = r[idx];

        localSum = d_jacobi * d_r * d_r;
        d[idx]   = d_jacobi * d_r;
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_jacobi = jacobi[idx + blockSize];
        Type d_r      = r[idx + blockSize];
        localSum += d_jacobi * d_r * d_r;
        d[idx + blockSize] = d_jacobi * d_r;
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < 32)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&devSum[0], localSum);
  }


  template <typename Type, int blockSize>
  __global__ void
  applyPreconditionComputeDotProductAndSaddKernel(Type *q,
                                                  Type *r,
                                                  Type *jacobi,
                                                  Type *devSum,
                                                  int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_jacobi = jacobi[idx];
        Type d_r      = r[idx];

        localSum = d_jacobi * d_r * d_r;
        q[idx]   = -1 * d_jacobi * d_r;
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_jacobi = jacobi[idx + blockSize];
        Type d_r      = r[idx + blockSize];
        localSum += d_jacobi * d_r * d_r;
        q[idx + blockSize] = -1 * d_jacobi * d_r;
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < 32)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&devSum[0], localSum);
  }


  template <typename Type, int blockSize>
  __global__ void
  scaleXRandComputeNormKernel(Type *q,
                              Type *r,
                              Type *d,
                              Type *x,
                              Type *devSum,
                              Type  alpha,
                              int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_r = r[idx];
        localSum = d_r * d_r;
        x[idx] += alpha * q[idx];
        r[idx] += alpha * d[idx];
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_r = r[idx + blockSize];
        localSum += d_r * d_r;
        x[idx + blockSize] += alpha * q[idx + blockSize];
        r[idx + blockSize] += alpha * d[idx + blockSize];
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < warpSize)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&devSum[0], localSum);
  }


  double
  linearSolverCGCUDA::applyPreconditionAndComputeDotProduct(double *jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    devSum[0] = 0.0;

    applyPreconditionAndComputeDotProductKernel<double,
                                                cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        d.begin(), r.begin(), jacobi, devSumPtr, d_xLenLocalDof);

    local_sum = devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGCUDA::applyPreconditionComputeDotProductAndSadd(double *jacobi)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    devSum[0] = 0.0;

    applyPreconditionComputeDotProductAndSaddKernel<double,
                                                    cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        q.begin(), r.begin(), jacobi, devSumPtr, d_xLenLocalDof);

    local_sum = devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  double
  linearSolverCGCUDA::scaleXRandComputeNorm(double *x, double &alpha)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    devSum[0] = 0.0;

    scaleXRandComputeNormKernel<double, cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        q.begin(), r.begin(), d.begin(), x, devSumPtr, alpha, d_xLenLocalDof);

    local_sum = devSum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return std::sqrt(sum);
  }

} // namespace dftfe
