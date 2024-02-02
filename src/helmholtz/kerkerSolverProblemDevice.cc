// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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

#include <constants.h>
#include <kerkerSolverProblemDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryTransfer.h>

namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrderElectro>
  kerkerSolverProblemDevice<FEOrderElectro>::kerkerSolverProblemDevice(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::init(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
      &                                basisOperationsPtr,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double> &        x,
    double                             kerkerMixingParameter,
    const unsigned int                 matrixFreeVectorComponent,
    const unsigned int                 matrixFreeQuadratureComponent)
  {
    d_basisOperationsPtr            = basisOperationsPtr;
    d_matrixFreeDataPRefinedPtr     = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPRefinedPtr   = &constraintMatrixPRefined;
    d_gamma                         = kerkerMixingParameter;
    d_matrixFreeVectorComponent     = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
    d_nLocalCells = d_matrixFreeDataPRefinedPtr->n_cell_batches();

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      x.get_partitioner(), 1, d_xDevice);


    d_xPtr      = &x;
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();

    computeDiagonalA();

    // Setup MatrixFree Mesh
    setupMatrixFree();

    // Setup MatrixFree Constraints
    setupconstraints();
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::reinit(
    distributedCPUVec<double> &x,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPointValues)
  {
    d_xPtr                  = &x;
    d_residualQuadValuesPtr = &quadPointValues;

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_xDevice.begin(),
                                             d_xPtr->begin());
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setupconstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      *d_constraintMatrixPRefinedPtr);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice, 1);
  }


  template <unsigned int FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::FEEvaluation<3, FEOrderElectro, C_num1DQuad<FEOrderElectro>()>
      fe_eval(*d_matrixFreeDataPRefinedPtr,
              d_matrixFreeVectorComponent,
              d_matrixFreeQuadratureComponent);

    dealii::VectorizedArray<double> zeroVec = 0.0;

    dealii::AlignedVector<dealii::VectorizedArray<double>> residualQuads(
      fe_eval.n_q_points, zeroVec);
    for (unsigned int macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_cell_batches();
         ++macrocell)
      {
        std::fill(residualQuads.begin(), residualQuads.end(), zeroVec);
        const unsigned int numSubCells =
          d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
            macrocell);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            dealii::CellId     subCellId = subCellPtr->id();
            const unsigned int cellIndex =
              d_basisOperationsPtr->cellIndex(subCellId);
            const double *tempVec =
              d_residualQuadValuesPtr->data() + fe_eval.n_q_points * cellIndex;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              residualQuads[q][iSubCell] = -tempVec[q];
          }

        fe_eval.reinit(macrocell);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_value(residualQuads[q], q);

        fe_eval.integrate(true, false);

        fe_eval.distribute_local_to_global(rhs);
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    dealii::QGauss<3>      quadrature(C_num1DQuad<FEOrderElectro>());
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
    const unsigned int     dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> elementalDiagonalA(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          elementalDiagonalA = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) +=
                (fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(i, q_point) +
                 4 * M_PI * d_gamma * fe_values.shape_value(i, q_point) *
                   fe_values.shape_value(i, q_point)) *
                fe_values.JxW(q_point);

          d_constraintMatrixPRefinedPtr->distribute_local_to_global(
            elementalDiagonalA, local_dof_indices, d_diagonalA);
        }

    // MPI operation to sync data
    d_diagonalA.compress(dealii::VectorOperation::add);

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          d_diagonalA(i) = 1.0 / d_diagonalA(i);

    d_diagonalA.compress(dealii::VectorOperation::insert);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_diagonalA.get_partitioner(), 1, d_diagonalAdevice);


    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_diagonalAdevice.begin(),
                                             d_diagonalA.begin());
  }


  template <unsigned int FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <typename Type, int M, int N, int K, int dim>
  __global__ void
  computeAXKernel(Type *      V,
                  const Type *U,
                  const Type *P,
                  const Type *J,
                  const int * map,
                  const Type  coeffHelmholtz)
  {
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const int mapShift = blockIdx.x * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedP[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type x[N], u[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of P
    // Y Direction
    for (int i = threadIdx.x; i < K * N; i += blockDim.x)
      {
        Type y[N], x[K];

        int a = i % K;
        int b = i / K;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

    __syncthreads();

    // 3rd GEMM of P
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

    __syncthreads();

    // 1st GEMM of D
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], x[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type t[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          t[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const int JShift = blockIdx.x * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + JShift];

    Type detJ;

    __syncthreads();

    // Gemm with Jacobian Factor
#pragma unroll
    for (int i = threadIdx.x; i < N * N * N; i += blockDim.x)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];

        detJ =
          sharedJ[0] * (sharedJ[4] * sharedJ[8] - sharedJ[5] * sharedJ[7]) -
          sharedJ[1] * (sharedJ[3] * sharedJ[8] - sharedJ[5] * sharedJ[6]) +
          sharedJ[2] * (sharedJ[3] * sharedJ[7] - sharedJ[4] * sharedJ[6]);
      }

    __syncthreads();

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[N], h[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          {
            h[j]                   = sharedX[i + j * N * N];
            sharedX[i + j * N * N] = coeffHelmholtz * detJ * h[j] + x[j];
          }
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], z[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

    __syncthreads();

    // 3rd GEMM of DT
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], t[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

    __syncthreads();

    // 1st GEMM of PT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

    __syncthreads();

    // 2nd GEMM of PT
    // Y Direction
    for (int i = threadIdx.x; i < N * K; i += blockDim.x)
      {
        Type x[K], y[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < K; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

    __syncthreads();

    // 3rd GEMM of PT
    // X Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setupMatrixFree()
  {
    constexpr int p            = FEOrderElectro + 1;
    constexpr int q            = p;
    constexpr int nDofsPerCell = p * p * p;
    constexpr int dim          = 3;

    auto dofInfo =
      d_matrixFreeDataPRefinedPtr->get_dof_info(d_matrixFreeVectorComponent);
    auto shapeInfo = d_matrixFreeDataPRefinedPtr->get_shape_info(
      d_matrixFreeVectorComponent, d_matrixFreeQuadratureComponent);
    auto mappingData = d_matrixFreeDataPRefinedPtr->get_mapping_info()
                         .cell_data[d_matrixFreeQuadratureComponent];
    auto shapeData = shapeInfo.get_shape_data();

    // Shape Function Values, Gradients and their Transposes
    // P(q*p), D(q*q), PT(p*q), DT(q*q)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      shapeFunction(2 * q * (p + q));

    for (int i = 0; i < p; i++)
      for (int j = 0; j < q; j++)
        {
          double value = shapeData.shape_values[j + i * q][0] *
                         std::sqrt(shapeData.quadrature.weight(j));
          shapeFunction[j + i * q]               = value;
          shapeFunction[i + j * p + q * (p + q)] = value;
        }

    for (int i = 0; i < q; i++)
      for (int j = 0; j < q; j++)
        {
          double grad = shapeData.shape_gradients_collocation[j + i * q][0] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
          shapeFunction[j + i * q + q * p]           = grad;
          shapeFunction[i + j * q + (2 * p + q) * q] = grad;
        }

    // Jacobian
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      jacobianFactor(dim * dim * d_nLocalCells);

    auto cellOffsets = mappingData.data_index_offsets;

    for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
      for (int k = 0; k < dim; k++)
        for (int i = 0; i < dim; i++)
          for (int j = 0; j < dim; j++)
            jacobianFactor[j + i * dim + cellIdx * dim * dim] +=
              mappingData
                .JxW_values[cellOffsets[cellIdx / dofInfo.vectorization_length]]
                           [0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][j][0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][i][0];

    // Map making
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::HOST> map(
      nDofsPerCell * d_nLocalCells);

    for (auto cellIdx = 0; cellIdx < d_nLocalCells; ++cellIdx)
      std::memcpy(map.data() + cellIdx * nDofsPerCell,
                  ((dofInfo.row_starts[cellIdx].second ==
                    dofInfo.row_starts[cellIdx + 1].second) &&
                   (dofInfo.row_starts_plain_indices[cellIdx] ==
                    dealii::numbers::invalid_unsigned_int)) ?
                    dofInfo.dof_indices.data() +
                      dofInfo.row_starts[cellIdx].first :
                    dofInfo.plain_dof_indices.data() +
                      dofInfo.row_starts_plain_indices[cellIdx],
                  nDofsPerCell * sizeof(unsigned int));

    // Construct the device vectors
    d_shapeFunction.resize(shapeFunction.size());
    d_shapeFunction.copyFrom(shapeFunction);

    d_jacobianFactor.resize(jacobianFactor.size());
    d_jacobianFactor.copyFrom(jacobianFactor);

    d_map.resize(map.size());
    d_map.copyFrom(map);

    d_shapeFunctionPtr  = d_shapeFunction.data();
    d_jacobianFactorPtr = d_jacobianFactor.data();
    d_mapPtr            = d_map.data();

    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
#endif
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    constexpr int dim     = 3;
    constexpr int p       = FEOrderElectro + 1;
    constexpr int q       = p;
    constexpr int threads = 64;
    // constexpr int threads =
    //  (FEOrderElectro < 7 ? 96 : FEOrderElectro == 7 ? 64 : 256);
    const int             blocks         = d_nLocalCells;
    const double          coeffHelmholtz = 4 * M_PI * d_gamma;
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

    dftfe::utils::deviceMemset(Ax.begin(), 0, d_xLen * sizeof(double));

    x.updateGhostValues();

    d_constraintsTotalPotentialInfo.distribute(x, 1);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeAXKernel<double, p * p, q, p, dim>
      <<<blocks, threads, smem>>>(Ax.begin(),
                                  x.begin(),
                                  d_shapeFunctionPtr,
                                  d_jacobianFactorPtr,
                                  d_mapPtr,
                                  coeffHelmholtz);

#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         computeAXKernel<double, p * p, q, p, dim>),
                       blocks,
                       threads,
                       smem,
                       0,
                       Ax.begin(),
                       x.begin(),
                       d_shapeFunctionPtr,
                       d_jacobianFactorPtr,
                       d_mapPtr,
                       coeffHelmholtz);
#endif

    d_constraintsTotalPotentialInfo.set_zero(x, 1);

    d_constraintsTotalPotentialInfo.distribute_slave_to_master(Ax, 1);

    Ax.accumulateAddLocallyOwned();
  }

#include "kerkerSolverProblemDevice.inst.cc"
} // namespace dftfe
