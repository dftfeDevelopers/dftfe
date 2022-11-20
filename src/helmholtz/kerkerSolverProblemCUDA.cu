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
#include <kerkerSolverProblemCUDA.h>

namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrderElectro>
  kerkerSolverProblemCUDA<FEOrderElectro>::kerkerSolverProblemCUDA(
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
  void kerkerSolverProblemCUDA<FEOrderElectro>::init(
    dealii::MatrixFree<3, double> &    matrixFreeDataPRefined,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double> &        x,
    double                             kerkerMixingParameter,
    const unsigned int                 matrixFreeVectorComponent,
    const unsigned int                 matrixFreeQuadratureComponent)
  {
    d_matrixFreeDataPRefinedPtr     = &matrixFreeDataPRefined;
    d_constraintMatrixPRefinedPtr   = &constraintMatrixPRefined;
    d_gamma                         = kerkerMixingParameter;
    d_matrixFreeVectorComponent     = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
    d_nLocalCells = d_matrixFreeDataPRefinedPtr->n_macro_cells();

    matrixFreeDataPRefined.initialize_dof_vector(x,
                                                 d_matrixFreeVectorComponent);
    d_xDevice.reinit(x.get_partitioner(), 1);

    d_xPtr      = &x;
    d_xLocalDof = d_xDevice.locallyOwnedDofsSize();
    d_xLen = d_xDevice.locallyOwnedDofsSize() + d_xDevice.ghostFlattenedSize();

    computeDiagonalA();

    // Setup MatrixFree Mesh
    setupMatrixFree();

    // Setup MatrixFree Constraints
    setupconstraints();
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::reinit(
    distributedCPUVec<double> &                          x,
    const std::map<dealii::CellId, std::vector<double>> &quadPointValues)
  {
    d_xPtr                      = &x;
    d_quadGradResidualValuesPtr = &quadPointValues;

    cudaUtils::copyHostVecToCUDAVec<double>(d_xPtr->begin(),
                                            d_xDevice.begin(),
                                            d_xLocalDof);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::setupconstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      *d_constraintMatrixPRefinedPtr);
    d_constraintsTotalPotentialInfo.precomputeMaps(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      d_xPtr->get_partitioner(),
      1);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice, 1);
  }


  template <unsigned int FEOrderElectro>
  distributedGPUVec<double> &
  kerkerSolverProblemCUDA<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::copyXfromDeviceToHost()
  {
    cudaUtils::copyCUDAVecToHostVec<double>(d_xDevice.begin(),
                                            d_xPtr->begin(),
                                            d_xLen);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::FEEvaluation<3, FEOrderElectro, C_num1DQuad<FEOrderElectro>()>
      fe_eval(*d_matrixFreeDataPRefinedPtr,
              d_matrixFreeVectorComponent,
              d_matrixFreeQuadratureComponent);

    Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
    for (unsigned int idim = 0; idim < 3; idim++)
      zeroTensor[idim] = make_vectorized_array(0.0);


    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      residualGradQuads(fe_eval.n_q_points, zeroTensor);
    for (unsigned int macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_macro_cells();
         ++macrocell)
      {
        std::fill(residualGradQuads.begin(),
                  residualGradQuads.end(),
                  zeroTensor);
        const unsigned int numSubCells =
          d_matrixFreeDataPRefinedPtr->n_components_filled(macrocell);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            dealii::CellId             subCellId = subCellPtr->id();
            const std::vector<double> &tempVec =
              d_quadGradResidualValuesPtr->find(subCellId)->second;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                residualGradQuads[q][0][iSubCell] = tempVec[3 * q + 0];
                residualGradQuads[q][1][iSubCell] = tempVec[3 * q + 1];
                residualGradQuads[q][2][iSubCell] = tempVec[3 * q + 2];
              }
          }

        fe_eval.reinit(macrocell);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_gradient(residualGradQuads[q], q);

        fe_eval.integrate(false, true);

        fe_eval.distribute_local_to_global(rhs);
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::computeDiagonalA()
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
    d_diagonalAdevice.reinit(d_diagonalA.get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(d_diagonalA.begin(),
                                            d_diagonalAdevice.begin(),
                                            d_xLocalDof);
  }


  template <unsigned int FEOrderElectro>
  distributedGPUVec<double> &
  kerkerSolverProblemCUDA<FEOrderElectro>::getPreconditioner()
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
                  Type        coeffHelmholtz)
  {
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // PT(q*p), D(q*q), P(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[M * K];
    Type *sharedZ  = &sharedY[M * K];
    Type *sharedT  = &sharedZ[M * K];
    Type *sharedPT = &sharedT[M * K];
    Type *sharedD  = &sharedPT[N * K];
    Type *sharedP  = &sharedD[N * N];
    Type *sharedDT = &sharedP[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const int mapShift = blockIdx.x * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedPT[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // First index is the fastest
    // Interpolation combined with Extraction
    // T -> PPPU
    // X -> TD1
    // Y -> TD2
    // Z -> TD3

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
              x[j] += sharedPT[j + k * N] * u[k];
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
              y[j] += sharedPT[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + j * K + b * K * N] = y[j];
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
              x[j] += sharedPT[j + k * N] * y[k];
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
            x[k] = sharedX[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + j * N + b * N * N] = z[j];
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

    // Copy Jacobian Action to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + JShift];

    Type detJ;

    __syncthreads();

    // Gemm with Jacobian Action
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
    // X -> TDT1
    // Y -> TDT2
    // Z -> TDT3
    // T -> PPPU

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
            z[k] = sharedZ[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + j * N + b * N * N] += y[j];
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
              y[j] += sharedP[j + k * K] * x[k];
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
            y[k] = sharedY[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedP[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + j * N + b * N * K] = x[j];
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
              y[j] += sharedP[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::setupMatrixFree()
  {
    constexpr int p   = FEOrderElectro + 1;
    constexpr int dim = 3;
    constexpr int q   = p;

    // shape info helps to obtain reference cell basis function and lex
    // numbering
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);
    const int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;

    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shapeInfo;

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPRefinedPtr->get_quadrature(
        d_matrixFreeQuadratureComponent);

    int numQuadPoints = std::cbrt(quadrature.size());

    dealii::QGauss<1> quad(numQuadPoints);
    shapeInfo.reinit(quad, dofHandler.get_fe());
    std::vector<unsigned int> lexMap3D = shapeInfo.lexicographic_numbering;

    const auto shapeValue = shapeInfo.data.front().shape_values;
    const auto shapeGradquad =
      shapeInfo.data.front().shape_gradients_collocation;

    dealii::FE_Q<1> feCell1D(FEOrderElectro);
    shapeInfo.reinit(quad, feCell1D);
    std::vector<unsigned int> lexMap1D = shapeInfo.lexicographic_numbering;
    std::vector<double>       quadWeights(q);

    for (int j = 0; j < q; j++)
      quadWeights[j] = quad.weight(lexMap1D[j]);

    thrust::host_vector<double> spVG(2 * p * q + 2 * q * q);

    for (int i = 0; i < p; i++)
      for (int j = 0; j < q; j++)
        {
          // PT(q*p), DT(q*q), P(p*q), D(q*q)
          double value =
            shapeValue[lexMap1D[j] + i * p] * std::sqrt(quadWeights[j]);
          double grad = shapeGradquad[lexMap1D[j] + lexMap1D[i] * p] *
                        std::sqrt(quadWeights[j]) / std::sqrt(quadWeights[i]);

          spVG[j + i * q]                     = value;
          spVG[j + i * q + q * p]             = grad;
          spVG[i + j * p + q * p + q * q]     = value;
          spVG[i + j * q + 2 * q * p + q * q] = grad;
        }

    // Map making
    thrust::host_vector<int> map(dofs_per_cell * d_nLocalCells);
    std::vector<dealii::types::global_dof_index> local_dof_globalIndices(
      dofs_per_cell);

    // Lexicographic Map making
    int cellIdx = 0;
    for (const auto &cell : dofHandler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_globalIndices);

            for (int dofIdx = 0; dofIdx < dofs_per_cell; dofIdx++)
              {
                dealii::types::global_dof_index globalIdx =
                  local_dof_globalIndices[lexMap3D[dofIdx]];
                int localIdx =
                  d_xPtr->get_partitioner()->global_to_local(globalIdx);
                map[dofIdx + cellIdx * dofs_per_cell] = localIdx;
              }
            cellIdx++;
          }
      }

    // Jacobian
    dealii::QGauss<dim> quadrature_formula(dofHandler.get_fe().degree + 1);
    const int           qPoints = quadrature_formula.size();

    dealii::FEValues<dim> fe_values(dofHandler.get_fe(),
                                    quadrature_formula,
                                    dealii::update_inverse_jacobians |
                                      dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    std::vector<dealii::DerivativeForm<1, dim, dim>> inv_jacobians_tensor;
    std::vector<double> detJacobian(d_nLocalCells * qPoints),
      invJac(d_nLocalCells * dim * dim);
    thrust::host_vector<double> jacobianAction(d_nLocalCells * dim * dim);

    cellIdx = 0;
    for (const auto &cell : dofHandler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            inv_jacobians_tensor = fe_values.get_inverse_jacobians();

            for (int d = 0; d < dim; d++)
              for (int e = 0; e < dim; e++)
                invJac[d + e * dim + cellIdx * dim * dim] =
                  inv_jacobians_tensor[0][d][e];

            for (int i = 0; i < qPoints; i++)
              detJacobian[i + cellIdx * qPoints] =
                fe_values.JxW(lexMap3D[i]) /
                quadrature_formula.weight(lexMap3D[i]);

            cellIdx++;
          }
      }

    for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
      for (int d = 0; d < dim; d++)
        for (int e = 0; e < dim; e++)
          for (int f = 0; f < dim; f++)
            jacobianAction[e + d * dim + cellIdx * dim * dim] +=
              invJac[d + f * dim + cellIdx * dim * dim] *
              invJac[e + f * dim + cellIdx * dim * dim] *
              detJacobian[cellIdx * qPoints];

    // Construct the device vectors
    d_shapeFunctionAll = spVG;
    d_jacobianAction   = jacobianAction;
    d_map              = map;

    d_shapeFunctionAllPtr = thrust::raw_pointer_cast(d_shapeFunctionAll.data());
    d_jacobianActionPtr   = thrust::raw_pointer_cast(d_jacobianAction.data());
    d_mapPtr              = thrust::raw_pointer_cast(d_map.data());

    constexpr size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

    cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblemCUDA<FEOrderElectro>::computeAX(
    distributedGPUVec<double> &Ax,
    distributedGPUVec<double> &x)
  {
    constexpr int dim = 3;
    constexpr int p   = FEOrderElectro + 1;
    constexpr int q   = p;
    constexpr int threads =
      (FEOrderElectro < 7 ? 96 : FEOrderElectro == 7 ? 64 : 256);
    const int        blocks         = d_nLocalCells;
    const double     coeffHelmholtz = 4 * M_PI * d_gamma;
    constexpr size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

    cudaUtils::set<double>(Ax.begin(), 0, d_xLen);

    x.updateGhostValues();

    d_constraintsTotalPotentialInfo.distribute(x, 1);

    computeAXKernel<double, p * p, q, p, dim>
      <<<blocks, threads, smem>>>(Ax.begin(),
                                  x.begin(),
                                  d_shapeFunctionAllPtr,
                                  d_jacobianActionPtr,
                                  d_mapPtr,
                                  coeffHelmholtz);

    d_constraintsTotalPotentialInfo.set_zero(x, 1);

    d_constraintsTotalPotentialInfo.distribute_slave_to_master(Ax, 1);

    Ax.compressAdd();
  }

#include "kerkerSolverProblemCUDA.inst.cu"
} // namespace dftfe
