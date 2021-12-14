// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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


#include <cudaHelpers.h>
#include <headers.h>
#include <cublas_v2.h>

namespace dftfe
{
  namespace
  {
    template <typename NumberType>
    __global__ void
    setKernel(const dataTypes::local_size_type size,
              const NumberType                 s,
              NumberType *                     arr)
    {
      const dataTypes::local_size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

      for (dataTypes::local_size_type index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        arr[index] = s;
    }

    template <typename NumberTypeComplex, typename NumberTypeReal>
    __global__ void
    copyComplexArrToRealArrsCUDAKernel(const dataTypes::local_size_type size,
                                       const NumberTypeComplex *complexArr,
                                       NumberTypeReal *         realArr,
                                       NumberTypeReal *         imagArr)
    {
      const dataTypes::local_size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

      for (dataTypes::local_size_type index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        {
          realArr[index] = complexArr[index].x;
          imagArr[index] = complexArr[index].y;
        }
    }

    template <typename NumberTypeComplex, typename NumberTypeReal>
    __global__ void
    copyRealArrsToComplexArrCUDAKernel(const dataTypes::local_size_type size,
                                       const NumberTypeReal *           realArr,
                                       const NumberTypeReal *           imagArr,
                                       NumberTypeComplex *complexArr)
    {
      const dataTypes::local_size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

      for (dataTypes::local_size_type index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        {
          complexArr[index].x = realArr[index];
          complexArr[index].y = imagArr[index];
        }
    }
  } // namespace

  namespace cudaUtils
  {
    void
    setupGPU()
    {
      int n_devices = 0;
      cudaGetDeviceCount(&n_devices);
      // std::cout<< "Number of Devices "<<n_devices<<std::endl;
      int device_id =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % n_devices;
      // std::cout<<"Device Id: "<<device_id<<" Task Id
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      cudaSetDevice(device_id);
      // int device = 0;
      // cudaGetDevice(&device);
      // std::cout<< "Device Id currently used is "<<device<< " for taskId:
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      cudaDeviceReset();
    }



    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsGPU(const dataTypes::local_size_type size,
                                const NumberTypeComplex *        complexArr,
                                NumberTypeReal *                 realArr,
                                NumberTypeReal *                 imagArr)
    {
      copyComplexArrToRealArrsCUDAKernel<NumberTypeComplex, NumberTypeReal>
        <<<size / cudaConstants::blockSize + 1, cudaConstants::blockSize>>>(
          size, complexArr, realArr, imagArr);
    }


    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrGPU(const dataTypes::local_size_type size,
                                const NumberTypeReal *           realArr,
                                const NumberTypeReal *           imagArr,
                                NumberTypeComplex *              complexArr)
    {
      copyRealArrsToComplexArrCUDAKernel<NumberTypeComplex, NumberTypeReal>
        <<<size / cudaConstants::blockSize + 1, cudaConstants::blockSize>>>(
          size, realArr, imagArr, complexArr);
    }

    template <typename NumberType>
    void
    copyCUDAVecToCUDAVec(const NumberType *               cudaVecSrc,
                         NumberType *                     cudaVecDst,
                         const dataTypes::local_size_type size)
    {
      CUDACHECK(cudaMemcpy(cudaVecDst,
                           cudaVecSrc,
                           size * sizeof(NumberType),
                           cudaMemcpyDeviceToDevice));
    }

    template <typename NumberType>
    void
    copyHostVecToCUDAVec(const NumberType *               hostVec,
                         NumberType *                     cudaVector,
                         const dataTypes::local_size_type size)
    {
      CUDACHECK(cudaMemcpy(cudaVector,
                           hostVec,
                           size * sizeof(NumberType),
                           cudaMemcpyHostToDevice));
    }

    template <typename NumberType>
    void
    copyCUDAVecToHostVec(const NumberType *               cudaVector,
                         NumberType *                     hostVec,
                         const dataTypes::local_size_type size)
    {
      CUDACHECK(cudaMemcpy(hostVec,
                           cudaVector,
                           size * sizeof(NumberType),
                           cudaMemcpyDeviceToHost));
    }


    template <typename NumberType, typename MemorySpace>
    Vector<NumberType, MemorySpace>::Vector()
      : d_data(NULL)
      , d_size(0)
    {}

    template <typename NumberType, typename MemorySpace>
    Vector<NumberType, MemorySpace>::Vector(
      const dataTypes::local_size_type size,
      const NumberType                 s)
    {
      d_data = NULL;
      d_size = 0;
      this->resize(size, s);
    }

    template <typename NumberType, typename MemorySpace>
    Vector<NumberType, MemorySpace>::~Vector()
    {
      this->clear();
    }


    template <typename NumberType, typename MemorySpace>
    void
    Vector<NumberType, MemorySpace>::resize(
      const dataTypes::local_size_type size)
    {
      this->clear();
      if (size > 0)
        {
          if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
            {
              d_data = (NumberType *)malloc(size * sizeof(NumberType));
              std::memset(d_data, 0, size * sizeof(NumberType));
            }
          else
            {
              CUDACHECK(
                cudaMalloc((void **)&d_data, size * sizeof(NumberType)));
              CUDACHECK(cudaMemset(d_data, 0, size * sizeof(NumberType)));
            }
          d_size = size;
        }
    }

    template <typename NumberType, typename MemorySpace>
    void
    Vector<NumberType, MemorySpace>::resize(
      const dataTypes::local_size_type size,
      const NumberType                 s)
    {
      this->resize(size);
      this->set(s);
    }

    template <typename NumberType, typename MemorySpace>
    void
    Vector<NumberType, MemorySpace>::set(const NumberType s)
    {
      if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
        std::fill(d_data, d_data + d_size, s);
      else
        setKernel<NumberType>
          <<<d_size / cudaConstants::blockSize + 1, cudaConstants::blockSize>>>(
            d_size, s, d_data);
    }

    template <typename NumberType, typename MemorySpace>
    NumberType *
    Vector<NumberType, MemorySpace>::begin()
    {
      return d_data;
    }


    template <typename NumberType, typename MemorySpace>
    const NumberType *
    Vector<NumberType, MemorySpace>::begin() const
    {
      return d_data;
    }


    template <typename NumberType, typename MemorySpace>
    dataTypes::local_size_type
    Vector<NumberType, MemorySpace>::size() const
    {
      return d_size;
    }

    template <typename NumberType, typename MemorySpace>
    void
    Vector<NumberType, MemorySpace>::clear()
    {
      if (d_data != NULL)
        {
          if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
            free(d_data);
          else
            CUDACHECK(cudaFree(d_data));
          d_size = 0;
        }
    }

    template class Vector<double, dftfe::MemorySpace::Host>;
    template class Vector<cuDoubleComplex, dftfe::MemorySpace::Host>;
    template class Vector<double, dftfe::MemorySpace::GPU>;
    template class Vector<cuDoubleComplex, dftfe::MemorySpace::GPU>;

    template void
    copyComplexArrToRealArrsGPU(const dataTypes::local_size_type size,
                                const cuDoubleComplex *          complexArr,
                                double *                         realArr,
                                double *                         imagArr);

    template void
    copyComplexArrToRealArrsGPU(const dataTypes::local_size_type size,
                                const cuFloatComplex *           complexArr,
                                float *                          realArr,
                                float *                          imagArr);

    template void
    copyRealArrsToComplexArrGPU(const dataTypes::local_size_type size,
                                const double *                   realArr,
                                const double *                   imagArr,
                                cuDoubleComplex *                complexArr);

    template void
    copyRealArrsToComplexArrGPU(const dataTypes::local_size_type size,
                                const float *                    realArr,
                                const float *                    imagArr,
                                cuFloatComplex *                 complexArr);

    template void
    copyCUDAVecToCUDAVec(const double *                   cudaVecSrc,
                         double *                         cudaVecDst,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToCUDAVec(const float *                    cudaVecSrc,
                         float *                          cudaVecDst,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToCUDAVec(const cuDoubleComplex *          cudaVecSrc,
                         cuDoubleComplex *                cudaVecDst,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToCUDAVec(const cuFloatComplex *           cudaVecSrc,
                         cuFloatComplex *                 cudaVecDst,
                         const dataTypes::local_size_type size);


    template void
    copyCUDAVecToCUDAVec(const double *                   hostVec,
                         double *                         cudaVector,
                         const dataTypes::local_size_type size);

    template void
    copyHostVecToCUDAVec(const float *                    hostVec,
                         float *                          cudaVector,
                         const dataTypes::local_size_type size);

    template void
    copyHostVecToCUDAVec(const cuDoubleComplex *          hostVec,
                         cuDoubleComplex *                cudaVector,
                         const dataTypes::local_size_type size);

    template void
    copyHostVecToCUDAVec(const cuFloatComplex *           hostVec,
                         cuFloatComplex *                 cudaVector,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToHostVec(const double *                   cudaVector,
                         double *                         hostVec,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToHostVec(const float *                    cudaVector,
                         float *                          hostVec,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToHostVec(const cuDoubleComplex *          cudaVector,
                         cuDoubleComplex *                hostVec,
                         const dataTypes::local_size_type size);

    template void
    copyCUDAVecToHostVec(const cuFloatComplex *           cudaVector,
                         cuFloatComplex *                 hostVec,
                         const dataTypes::local_size_type size);


  } // namespace cudaUtils
} // namespace dftfe
