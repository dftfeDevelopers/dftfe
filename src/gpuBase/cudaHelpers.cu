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

namespace dftfe
{
  namespace
  {
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
  } // namespace cudaUtils
} // namespace dftfe
