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
// @author Sambit Das, Gourab Panigrahi
//


#include <deviceHelpers.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <dftUtils.h>
#include <headers.h>
#include <cublas_v2.h>

namespace dftfe
{
  namespace
  {
    template <typename NumberTypeComplex, typename NumberTypeReal>
    __global__ void
    copyComplexArrToRealArrsDeviceKernel(const dftfe::size_type size,
                                         const NumberTypeComplex *complexArr,
                                         NumberTypeReal *         realArr,
                                         NumberTypeReal *         imagArr)
    {
      const dftfe::size_type globalId =
        threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          realArr[idx] = complexArr[idx].x;
          imagArr[idx] = complexArr[idx].y;
        }
    }

    template <typename NumberTypeComplex, typename NumberTypeReal>
    __global__ void
    copyRealArrsToComplexArrDeviceKernel(const dftfe::size_type size,
                                         const NumberTypeReal *realArr,
                                         const NumberTypeReal *imagArr,
                                         NumberTypeComplex *   complexArr)
    {
      const dftfe::size_type globalId =
        threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          complexArr[idx].x = realArr[idx];
          complexArr[idx].y = imagArr[idx];
        }
    }
  } // namespace

  namespace deviceUtils
  {
    void
    setupDevice()
    {
      int n_devices = 0;
      dftfe::utils::getDeviceCount(&n_devices);
      // std::cout<< "Number of Devices "<<n_devices<<std::endl;
      int device_id =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % n_devices;
      // std::cout<<"Device Id: "<<device_id<<" Task Id
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      dftfe::utils::setDevice(device_id);
      // int device = 0;
      // dftfe::utils::getDevice(&device);
      // std::cout<< "Device Id currently used is "<<device<< " for taskId:
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      dftfe::utils::deviceReset();
    }



    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsDevice(const dftfe::size_type size,
                                   const NumberTypeComplex *        complexArr,
                                   NumberTypeReal *                 realArr,
                                   NumberTypeReal *                 imagArr)
    {
      copyComplexArrToRealArrsDeviceKernel
        <<<size / dftfe::utils::DEVICE_BLOCK_SIZE + 1, dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size, dftfe::utils::makeDataTypeDeviceCompatible(complexArr), realArr, imagArr);
    }


    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                   const NumberTypeReal *           realArr,
                                   const NumberTypeReal *           imagArr,
                                   NumberTypeComplex *              complexArr)
    {
      copyRealArrsToComplexArrDeviceKernel
        <<<size / dftfe::utils::DEVICE_BLOCK_SIZE + 1, dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size, realArr, imagArr, dftfe::utils::makeDataTypeDeviceCompatible(complexArr));
    }

    void
    add(double *        y,
        const double *  x,
        const double    alpha,
        const int       size,
        cublasHandle_t &cublasHandle)
    {
      int incx = 1, incy = 1;
      cublasCheck(cublasDaxpy(cublasHandle, size, &alpha, x, incx, y, incy));
    }

    double
    l2_norm(const double *  x,
            const int       size,
            const MPI_Comm &mpi_communicator,
            cublasHandle_t &cublasHandle)
    {
      int    incx = 1;
      double local_nrm, nrm = 0;

      cublasCheck(cublasDnrm2(cublasHandle, size, x, incx, &local_nrm));

      local_nrm *= local_nrm;
      MPI_Allreduce(&local_nrm, &nrm, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      return std::sqrt(nrm);
    }

    double
    dot(const double *  x,
        const double *  y,
        const int       size,
        const MPI_Comm &mpi_communicator,
        cublasHandle_t &cublasHandle)
    {
      int    incx = 1, incy = 1;
      double local_sum, sum = 0;

      cublasCheck(cublasDdot(cublasHandle, size, x, incx, y, incy, &local_sum));
      MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      return sum;
    }

    template <typename NumberType>
    __global__ void
    saddKernel(NumberType *     y,
               NumberType *     x,
               const NumberType beta,
               const int        size)
    {
      const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (int idx = globalId; idx < size; idx += blockDim.x * gridDim.x)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }

    template <typename NumberType>
    void
    sadd(NumberType *y, NumberType *x, const NumberType beta, const int size)
    {
      const int gridSize = (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
                           (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
      saddKernel<NumberType>
        <<<gridSize, dftfe::utils::DEVICE_BLOCK_SIZE>>>(y, x, beta, size);
    }


    template void
    copyComplexArrToRealArrsDevice(const dftfe::size_type size,
                                   const  std::complex<double> *          complexArr,
                                   double *                         realArr,
                                   double *                         imagArr);

    template void
    copyComplexArrToRealArrsDevice(const dftfe::size_type size,
                                   const  std::complex<float> *           complexArr,
                                   float *                          realArr,
                                   float *                          imagArr);

    template void
    copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                   const double *                   realArr,
                                   const double *                   imagArr,
                                   std::complex<double> *                complexArr);

    template void
    copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                   const float *                    realArr,
                                   const float *                    imagArr,
                                   std::complex<float> *                 complexArr);

    template void
    sadd(double *y, double *x, const double beta, const int size);

  } // namespace deviceUtils
} // namespace dftfe
