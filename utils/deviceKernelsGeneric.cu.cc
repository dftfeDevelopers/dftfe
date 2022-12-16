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


#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceBlasWrapper.h>
#include <dftUtils.h>
#include <headers.h>

namespace dftfe
{
  namespace
  {
    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyComplexArrToRealArrsDeviceKernel(const dftfe::size_type  size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal *         realArr,
                                         ValueTypeReal *         imagArr)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          realArr[idx] = complexArr[idx].x;
          imagArr[idx] = complexArr[idx].y;
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyRealArrsToComplexArrDeviceKernel(const dftfe::size_type size,
                                         const ValueTypeReal *  realArr,
                                         const ValueTypeReal *  imagArr,
                                         ValueTypeComplex *     complexArr)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          complexArr[idx].x = realArr[idx];
          complexArr[idx].y = imagArr[idx];
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    copyValueType1ArrToValueType2ArrDeviceKernel(
      const dftfe::size_type size,
      const ValueType1 *     valueType1Arr,
      ValueType2 *           valueType2Arr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
    }


    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyFromBlockDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + copyFromVecStartingContiguousBlockIds[blockIndex] +
              intraBlockIndex,
            copyFromVec[index]);
        }
    }


    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockConstantStrideDeviceKernel(
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries = numBlocks * blockSizeTo;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex      = index / blockSizeTo;
            unsigned int intraBlockIndex = index - blockIndex * blockSizeTo;
            dftfe::utils::copyValue(copyToVec + index,
                                    copyFromVec[blockIndex * blockSizeFrom +
                                                startingId + intraBlockIndex]);
          }
      }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyFromBlockConstantStrideDeviceKernel(
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries = numBlocks * blockSizeFrom;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex      = index / blockSizeFrom;
            unsigned int intraBlockIndex = index - blockIndex * blockSizeFrom;
            dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                      startingId + intraBlockIndex,
                                    copyFromVec[index]);
          }
      }
    }

    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    ascalDeviceKernel(const dftfe::size_type n,
                      ValueType1 *           x,
                      const ValueType2       a)
    {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex] strided block wise
    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedBlockScaleDeviceKernel(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1              a,
        const ValueType1 *             s,
        ValueType2 *      x)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          dftfe::utils::copyValue(
            x + index,dftfe::utils::mult(dftfe::utils::mult(a,s[blockIndex]),x[index]));
        }
    }

    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    axpbyDeviceKernel(const dftfe::size_type n,
                      const ValueType1 *     x,
                      ValueType1 *           y,
                      const ValueType2       a,
                      const ValueType2       b)
    {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(y + i,
                                dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                  dftfe::utils::mult(b, y[i])));
    }

  } // namespace

  namespace utils
  {
    namespace deviceKernelsGeneric
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



      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrsDevice(const dftfe::size_type  size,
                                     const ValueTypeComplex *complexArr,
                                     ValueTypeReal *         realArr,
                                     ValueTypeReal *         imagArr)
      {
        copyComplexArrToRealArrsDeviceKernel<<<
          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size,
          dftfe::utils::makeDataTypeDeviceCompatible(complexArr),
          realArr,
          imagArr);
      }



      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const ValueTypeReal *  realArr,
                                     const ValueTypeReal *  imagArr,
                                     ValueTypeComplex *     complexArr)
      {
        copyRealArrsToComplexArrDeviceKernel<<<
          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size,
          realArr,
          imagArr,
          dftfe::utils::makeDataTypeDeviceCompatible(complexArr));
      }

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr)
      {
        copyValueType1ArrToValueType2ArrDeviceKernel<<<
          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size,
          dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
          dftfe::utils::makeDataTypeDeviceCompatible(valueType2Arr));
      }

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVec,
        ValueType2 *                   copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
      {
        stridedCopyToBlockDeviceKernel<<<(contiguousBlockSize *
                                          numContiguousBlocks) /
                                             dftfe::utils::DEVICE_BLOCK_SIZE +
                                           1,
                                         dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          contiguousBlockSize,
          numContiguousBlocks,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
          copyFromVecStartingContiguousBlockIds);
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVecBlock,
        ValueType2 *                   copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
      {
        stridedCopyFromBlockDeviceKernel<<<(contiguousBlockSize *
                                            numContiguousBlocks) /
                                               dftfe::utils::DEVICE_BLOCK_SIZE +
                                             1,
                                           dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          contiguousBlockSize,
          numContiguousBlocks,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVecBlock),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec),
          copyFromVecStartingContiguousBlockIds);
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec)
      {
        stridedCopyToBlockConstantStrideDeviceKernel<<<
          (blockSizeTo * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          blockSizeTo,
          blockSizeFrom,
          numBlocks,
          startingId,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
      {
        stridedCopyFromBlockConstantStrideDeviceKernel<<<
          (blockSizeFrom * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          blockSizeTo,
          blockSizeFrom,
          numBlocks,
          startingId,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }

      template <typename ValueType1, typename ValueType2>
      void
      axpby(const dftfe::size_type n,
            const ValueType1 *     x,
            ValueType1 *           y,
            const ValueType2       a,
            const ValueType2       b)
      {
        axpbyDeviceKernel<<<
          n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          n,
          dftfe::utils::makeDataTypeDeviceCompatible(x),
          dftfe::utils::makeDataTypeDeviceCompatible(y),
          dftfe::utils::makeDataTypeDeviceCompatible(a),
          dftfe::utils::makeDataTypeDeviceCompatible(b));
      }

      template <typename ValueType1, typename ValueType2>
      void
      ascal(const dftfe::size_type n, ValueType1 *x, const ValueType2 a)
      {
        ascalDeviceKernel<<<
          n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          n,
          dftfe::utils::makeDataTypeDeviceCompatible(x),
          dftfe::utils::makeDataTypeDeviceCompatible(a));
      }

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1              a,
        const ValueType1 *             s,
        ValueType2 *      x)
      {
        stridedBlockScaleDeviceKernel<<<
          (contiguousBlockSize*numContiguousBlocks)/ dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(contiguousBlockSize,
                                      numContiguousBlocks,
                                      dftfe::utils::makeDataTypeDeviceCompatible(a),
                                      dftfe::utils::makeDataTypeDeviceCompatible(s),
                                      dftfe::utils::makeDataTypeDeviceCompatible(x));
      }

      void
      add(double *                          y,
          const double *                    x,
          const double                      alpha,
          const int                         size,
          dftfe::utils::deviceBlasHandle_t &deviceBlasHandle)
      {
        int incx = 1, incy = 1;
        dftfe::utils::deviceBlasWrapper::axpy(
          deviceBlasHandle, size, &alpha, x, incx, y, incy);
      }

      double
      l2_norm(const double *                    x,
              const int                         size,
              const MPI_Comm &                  mpi_communicator,
              dftfe::utils::deviceBlasHandle_t &deviceBlasHandle)
      {
        int    incx = 1;
        double local_nrm, nrm = 0;

        dftfe::utils::deviceBlasWrapper::nrm2(
          deviceBlasHandle, size, x, incx, &local_nrm);

        local_nrm *= local_nrm;
        MPI_Allreduce(
          &local_nrm, &nrm, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        return std::sqrt(nrm);
      }

      double
      dot(const double *                    x,
          const double *                    y,
          const int                         size,
          const MPI_Comm &                  mpi_communicator,
          dftfe::utils::deviceBlasHandle_t &deviceBlasHandle)
      {
        int    incx = 1, incy = 1;
        double local_sum, sum = 0;

        dftfe::utils::deviceBlasWrapper::dot(
          deviceBlasHandle, size, x, incx, y, incy, &local_sum);
        MPI_Allreduce(
          &local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        return sum;
      }

      template <typename ValueType>
      __global__ void
      saddKernel(ValueType *     y,
                 ValueType *     x,
                 const ValueType beta,
                 const int       size)
      {
        const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

        for (int idx = globalId; idx < size; idx += blockDim.x * gridDim.x)
          {
            y[idx] = beta * y[idx] - x[idx];
            x[idx] = 0;
          }
      }

      template <typename ValueType>
      void
      sadd(ValueType *y, ValueType *x, const ValueType beta, const int size)
      {
        const int gridSize =
          (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
          (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
        saddKernel<ValueType>
          <<<gridSize, dftfe::utils::DEVICE_BLOCK_SIZE>>>(y, x, beta, size);
      }


      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type      size,
                                     const std::complex<double> *complexArr,
                                     double *                    realArr,
                                     double *                    imagArr);

      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type     size,
                                     const std::complex<float> *complexArr,
                                     float *                    realArr,
                                     float *                    imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const double *         realArr,
                                     const double *         imagArr,
                                     std::complex<double> * complexArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const float *          realArr,
                                     const float *          imagArr,
                                     std::complex<float> *  complexArr);
      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type     size,
                                     const std::complex<float> *complexArr,
                                     double *                   realArr,
                                     double *                   imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const double *         realArr,
                                     const double *         imagArr,
                                     std::complex<float> *  complexArr);

      template void
      sadd(double *y, double *x, const double beta, const int size);

      // for axpby
      template void
      axpby(const dftfe::size_type n,
            const double *         x,
            double *               y,
            const double           a,
            const double           b);

      template void
      axpby(const dftfe::size_type n,
            const float *          x,
            float *                y,
            const float            a,
            const float            b);

      template void
      axpby(const dftfe::size_type      n,
            const std::complex<double> *x,
            std::complex<double> *      y,
            const std::complex<double>  a,
            const std::complex<double>  b);

      template void
      axpby(const dftfe::size_type     n,
            const std::complex<float> *x,
            std::complex<float> *      y,
            const std::complex<float>  a,
            const std::complex<float>  b);


      template void
      axpby(const dftfe::size_type      n,
            const std::complex<double> *x,
            std::complex<double> *      y,
            const double                a,
            const double                b);

      template void
      axpby(const dftfe::size_type     n,
            const std::complex<float> *x,
            std::complex<float> *      y,
            const double               a,
            const double               b);


      // for ascal
      template void
      ascal(const dftfe::size_type n, double *x, const double a);

      template void
      ascal(const dftfe::size_type n, float *x, const float a);

      template void
      ascal(const dftfe::size_type     n,
            std::complex<double> *     x,
            const std::complex<double> a);

      template void
      ascal(const dftfe::size_type    n,
            std::complex<float> *     x,
            const std::complex<float> a);

      template void
      ascal(const dftfe::size_type n, std::complex<double> *x, double a);

      template void
      ascal(const dftfe::size_type n, std::complex<float> *x, double a);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       double *               valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       float *                valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(
        const dftfe::size_type      size,
        const std::complex<double> *valueType1Arr,
        std::complex<double> *      valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type     size,
                                       const std::complex<float> *valueType1Arr,
                                       std::complex<float> *valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       float *                valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       double *               valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(
        const dftfe::size_type      size,
        const std::complex<double> *valueType1Arr,
        std::complex<float> *       valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type     size,
                                       const std::complex<float> *valueType1Arr,
                                       std::complex<double> *valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<float> *  valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       std::complex<float> *  valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<double> * valueType2Arr);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       std::complex<double> * valueType2Arr);


      // strided copy to block
      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVec,
        double *                       copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVec,
        float *                        copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVec,
        std::complex<double> *         copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVec,
        std::complex<float> *          copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVec,
        float *                        copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVec,
        double *                       copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVec,
        std::complex<float> *          copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVec,
        std::complex<double> *         copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      // strided copy from block
      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVecBlock,
        double *                       copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVecBlock,
        float *                        copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVecBlock,
        std::complex<double> *         copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVecBlock,
        std::complex<float> *          copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVecBlock,
        float *                        copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVecBlock,
        double *                       copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVecBlock,
        std::complex<float> *          copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVecBlock,
        std::complex<double> *         copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      // strided copy to block constant stride
      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       double *               copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       float *                copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double> *      copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<float> *      copyToVec);


      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       float *                copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       double *               copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<float> *       copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<double> *     copyToVec);

      // strided copy from block constant stride
      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const double *         copyFromVec,
                                         double *               copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const float *          copyFromVec,
                                         float *                copyToVec);

      template void
      stridedCopyFromBlockConstantStride(
        const dftfe::size_type      blockSizeTo,
        const dftfe::size_type      blockSizeFrom,
        const dftfe::size_type      numBlocks,
        const dftfe::size_type      startingId,
        const std::complex<double> *copyFromVec,
        std::complex<double> *      copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const std::complex<float> *copyFromVec,
                                         std::complex<float> *      copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const double *         copyFromVec,
                                         float *                copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const float *          copyFromVec,
                                         double *               copyToVec);

      template void
      stridedCopyFromBlockConstantStride(
        const dftfe::size_type      blockSizeTo,
        const dftfe::size_type      blockSizeFrom,
        const dftfe::size_type      numBlocks,
        const dftfe::size_type      startingId,
        const std::complex<double> *copyFromVec,
        std::complex<float> *       copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const std::complex<float> *copyFromVec,
                                         std::complex<double> *     copyToVec);

      //stridedBlockScale
      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double              a,
        const double *             s,
        double *      x);

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float              a,
        const float *             s,
        float *      x);

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double>              a,
        const std::complex<double> *             s,
        std::complex<double> *      x);

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float>              a,
        const std::complex<float> *             s,
        std::complex<float> *      x);  

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double              a,
        const double *             s,
        float *      x); 

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float              a,
        const float *             s,
        double *      x); 

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double>              a,
        const std::complex<double> *             s,
        std::complex<float> *      x); 

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float>              a,
        const std::complex<float> *             s,
        std::complex<double> *      x);  

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double              a,
        const double *             s,
        std::complex<double> *      x); 

      template 
      void
      stridedBlockScale(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double              a,
        const double *             s,
        std::complex<float> *      x);         

    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe
