// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
#include <dftUtils.h>
#include <headers.h>

namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    __global__ void
    saddKernel(ValueType *            y,
               ValueType *            x,
               const ValueType        beta,
               const dftfe::size_type size)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }


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
    interpolateNodalDataToQuadDeviceKernel(
      const dftfe::size_type numDofsPerElem,
      const dftfe::size_type numQuadPoints,
      const dftfe::size_type numVecs,
      const ValueType2 *     parentShapeFunc,
      const ValueType1 *     mapPointToCellIndex,
      const ValueType1 *     mapPointToProcLocal,
      const ValueType1 *     mapPointToShapeFuncIndex,
      const ValueType2 *     parentNodalValues,
      ValueType2 *           quadValues)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries = numQuadPoints * numVecs;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type pointIndex = index / numVecs;
          dftfe::size_type iCellIndex = mapPointToCellIndex[pointIndex];
          dftfe::size_type iShapeFuncIndex =
            mapPointToShapeFuncIndex[pointIndex];
          dftfe::size_type iProcLocalIndex = mapPointToProcLocal[pointIndex];

          dftfe::size_type iVec = index - pointIndex * numVecs;



          for (dftfe::size_type iParentNode = 0; iParentNode < numDofsPerElem;
               iParentNode++)
            {
              dftfe::utils::copyValue(
                quadValues + iProcLocalIndex * numVecs + iVec,
                dftfe::utils::add(
                  quadValues[iProcLocalIndex * numVecs + iVec],
                  dftfe::utils::mult(
                    parentShapeFunc[iShapeFuncIndex + iParentNode],
                    parentNodalValues[iCellIndex * numVecs * numDofsPerElem +
                                      iParentNode + iVec * numDofsPerElem])));
            }
        }
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
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        copyComplexArrToRealArrsDeviceKernel<<<
          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size,
          dftfe::utils::makeDataTypeDeviceCompatible(complexArr),
          realArr,
          imagArr);
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(copyComplexArrToRealArrsDeviceKernel,
                           size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                           0,
                           0,
                           size,
                           dftfe::utils::makeDataTypeDeviceCompatible(
                             complexArr),
                           realArr,
                           imagArr);
#endif
      }



      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const ValueTypeReal *  realArr,
                                     const ValueTypeReal *  imagArr,
                                     ValueTypeComplex *     complexArr)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        copyRealArrsToComplexArrDeviceKernel<<<
          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          size,
          realArr,
          imagArr,
          dftfe::utils::makeDataTypeDeviceCompatible(complexArr));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(copyRealArrsToComplexArrDeviceKernel,
                           size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                           0,
                           0,
                           size,
                           realArr,
                           imagArr,
                           dftfe::utils::makeDataTypeDeviceCompatible(
                             complexArr));
#endif
      }



      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size)
      {
        const dftfe::size_type gridSize =
          (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
          (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        saddKernel<<<gridSize, dftfe::utils::DEVICE_BLOCK_SIZE>>>(y,
                                                                  x,
                                                                  beta,
                                                                  size);
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(saddKernel,
                           gridSize,
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                           0,
                           0,
                           y,
                           x,
                           beta,
                           size);
#endif
      }



      template <typename ValueType1, typename ValueType2>
      void
      interpolateNodalDataToQuadDevice(
        const dftfe::size_type numDofsPerElem,
        const dftfe::size_type numQuadPoints,
        const dftfe::size_type numVecs,
        const ValueType2 *     parentShapeFunc,
        const ValueType1 *     mapPointToCellIndex,
        const ValueType1 *     mapPointToProcLocal,
        const ValueType1 *     mapPointToShapeFuncIndex,
        const ValueType2 *     parentNodalValues,
        ValueType2 *           quadValues)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        interpolateNodalDataToQuadDeviceKernel<<<
          (numQuadPoints * numVecs) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          numDofsPerElem,
          numQuadPoints,
          numVecs,
          dftfe::utils::makeDataTypeDeviceCompatible(parentShapeFunc),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToCellIndex),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToProcLocal),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToShapeFuncIndex),
          dftfe::utils::makeDataTypeDeviceCompatible(parentNodalValues),
          dftfe::utils::makeDataTypeDeviceCompatible(quadValues));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          interpolateNodalDataToQuadDeviceKernel,
          (numQuadPoints * numVecs) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numDofsPerElem,
          numQuadPoints,
          numVecs,
          dftfe::utils::makeDataTypeDeviceCompatible(parentShapeFunc),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToCellIndex),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToProcLocal),
          dftfe::utils::makeDataTypeDeviceCompatible(mapPointToShapeFuncIndex),
          dftfe::utils::makeDataTypeDeviceCompatible(parentNodalValues),
          dftfe::utils::makeDataTypeDeviceCompatible(quadValues));
#endif
      }


      template void
      interpolateNodalDataToQuadDevice(
        const dftfe::size_type numDofsPerElem,
        const dftfe::size_type numQuadPoints,
        const dftfe::size_type numVecs,
        const double *         parentShapeFunc,
        const size_type *      mapPointToCellIndex,
        const size_type *      mapPointToProcLocal,
        const size_type *      mapPointToShapeFuncIndex,
        const double *         parentNodalValues,
        double *               quadValues);


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
      sadd(double *               y,
           double *               x,
           const double           beta,
           const dftfe::size_type size);
    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe
