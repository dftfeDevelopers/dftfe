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

namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    __global__ void
    saddKernel(ValueType        *y,
               ValueType        *x,
               const ValueType   beta,
               const dftfe::uInt size)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }


    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyComplexArrToRealArrsDeviceKernel(const dftfe::uInt       size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal          *realArr,
                                         ValueTypeReal          *imagArr)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          realArr[idx] = complexArr[idx].x;
          imagArr[idx] = complexArr[idx].y;
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyRealArrsToComplexArrDeviceKernel(const dftfe::uInt    size,
                                         const ValueTypeReal *realArr,
                                         const ValueTypeReal *imagArr,
                                         ValueTypeComplex    *complexArr)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          complexArr[idx].x = realArr[idx];
          complexArr[idx].y = imagArr[idx];
        }
    }



    template <typename ValueType1, typename ValueType2>
    __global__ void
    interpolateNodalDataToQuadDeviceKernel(
      const dftfe::uInt numDofsPerElem,
      const dftfe::uInt numQuadPoints,
      const dftfe::uInt numVecs,
      const ValueType2 *parentShapeFunc,
      const ValueType1 *mapPointToCellIndex,
      const ValueType1 *mapPointToProcLocal,
      const ValueType1 *mapPointToShapeFuncIndex,
      const ValueType2 *parentNodalValues,
      ValueType2       *quadValues)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = numQuadPoints * numVecs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt pointIndex      = index / numVecs;
          dftfe::uInt iCellIndex      = mapPointToCellIndex[pointIndex];
          dftfe::uInt iShapeFuncIndex = mapPointToShapeFuncIndex[pointIndex];
          dftfe::uInt iProcLocalIndex = mapPointToProcLocal[pointIndex];

          dftfe::uInt iVec = index - pointIndex * numVecs;



          for (dftfe::uInt iParentNode = 0; iParentNode < numDofsPerElem;
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
      setupDevice(const int &mpi_rank)
      {
        int n_devices = 0;
        dftfe::utils::getDeviceCount(&n_devices);
        // std::cout<< "Number of Devices "<<n_devices<<std::endl;
        int device_id = mpi_rank % n_devices;
        // std::cout<<"Device Id: "<<device_id<<" Task Id
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::setDevice(device_id);
        // dftfe::Int device = 0;
        // dftfe::utils::getDevice(&device);
        // std::cout<< "Device Id currently used is "<<device<< " for taskId:
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::deviceReset();
      }


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrsDevice(const dftfe::uInt       size,
                                     const ValueTypeComplex *complexArr,
                                     ValueTypeReal          *realArr,
                                     ValueTypeReal          *imagArr)
      {
        DFTFE_LAUNCH_KERNEL(copyComplexArrToRealArrsDeviceKernel,
                            size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                            0,
                            0,
                            size,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              complexArr),
                            realArr,
                            imagArr);
      }



      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::uInt    size,
                                     const ValueTypeReal *realArr,
                                     const ValueTypeReal *imagArr,
                                     ValueTypeComplex    *complexArr)
      {
        DFTFE_LAUNCH_KERNEL(copyRealArrsToComplexArrDeviceKernel,
                            size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                            0,
                            0,
                            size,
                            realArr,
                            imagArr,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              complexArr));
      }



      template <typename ValueType>
      void
      sadd(ValueType        *y,
           ValueType        *x,
           const ValueType   beta,
           const dftfe::uInt size)
      {
        const dftfe::uInt gridSize =
          (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
          (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
        DFTFE_LAUNCH_KERNEL(saddKernel,
                            gridSize,
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                            0,
                            0,
                            y,
                            x,
                            beta,
                            size);
      }



      template <typename ValueType1, typename ValueType2>
      void
      interpolateNodalDataToQuadDevice(
        const dftfe::uInt numDofsPerElem,
        const dftfe::uInt numQuadPoints,
        const dftfe::uInt numVecs,
        const ValueType2 *parentShapeFunc,
        const ValueType1 *mapPointToCellIndex,
        const ValueType1 *mapPointToProcLocal,
        const ValueType1 *mapPointToShapeFuncIndex,
        const ValueType2 *parentNodalValues,
        ValueType2       *quadValues)
      {
        DFTFE_LAUNCH_KERNEL(
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
      }


      template void
      interpolateNodalDataToQuadDevice(
        const dftfe::uInt  numDofsPerElem,
        const dftfe::uInt  numQuadPoints,
        const dftfe::uInt  numVecs,
        const double      *parentShapeFunc,
        const dftfe::uInt *mapPointToCellIndex,
        const dftfe::uInt *mapPointToProcLocal,
        const dftfe::uInt *mapPointToShapeFuncIndex,
        const double      *parentNodalValues,
        double            *quadValues);


      template void
      copyComplexArrToRealArrsDevice(const dftfe::uInt           size,
                                     const std::complex<double> *complexArr,
                                     double                     *realArr,
                                     double                     *imagArr);

      template void
      copyComplexArrToRealArrsDevice(const dftfe::uInt          size,
                                     const std::complex<float> *complexArr,
                                     float                     *realArr,
                                     float                     *imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::uInt     size,
                                     const double         *realArr,
                                     const double         *imagArr,
                                     std::complex<double> *complexArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::uInt    size,
                                     const float         *realArr,
                                     const float         *imagArr,
                                     std::complex<float> *complexArr);
      template void
      copyComplexArrToRealArrsDevice(const dftfe::uInt          size,
                                     const std::complex<float> *complexArr,
                                     double                    *realArr,
                                     double                    *imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::uInt    size,
                                     const double        *realArr,
                                     const double        *imagArr,
                                     std::complex<float> *complexArr);

      template void
      sadd(double *y, double *x, const double beta, const dftfe::uInt size);
    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe
