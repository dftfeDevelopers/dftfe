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
// @author Vishal Subramanian
//


#include "exactExchangeClassKernel.h"
#include "DeviceKernelLauncherHelpers.h"
namespace dftfe
{

  namespace exchangeInternal
  {
	    template <typename ValueType>
    void computeHadamardProductKernel( const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> &cellLeveQuadValuesInput,
                           const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> &cellLevelQuadValuesOperator,
                                 dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> &cellLevelQuadValuesInputOperator,
                           const unsigned int numTotalQuadPoints,
                           const unsigned int strideInputLength,
                           const unsigned int strideOperatorLength)
    {
      for(unsigned int iQuad = 0; iQuad < numTotalQuadPoints; iQuad++)
        {
          for( unsigned int iInput = 0; iInput < strideInputLength; iInput++)
            {
              for( unsigned int iOperator = 0; iOperator < strideOperatorLength; iOperator++)
                {
                  cellLevelQuadValuesInputOperator.data()[iQuad*strideInputLength*strideOperatorLength + iInput*strideOperatorLength + iOperator] =
                    cellLeveQuadValuesInput.data()[iQuad*strideInputLength + iInput] *
                    cellLevelQuadValuesOperator.data()[iQuad*strideOperatorLength + iOperator];
                }
            }
        }
    }


	        template <typename ValueType>
    void setBoundaryValues(const dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::HOST> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::HOST> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson)
    {
      for(unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++)
        {
          for(unsigned int iVec = 0 ; iVec < blockSizePoisson; iVec++)
            {
              boundaryValuesVec.data()[iNode*blockSizePoisson + iVec] =
                total_charge_memSpace.data()[iVec] *
                distVecMemSpace.data()[iNode];
            }
        }
    }


		   template <typename ValueType>
    void computeFockExchange( dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength)
    {
      for(unsigned int iQuad = 0; iQuad < numTotalQuadPoints; iQuad++)
        {
          for( unsigned int iInput = 0; iInput < strideInputLength; iInput++)
            {
              for( unsigned int iOperator = 0; iOperator < strideOperatorLength; iOperator++)
                {
                  VFockQuadInputValues.data()[iQuad*strideInputLength + iInput] -=
                    orbitalOccupancyOperator.data()[iOperator] *
                    cellLevelQuadValuesInputOperator.data()[iQuad*strideInputLength*strideOperatorLength + iInput*strideOperatorLength + iOperator] *
                    cellLevelQuadValuesOperator.data()[iQuad*strideOperatorLength + iOperator];
                }
            }
        }
    }

		   template void
			   computeFockExchange( dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength);

		   template void 
			   setBoundaryValues(const dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson);

		   template void
			   computeHadamardProductKernel( const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> &cellLeveQuadValuesInput,
                           const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> &cellLevelQuadValuesOperator,
                                 dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::HOST> &cellLevelQuadValuesInputOperator,
                           const unsigned int numTotalQuadPoints,
                           const unsigned int strideInputLength,
                           const unsigned int strideOperatorLength);

	  #ifdef DFTFE_WITH_DEVICE

    template <typename ValueType>
    __global__ void computeHadamardProductDeviceKernel(const ValueType *cellLeveQuadValuesInput,
                                       const ValueType *cellLevelQuadValuesOperator,
                                       ValueType *cellLevelQuadValuesInputOperator,
                                       const unsigned int numTotalQuadPoints,
                                       const unsigned int strideInputLength,
                                       const unsigned int strideOperatorLength)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numTotalQuadPoints * strideInputLength * strideOperatorLength;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x) {
          unsigned int quadIndex = index /(strideInputLength * strideOperatorLength);
          unsigned int poissonWaveIndex = index - quadIndex*(strideInputLength * strideOperatorLength);
          unsigned int inputWaveIndex = poissonWaveIndex / strideOperatorLength;
          unsigned int operatorWaveIndex = poissonWaveIndex % strideOperatorLength;
          dftfe::utils::copyValue (
            cellLevelQuadValuesInputOperator + index,
            dftfe::utils::mult(
              cellLevelQuadValuesOperator[quadIndex*strideOperatorLength + operatorWaveIndex],
              cellLeveQuadValuesInput[quadIndex*strideInputLength + inputWaveIndex]));
        }
    }

        template <typename ValueType>
    void computeHadamardProductKernel( const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLeveQuadValuesInput,
                                 const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesOperator,
                                 dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesInputOperator,
                                 const unsigned int numTotalQuadPoints,
                                 const unsigned int strideInputLength,
                                 const unsigned int strideOperatorLength)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeHadamardProductDeviceKernel<<<(numTotalQuadPoints * strideInputLength * strideOperatorLength) /
                                dftfe::utils::DEVICE_BLOCK_SIZE +
                              1,
                            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(cellLeveQuadValuesInput.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesInputOperator.begin()),
        numTotalQuadPoints,
        strideInputLength,
        strideOperatorLength);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        computeHadamardProductDeviceKernel,
        (numTotalQuadPoints * strideInputLength * strideOperatorLength) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0,
        dftfe::utils::makeDataTypeDeviceCompatible(cellLeveQuadValuesInput.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesInputOperator.begin()),
        numTotalQuadPoints,
        strideInputLength,
        strideOperatorLength);
#endif
    }


	    template <typename ValueType>
    __global__ void setBoundaryValuesDeviceKernel(const ValueType *distVecMemSpace,
                                       const ValueType *total_charge_memSpace,
                                       ValueType *boundaryValuesVec,
                                       const unsigned int locallyOwnedDofs,
                                       const unsigned int blockSizePoisson)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        locallyOwnedDofs * blockSizePoisson;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x) {
          unsigned int nodeIndex = index /blockSizePoisson;
          unsigned int poissonWaveIndex = index % blockSizePoisson;
          dftfe::utils::copyValue (
            boundaryValuesVec + index,
            dftfe::utils::mult(
              total_charge_memSpace[poissonWaveIndex],
              distVecMemSpace[nodeIndex]));
        }
    }

	       template <typename ValueType>
    void setBoundaryValues(const dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::DEVICE> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::DEVICE> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      setBoundaryValuesDeviceKernel<<<(locallyOwnedDofs * blockSizePoisson) /
                                               dftfe::utils::DEVICE_BLOCK_SIZE +
                                             1,
                                           dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(distVecMemSpace.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(total_charge_memSpace.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(boundaryValuesVec.begin()),
        locallyOwnedDofs,
        blockSizePoisson);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        setBoundaryValuesDeviceKernel,
        (locallyOwnedDofs * blockSizePoisson) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0,
        dftfe::utils::makeDataTypeDeviceCompatible(distVecMemSpace.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(total_charge_memSpace.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(boundaryValuesVec.begin()),
        locallyOwnedDofs,
        blockSizePoisson);
#endif
    }


	           template <typename ValueType>
    __global__ void computeFockExchangeDeviceKernel(ValueType * VFockQuadInputValues,
                                    const ValueType *orbitalOccupancyOperator,
                                    const ValueType * cellLevelQuadValuesOperator,
                                    const ValueType * cellLevelQuadValuesInputOperator,
                                    const unsigned int numTotalQuadPoints,
                                    const unsigned int strideOperatorLength,
                                    const unsigned int strideInputLength)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numTotalQuadPoints * strideInputLength ;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int quadIndex = index /(strideInputLength);
          unsigned int inputWaveIndex = index % strideInputLength;
          for (unsigned int operatorWaveIndex = 0; operatorWaveIndex < strideOperatorLength; operatorWaveIndex++)
            {
              dftfe::utils::copyValue(
                VFockQuadInputValues + quadIndex*strideInputLength + inputWaveIndex,
                          dftfe::utils::add(
                  VFockQuadInputValues[index],
                  dftfe::utils::mult(-1.0,
                                     dftfe::utils::mult(
                                       orbitalOccupancyOperator[operatorWaveIndex],
                                       dftfe::utils::mult(cellLevelQuadValuesOperator[quadIndex*strideOperatorLength + operatorWaveIndex],
                                                          cellLevelQuadValuesInputOperator[quadIndex*strideOperatorLength*strideInputLength + inputWaveIndex*strideOperatorLength +  operatorWaveIndex]))
                                     )));
            }
        }
    }


		       template <typename ValueType>
    void computeFockExchange( dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeFockExchangeDeviceKernel<<<(numTotalQuadPoints * strideInputLength) /
                                          dftfe::utils::DEVICE_BLOCK_SIZE +
                                        1,
                                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(VFockQuadInputValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancyOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesInputOperator.begin()),
        numTotalQuadPoints,
        strideOperatorLength,
        strideInputLength);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        computeFockExchangeDeviceKernel,
        (numTotalQuadPoints * strideInputLength) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0,
        dftfe::utils::makeDataTypeDeviceCompatible(VFockQuadInputValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancyOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesOperator.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValuesInputOperator.begin()),
        numTotalQuadPoints,
        strideOperatorLength,
        strideInputLength);
#endif
    }

		       template void
                           computeFockExchange( dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength);

		                         template void
                           setBoundaryValues(const dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::DEVICE> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::DEVICE> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson);

                   template void
                           computeHadamardProductKernel( const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> &cellLeveQuadValuesInput,
                           const dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesOperator,
                                 dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesInputOperator,
                           const unsigned int numTotalQuadPoints,
                           const unsigned int strideInputLength,
                           const unsigned int strideOperatorLength);

		       #endif

  }
}
