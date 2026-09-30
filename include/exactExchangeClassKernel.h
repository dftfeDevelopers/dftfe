// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXACTEXCHANGECLASSKERNEL_H
#define DFTFE_EXACTEXCHANGECLASSKERNEL_H

#include "exactExchangeClass.h"

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
                                 const unsigned int strideOperatorLength);


                template <typename ValueType>
    void setBoundaryValues(const dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::HOST> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::HOST> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson);

		template <typename ValueType>
    void computeFockExchange( dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength);
		#if defined(DFTFE_WITH_DEVICE)
		template <typename ValueType>
    void computeHadamardProductKernel( const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLeveQuadValuesInput,
                                 const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesOperator,
                                 dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValuesInputOperator,
                                 const unsigned int numTotalQuadPoints,
                                 const unsigned int strideInputLength,
                                 const unsigned int strideOperatorLength);


		template <typename ValueType>
    void setBoundaryValues(const dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::DEVICE> &distVecMemSpace,
                      const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> &total_charge_memSpace,
                      dftfe::linearAlgebra::MultiVector<ValueType, dftfe::utils::MemorySpace::DEVICE> & boundaryValuesVec,
                      const unsigned int locallyOwnedDofs,
                      const unsigned int blockSizePoisson);
    
	   template <typename ValueType>
    void computeFockExchange( dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & VFockQuadInputValues,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & orbitalOccupancyOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesOperator,
                        const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE> & cellLevelQuadValuesInputOperator,
                        const unsigned int numTotalQuadPoints,
                        const unsigned int strideOperatorLength,
                        const unsigned int strideInputLength);

	   #endif
	}
}

#endif 
