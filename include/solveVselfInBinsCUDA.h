// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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

#if defined(DFTFE_WITH_GPU)
#ifndef solveVselfInBinsCUDA_H_
#define solveVselfInBinsCUDA_H_

#include <headers.h>
#include <operatorCUDA.h>
#include <constraintMatrixInfoCUDA.h>

namespace dftfe
{
	namespace poissonCUDA
	{
		void solveVselfInBins
			(operatorDFTCUDAClass & operatorMatrix,
			 const dealii::MatrixFree<3,double> & matrixFreeData,
       const unsigned int mfDofHandlerIndex,
			 const dealii::AffineConstraints<double> & hangingPeriodicConstraintMatrix,
			 const double * rhsFlattenedH,
			 const double * diagonalAH,
			 const double * inhomoIdsColoredVecFlattenedH,
			 const unsigned int localSize,
			 const unsigned int ghostSize,
			 const unsigned int numberBins,
			 const MPI_Comm & mpiComm,  
			 double * xH);

		void cgSolver(cublasHandle_t &handle,
				dftUtils::constraintMatrixInfoCUDA & constraintsMatrixDataInfoCUDA,
				const double * bD,
				const double * diagonalAD,
				const thrust::device_vector<double> & poissonCellStiffnessMatricesD,
				const thrust::device_vector<double> & inhomoIdsColoredVecFlattenedD,
				const thrust::device_vector<dealii::types::global_dof_index> & cellLocalProcIndexIdMapD,
				const unsigned int localSize,
				const unsigned int ghostSize,
				const unsigned int numberBins,
				const unsigned int totalLocallyOwnedCells,
				const unsigned int numberNodesPerElement,
				const unsigned int debugLevel,
				const unsigned int maxIter,
				const double absTol,  
				const MPI_Comm & mpiComm,
				distributedGPUVec<double> & x);
	}
}
#endif
#endif
