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
#ifndef forceCUDA_H_
#define forceCUDA_H_

#include <headers.h>
#include <operatorCUDA.h>
#include <vectorUtilitiesCUDA.h>

namespace dftfe
{
   namespace forceCUDA
   {
     void computeELocWfcEshelbyTensorNonPeriodicH(operatorDFTCUDAClass & operatorMatrix,
                                                 const double * psiQuadValuesH,
                                                 const double * gradPsiQuadValuesXH,
                                                 const double * gradPsiQuadValuesYH,
                                                 const double * gradPsiQuadValuesZH,
                                                 const double * eigenValuesH,
                                                 const double * partialOccupanciesH,
                                                 const unsigned int numCells,
                                                 const unsigned int numQuads,
                                                 const unsigned int numPsi,
                                                 double * eshelbyTensorQuadValuesH00,
                                                 double * eshelbyTensorQuadValuesH10,
                                                 double * eshelbyTensorQuadValuesH11,
                                                 double * eshelbyTensorQuadValuesH20,
                                                 double * eshelbyTensorQuadValuesH21,
                                                 double * eshelbyTensorQuadValuesH22);

     void computeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
                                                  const thrust::device_vector<double> & psiQuadValuesD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesXD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesYD,
                                                  const thrust::device_vector<double> & gradPsiQuadValuesZD,
                                                  const thrust::device_vector<double> & eigenValuesD,
                                                  const thrust::device_vector<double> & partialOccupanciesD,
                                                  const unsigned int numCells,
                                                  const unsigned int numQuads,
                                                  const unsigned int numPsi,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD22);

     void computeNonLocalProjectorKetTimesPsiTimesVH(operatorDFTCUDAClass & operatorMatrix,
                                                     const double * X,
                                                     const unsigned int startingVecId,
                                                     const unsigned int BVec,
                                                     const unsigned int N,
                                                     double * projectorKetTimesPsiTimesVH);
   }
}
#endif
#endif
