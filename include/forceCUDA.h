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
     void computeELocWfcEshelbyTensorNonPeriodic(const double * psiQuadValuesD,
                                                 const double * gradPsiQuadValuesXD,
                                                 const double * gradPsiQuadValuesYD,
                                                 const double * gradPsiQuadValuesZD,
                                                 const double * eigenValuesD,
                                                 const double * partialOccupanciesD,
                                                 const unsigned int numCells,
                                                 const unsigned int numPsi,
                                                 const unsigned int numQuads,
                                                 double * eshelbyTensorQuadValuesD);

     void computeELocWfcEshelbyTensorNonPeriodicH(const double * psiQuadValuesH,
                                                 const double * gradPsiQuadValuesXH,
                                                 const double * gradPsiQuadValuesYH,
                                                 const double * gradPsiQuadValuesZH,
                                                 const double * eigenValuesH,
                                                 const double * partialOccupanciesH,
                                                 const unsigned int numCells,
                                                 const unsigned int numPsi,
                                                 const unsigned int numQuads,
                                                 double * eshelbyTensorQuadValuesH);
   }
}
#endif
#endif
