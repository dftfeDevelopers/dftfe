// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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

#if defined(DFTFE_WITH_GPU)
#include <vectorUtilitiesCUDA.h>
#include <cublas_v2.h>

namespace dftfe
{

  namespace vectorToolsCUDA
  {

       void copyHostVecToCUDAVec(const double* hostVec,
                                      double* cudaVector,
                                      const unsigned int size)
       {
                  cudaMemcpy(cudaVector,
                             hostVec,
                             size*sizeof(double),
                             cudaMemcpyHostToDevice);

       }

       void copyCUDAVecToHostVec(const double* cudaVector,
                                      double* hostVec,
                                      const unsigned int size)
       {
                  cudaMemcpy(hostVec,
                             cudaVector,
                             size*sizeof(double),
                             cudaMemcpyDeviceToHost);

       }

       cudaThrustVector::cudaThrustVector()
       {
       }

       void cudaThrustVector::resize(const unsigned int size)
       {
           d_data.resize(size,0.0);
       }

       double * cudaThrustVector::begin()
       {
           return thrust::raw_pointer_cast(&d_data[0]);
       }


       const double * cudaThrustVector::begin() const
       {
           return thrust::raw_pointer_cast(&d_data[0]);
       }


       unsigned int cudaThrustVector::size() const
       {
           return d_data.size();
       }

       double cudaThrustVector::l2_norm() const
       {
            cublasHandle_t handle;
            cublasCreate(&handle);
            double result;
            cublasDnrm2(handle, 
                        size(),
                        begin(),
                        1, 
                        &result);
            cublasDestroy(handle);
            return result;
       }


       double cudaThrustVector::l1_norm() const
       {
            cublasHandle_t handle;
            cublasCreate(&handle);
            double result;
            cublasDasum(handle,
                        size(),
                        begin(),
                        1,
                        &result);
            cublasDestroy(handle);
            return result;
       }


  }//end of namespace

}
#endif
