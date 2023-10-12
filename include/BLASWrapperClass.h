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

#ifndef BLASWrapperClass_h
#define BLASWrapperClass_h

#    include <complex>
#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>
#    include "process_grid.h"
#    include "scalapackWrapper.h"
#    include <liearAlgebraOperations.h>


namespace dftfe
{
    template<dftfe::utils::MemorySpace memorySpace>    
    class BLASWrapperClassBase
    {
        public:
        //Real-Single Precision GEMM
        void
        xgemm(const char *        transA,
        const char *        transB,
        const unsigned int *m,
        const unsigned int *n,
        const unsigned int *k,
        const float *       alpha,
        const float *       A,
        const unsigned int *lda,
        const float *       B,
        const unsigned int *ldb,
        const float *       beta,
        float *             C,
        const unsigned int *ldc);
        //Complex-Single Precision GEMM
        void
        xgemm(const char *               transA,
        const char *               transB,
        const unsigned int *       m,
        const unsigned int *       n,
        const unsigned int *       k,
        const std::complex<float> *alpha,
        const std::complex<float> *A,
        const unsigned int *       lda,
        const std::complex<float> *B,
        const unsigned int *       ldb,
        const std::complex<float> *beta,
        std::complex<float> *      C,
        const unsigned int *       ldc);
        
        //Real-double precison GEMM
        void
        xgemm(const char *        transA,
        const char *        transB,
        const unsigned int *m,
        const unsigned int *n,
        const unsigned int *k,
        const double *      alpha,
        const double *      A,
        const unsigned int *lda,
        const double *      B,
        const unsigned int *ldb,
        const double *      beta,
        double *            C,
        const unsigned int *ldc);


        //Complex-double precision GEMM
        void
        xgemm(const char *                transA,
        const char *                transB,
        const unsigned int *        m,
        const unsigned int *        n,
        const unsigned int *        k,
        const std::complex<double> *alpha,
        const std::complex<double> *A,
        const unsigned int *        lda,
        const std::complex<double> *B,
        const unsigned int *        ldb,
        const std::complex<double> *beta,
        std::complex<double> *      C,
        const unsigned int *        ldc);

        //Real-Double scaling of Real-vector
        void
        xscal(const unsigned int *n,
           const double *      alpha,
           double *            x,
           const unsigned int *inc);

        //Real-Float scaling of Real-vector
        void
        xscal(const unsigned int *n,
           const float *       alpha,
           float *             x,
           const unsigned int *inc);

        //Complex-double scaling of complex-vector
        void
        xscal(const unsigned int *        n,
           const std::complex<double> *alpha,
           std::complex<double> *      x,
           const unsigned int *        inc);
        
        //Real-double scaling of complex-vector
        void
        xscal(const unsigned int *  n,
            const double *        alpha,
            std::complex<double> *x,
            const unsigned int *  inc);

        private:

    };

    class BLASWrapperClass<dftfe::utils::MemorySpace::HOST> : public BLASWrapperClassBase<dftfe::utils::MemorySpace::HOST>
    {};

#if defined(DFTFE_WITH_DEVICE)
    class BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE> : public BLASWrapperClassBase<dftfe::utils::MemorySpace::DEVICE>
    {}; 

#endif

}


#endif