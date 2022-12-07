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
// @author Sambit Das, David M. Rogers
//

#if defined(DFTFE_WITH_DEVICE)
#  if defined(DFTFE_WITH_DEVICE_CUDA)
#    include <iostream>

#    include <deviceDirectCCLWrapper.h>
#    include <deviceHelpers.h>
#    include <DeviceDataTypeOverloads.h>
#    include <DeviceKernelLauncherConstants.h>
#    if defined(DFTFE_WITH_NCCL)
#      include <nccl.h>
#    endif

namespace dftfe
{
  namespace utils
  {
#    define MPICHECK(cmd)                                                      \
      do                                                                       \
        {                                                                      \
          int e = cmd;                                                         \
          if (e != MPI_SUCCESS)                                                \
            {                                                                  \
              printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
              exit(EXIT_FAILURE);                                              \
            }                                                                  \
        }                                                                      \
      while (0)

#    if defined(DFTFE_WITH_NCCL)
#      define NCCLCHECK(cmd)                              \
        do                                                \
          {                                               \
            ncclResult_t r = cmd;                         \
            if (r != ncclSuccess)                         \
              {                                           \
                printf("Failed, NCCL error %s:%d '%s'\n", \
                       __FILE__,                          \
                       __LINE__,                          \
                       ncclGetErrorString(r));            \
                exit(EXIT_FAILURE);                       \
              }                                           \
          }                                               \
        while (0)
#    endif

    DeviceCCLWrapper::DeviceCCLWrapper()
      : commCreated(false)
    {}

    void
    DeviceCCLWrapper::init(const MPI_Comm &mpiComm)
    {
      MPICHECK(MPI_Comm_size(mpiComm, &totalRanks));
      MPICHECK(MPI_Comm_rank(mpiComm, &myRank));
#    ifdef DFTFE_WITH_NCCL
      ncclIdPtr   = (void *)(new ncclUniqueId);
      ncclCommPtr = (void *)(new ncclComm_t);
      if (myRank == 0)
        ncclGetUniqueId((ncclUniqueId *)ncclIdPtr);
      MPICHECK(MPI_Bcast(
        ncclIdPtr, sizeof(*((ncclUniqueId *)ncclIdPtr)), MPI_BYTE, 0, mpiComm));
      NCCLCHECK(ncclCommInitRank((ncclComm_t *)ncclCommPtr,
                                 totalRanks,
                                 *((ncclUniqueId *)ncclIdPtr),
                                 myRank));
      commCreated = true;
#    endif
    }

    DeviceCCLWrapper::~DeviceCCLWrapper()
    {
#    ifdef DFTFE_WITH_NCCL
      if (commCreated)
        {
          ncclCommDestroy(*((ncclComm_t *)ncclCommPtr));
          delete (ncclComm_t *)ncclCommPtr;
          delete (ncclUniqueId *)ncclIdPtr;
        }
#    endif
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const float *   send,
                                                   float *         recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#    ifdef DFTFE_WITH_NCCL
      NCCLCHECK(ncclAllReduce((const void *)send,
                              (void *)recv,
                              size,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
#    endif
      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const double *  send,
                                                   double *        recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#    ifdef DFTFE_WITH_NCCL
      NCCLCHECK(ncclAllReduce((const void *)send,
                              (void *)recv,
                              size,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
#    endif
      return 0;
    }


    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<double> *send,
      std::complex<double> *      recv,
      int                         size,
      double *                    tempReal,
      double *                    tempImag,
      deviceStream_t &            stream)
    {
      deviceUtils::copyComplexArrToRealArrsDevice(size,
                                                  send,
                                                  tempReal,
                                                  tempImag);
#    ifdef DFTFE_WITH_NCCL
      ncclGroupStart();
      NCCLCHECK(ncclAllReduce((const void *)tempReal,
                              (void *)tempReal,
                              size,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)tempImag,
                              (void *)tempImag,
                              size,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      ncclGroupEnd();
#    endif

      deviceUtils::copyRealArrsToComplexArrDevice(size,
                                                  tempReal,
                                                  tempImag,
                                                  recv);
      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<float> *send,
      std::complex<float> *      recv,
      int                        size,
      float *                    tempReal,
      float *                    tempImag,
      deviceStream_t &           stream)
    {
      deviceUtils::copyComplexArrToRealArrsDevice(size,
                                                  send,
                                                  tempReal,
                                                  tempImag);
#    ifdef DFTFE_WITH_NCCL
      ncclGroupStart();
      NCCLCHECK(ncclAllReduce((const void *)tempReal,
                              (void *)tempReal,
                              size,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)tempImag,
                              (void *)tempImag,
                              size,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      ncclGroupEnd();
#    endif

      deviceUtils::copyRealArrsToComplexArrDevice(size,
                                                  tempReal,
                                                  tempImag,
                                                  recv);
      return 0;
    }


    int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const double *  send1,
      const float *   send2,
      double *        recv1,
      float *         recv2,
      int             size1,
      int             size2,
      deviceStream_t &stream)
    {
#    ifdef DFTFE_WITH_NCCL
      ncclGroupStart();
      NCCLCHECK(ncclAllReduce((const void *)send1,
                              (void *)recv1,
                              size1,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)send2,
                              (void *)recv2,
                              size2,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      ncclGroupEnd();
#    endif
      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const std::complex<double> *send1,
      const std::complex<float> * send2,
      std::complex<double> *      recv1,
      std::complex<float> *       recv2,
      int                         size1,
      int                         size2,
      double *                    tempReal1,
      float *                     tempReal2,
      double *                    tempImag1,
      float *                     tempImag2,
      deviceStream_t &            stream)
    {
      deviceUtils::copyComplexArrToRealArrsDevice(size1,
                                                  send1,
                                                  tempReal1,
                                                  tempImag1);

      deviceUtils::copyComplexArrToRealArrsDevice(size2,
                                                  send2,
                                                  tempReal2,
                                                  tempImag2);

#    ifdef DFTFE_WITH_NCCL
      ncclGroupStart();
      NCCLCHECK(ncclAllReduce((const void *)tempReal1,
                              (void *)tempReal1,
                              size1,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)tempImag1,
                              (void *)tempImag1,
                              size1,
                              ncclDouble,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)tempReal2,
                              (void *)tempReal2,
                              size2,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      NCCLCHECK(ncclAllReduce((const void *)tempImag2,
                              (void *)tempImag2,
                              size2,
                              ncclFloat,
                              ncclSum,
                              *((ncclComm_t *)ncclCommPtr),
                              stream));
      ncclGroupEnd();
#    endif

      deviceUtils::copyRealArrsToComplexArrDevice(size1,
                                                  tempReal1,
                                                  tempImag1,
                                                  recv1);

      deviceUtils::copyRealArrsToComplexArrDevice(size2,
                                                  tempReal2,
                                                  tempImag2,
                                                  recv2);

      return 0;
    }
  } // namespace utils
} // namespace dftfe
#  endif
#endif
