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
// @author Sambit Das, David M. Rogers
//

#if defined(DFTFE_WITH_DEVICE)
#  include <iostream>

#  include <deviceDirectCCLWrapper.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceAPICalls.h>
#  include <Exceptions.h>
#  if defined(DFTFE_WITH_CUDA_NCCL)
#    include <nccl.h>
#  elif defined(DFTFE_WITH_HIP_RCCL)
#    include <rccl.h>
#  endif

namespace dftfe
{
  namespace utils
  {
    DeviceCCLWrapper::DeviceCCLWrapper()
      : d_mpiComm(MPI_COMM_NULL)
    {}

    void
    DeviceCCLWrapper::init(const MPI_Comm &mpiComm, const bool useDCCL)
    {
      MPICHECK(MPI_Comm_dup(mpiComm, &d_mpiComm));
      MPICHECK(MPI_Comm_size(mpiComm, &totalRanks));
      MPICHECK(MPI_Comm_rank(mpiComm, &myRank));
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (!ncclCommInit && useDCCL)
        {
          ncclIdPtr   = new ncclUniqueId;
          ncclCommPtr = new ncclComm_t;
          if (myRank == 0)
            ncclGetUniqueId(ncclIdPtr);
          MPICHECK(
            MPI_Bcast(ncclIdPtr, sizeof(*ncclIdPtr), MPI_BYTE, 0, d_mpiComm));
          NCCLCHECK(
            ncclCommInitRank(ncclCommPtr, totalRanks, *ncclIdPtr, myRank));
          ncclCommInit = true;
        }
#  endif
      if (!commStreamCreated)
        {
          dftfe::utils::deviceStreamCreate(&d_deviceCommStream, true);
          commStreamCreated = true;
        }
    }

    DeviceCCLWrapper::~DeviceCCLWrapper()
    {
      if (d_mpiComm != MPI_COMM_NULL)
        MPI_Comm_free(&d_mpiComm);
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          ncclCommDestroy(*ncclCommPtr);
          delete ncclCommPtr;
          delete ncclIdPtr;
        }
#  endif
      if (commStreamCreated)
        dftfe::utils::deviceStreamDestroy(d_deviceCommStream);
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const float *   send,
                                                   float *         recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif
      return 0;
    }

    int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const double *  send,
                                                   double *        recv,
                                                   int             size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif
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
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size,
                                                               send,
                                                               tempReal,
                                                               tempImag);
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal,
                                  (void *)tempReal,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag,
                                  (void *)tempImag,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size,
                                                               tempReal,
                                                               tempImag,
                                                               recv);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif


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
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size,
                                                               send,
                                                               tempReal,
                                                               tempImag);
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal,
                                  (void *)tempReal,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag,
                                  (void *)tempImag,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size,
                                                               tempReal,
                                                               tempImag,
                                                               recv);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send == recv)
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
          else
            MPICHECK(MPI_Allreduce(send,
                                   recv,
                                   size,
                                   dataTypes::mpi_type_id(recv),
                                   MPI_SUM,
                                   d_mpiComm));
        }
#  endif

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
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)send1,
                                  (void *)recv1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)send2,
                                  (void *)recv2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send1 == recv1 && send2 == recv2)
            {
              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Allreduce(send1,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(send2,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
        }
#  endif
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
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (ncclCommInit)
        {
          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size1,
                                                               send1,
                                                               tempReal1,
                                                               tempImag1);

          deviceKernelsGeneric::copyComplexArrToRealArrsDevice(size2,
                                                               send2,
                                                               tempReal2,
                                                               tempImag2);

          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)tempReal1,
                                  (void *)tempReal1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag1,
                                  (void *)tempImag1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempReal2,
                                  (void *)tempReal2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)tempImag2,
                                  (void *)tempImag2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  *ncclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size1,
                                                               tempReal1,
                                                               tempImag1,
                                                               recv1);

          deviceKernelsGeneric::copyRealArrsToComplexArrDevice(size2,
                                                               tempReal2,
                                                               tempImag2,
                                                               recv2);
        }
#  endif
#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!ncclCommInit)
        {
          dftfe::utils::deviceStreamSynchronize(stream);
          if (send1 == recv1 && send2 == recv2)
            {
              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Allreduce(send1,
                                     recv1,
                                     size1,
                                     dataTypes::mpi_type_id(recv1),
                                     MPI_SUM,
                                     d_mpiComm));

              MPICHECK(MPI_Allreduce(send2,
                                     recv2,
                                     size2,
                                     dataTypes::mpi_type_id(recv2),
                                     MPI_SUM,
                                     d_mpiComm));
            }
        }
#  endif
      return 0;
    }
  } // namespace utils
} // namespace dftfe
#endif
