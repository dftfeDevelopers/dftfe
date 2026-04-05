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
#  include <DeviceKernelLauncherHelpers.h>
#  include <DeviceAPICalls.h>
#  include <Exceptions.h>

namespace dftfe
{
  namespace utils
  {
    DeviceCCLWrapper::DeviceCCLWrapper()
      : d_mpiComm(MPI_COMM_NULL)
    {
      d_deviceDirectDCCLInstanceCounter++;
    }

    void
    DeviceCCLWrapper::init(const MPI_Comm &mpiComm, const bool useDCCL)
    {
      MPICHECK(MPI_Comm_dup(mpiComm, &d_mpiComm));
      MPICHECK(MPI_Comm_size(mpiComm, &totalRanks));
      MPICHECK(MPI_Comm_rank(mpiComm, &myRank));

#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (!dcclCommInit && useDCCL)
        {
          dcclIdPtr   = new ncclUniqueId;
          dcclCommPtr = new ncclComm_t;
          if (myRank == 0)
            ncclGetUniqueId(dcclIdPtr);
          MPICHECK(
            MPI_Bcast(dcclIdPtr, sizeof(*dcclIdPtr), MPI_BYTE, 0, d_mpiComm));
          NCCLCHECK(
            ncclCommInitRank(dcclCommPtr, totalRanks, *dcclIdPtr, myRank));
          dcclCommInit = true;
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (!dcclCommInit && useDCCL)
        {
          ccl::kvs::address_type onecclIdAddr;
          if (myRank == 0)
            {
              dcclIdPtr    = ccl::create_main_kvs();
              onecclIdAddr = dcclIdPtr->get_address();
              MPICHECK(MPI_Bcast(onecclIdAddr.data(),
                                 onecclIdAddr.size(),
                                 MPI_BYTE,
                                 0,
                                 d_mpiComm));
            }
          else
            {
              MPICHECK(MPI_Bcast(onecclIdAddr.data(),
                                 onecclIdAddr.size(),
                                 MPI_BYTE,
                                 0,
                                 d_mpiComm));
              dcclIdPtr = ccl::create_kvs(onecclIdAddr);
            }

          ccl::vector_class<ccl::pair_class<int, ccl::device>> rankDeviceMap;
          rankDeviceMap.push_back(
            {myRank, ccl::create_device(dftfe::utils::syclDevice)});
          auto onecclContext = ccl::create_context(dftfe::utils::syclContext);
          auto comms         = ccl::create_communicators(totalRanks,
                                                 rankDeviceMap,
                                                 onecclContext,
                                                 dcclIdPtr);
          dcclCommPtr =
            std::make_shared<ccl::communicator>(std::move(comms[0]));
          dcclCommInit = true;
        }
#  endif

      if (!commStreamCreated)
        {
          dftfe::utils::deviceStreamCreate(d_deviceCommStream, true);
          commStreamCreated = true;
        }
    }

    DeviceCCLWrapper::~DeviceCCLWrapper()
    {
      if (d_mpiComm != MPI_COMM_NULL)
        MPI_Comm_free(&d_mpiComm);
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          ncclCommDestroy(*dcclCommPtr);
          delete dcclCommPtr;
          delete dcclIdPtr;
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          dcclCommPtr.reset();
          dcclIdPtr.reset();
        }
#  endif

      d_deviceDirectDCCLInstanceCounter--;
      if (commStreamCreated && d_deviceDirectDCCLInstanceCounter == 0)
        {
          dftfe::utils::deviceStreamDestroy(d_deviceCommStream);
          commStreamCreated = false;
        }
    }

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const float    *send,
                                                   float          *recv,
                                                   dftfe::Int      size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclFloat,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size,
                                         ccl::datatype::float32,
                                         ccl::reduction::sum,
                                         *dcclCommPtr,
                                         devStream));
          deviceEvent_t commEvent = e.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(const double   *send,
                                                   double         *recv,
                                                   dftfe::Int      size,
                                                   deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size,
                                  ncclDouble,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size,
                                         ccl::datatype::float64,
                                         ccl::reduction::sum,
                                         *dcclCommPtr,
                                         devStream));
          deviceEvent_t commEvent = e.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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


    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<double> *send,
      std::complex<double>       *recv,
      dftfe::Int                  size,
      deviceStream_t             &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size * 2,
                                  ncclDouble,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size * 2,
                                         ccl::datatype::float64,
                                         ccl::reduction::sum,
                                         *dcclCommPtr,
                                         devStream));
          deviceEvent_t commEvent = e.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceWrapper(
      const std::complex<float> *send,
      std::complex<float>       *recv,
      dftfe::Int                 size,
      deviceStream_t            &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclAllReduce((const void *)send,
                                  (void *)recv,
                                  size * 2,
                                  ncclFloat,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));
          ccl::event e;
          ONECCLCHECK(e = ccl::allreduce((const void *)send,
                                         (void *)recv,
                                         size * 2,
                                         ccl::datatype::float32,
                                         ccl::reduction::sum,
                                         *dcclCommPtr,
                                         devStream));
          deviceEvent_t commEvent = e.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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


    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const double   *send1,
      const float    *send2,
      double         *recv1,
      float          *recv2,
      dftfe::Int      size1,
      dftfe::Int      size2,
      deviceStream_t &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)send1,
                                  (void *)recv1,
                                  size1,
                                  ncclDouble,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)send2,
                                  (void *)recv2,
                                  size2,
                                  ncclFloat,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
        }
#  endif
#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));

          ccl::event e1, e2;
          ONECCLCHECK(ccl::group_start());

          ONECCLCHECK(e1 = ccl::allreduce((const void *)send1,
                                          (void *)recv1,
                                          size1,
                                          ccl::datatype::float64,
                                          ccl::reduction::sum,
                                          *dcclCommPtr,
                                          devStream));

          ONECCLCHECK(e2 = ccl::allreduce((const void *)send2,
                                          (void *)recv2,
                                          size2,
                                          ccl::datatype::float32,
                                          ccl::reduction::sum,
                                          *dcclCommPtr,
                                          devStream));

          ONECCLCHECK(ccl::group_end());
          deviceEvent_t commEvent1 = e1.get_native();
          deviceEvent_t commEvent2 = e2.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent1, 0);
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent2, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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

    dftfe::Int
    DeviceCCLWrapper::deviceDirectAllReduceMixedPrecGroupWrapper(
      const std::complex<double> *send1,
      const std::complex<float>  *send2,
      std::complex<double>       *recv1,
      std::complex<float>        *recv2,
      dftfe::Int                  size1,
      dftfe::Int                  size2,
      deviceStream_t             &stream)
    {
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      if (dcclCommInit)
        {
          NCCLCHECK(ncclGroupStart());
          NCCLCHECK(ncclAllReduce((const void *)send1,
                                  (void *)recv1,
                                  size1 * 2,
                                  ncclDouble,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
          NCCLCHECK(ncclAllReduce((const void *)send2,
                                  (void *)recv2,
                                  size2 * 2,
                                  ncclFloat,
                                  ncclSum,
                                  *dcclCommPtr,
                                  stream));
          NCCLCHECK(ncclGroupEnd());
        }
#  endif

#  if defined(DFTFE_WITH_SYCL_ONECCL)
      if (dcclCommInit)
        {
          auto devStream =
            ccl::create_stream(dftfe::utils::queueRegistry.at(stream));

          ccl::event e1, e2;
          ONECCLCHECK(ccl::group_start());

          ONECCLCHECK(e1 = ccl::allreduce((const void *)send1,
                                          (void *)recv1,
                                          size1 * 2,
                                          ccl::datatype::float64,
                                          ccl::reduction::sum,
                                          *dcclCommPtr,
                                          devStream));

          ONECCLCHECK(e2 = ccl::allreduce((const void *)send2,
                                          (void *)recv2,
                                          size2 * 2,
                                          ccl::datatype::float32,
                                          ccl::reduction::sum,
                                          *dcclCommPtr,
                                          devStream));

          ONECCLCHECK(ccl::group_end());
          deviceEvent_t commEvent1 = e1.get_native();
          deviceEvent_t commEvent2 = e2.get_native();
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent1, 0);
          dftfe::utils::deviceStreamWaitEvent(stream, commEvent2, 0);
        }
#  endif

#  if defined(DFTFE_WITH_DEVICE_AWARE_MPI)
      if (!dcclCommInit)
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
