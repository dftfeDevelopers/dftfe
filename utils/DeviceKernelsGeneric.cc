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
// @author Sambit Das, Gourab Panigrahi
//


#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherHelpers.h>
#include <DeviceAPICalls.h>

namespace dftfe
{
  namespace utils
  {
    namespace deviceKernelsGeneric
    {
      void
      setupDevice(const int &mpi_rank)
      {
        int n_devices = 0;
        dftfe::utils::getDeviceCount(&n_devices);
        // std::cout<< "Number of Devices "<<n_devices<<std::endl;
        int device_id = mpi_rank % n_devices;
        // std::cout<<"Device Id: "<<device_id<<" Task Id
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::setDevice(device_id);
        // dftfe::Int device = 0;
        // dftfe::utils::getDevice(&device);
        // std::cout<< "Device Id currently used is "<<device<< " for taskId:
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::deviceReset();
      }
    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe
