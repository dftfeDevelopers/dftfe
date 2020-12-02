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


#include <headers.h>
#include <cudaHelpers.h>

namespace dftfe
{
	void setupGPU()
	{
		int n_devices = 0; cudaGetDeviceCount(&n_devices); 
		//std::cout<< "Number of Devices "<<n_devices<<std::endl;
		int device_id = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)%n_devices;
		//std::cout<<"Device Id: "<<device_id<<" Task Id "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
		cudaSetDevice(device_id);
		//int device = 0;
		//cudaGetDevice(&device);
		//std::cout<< "Device Id currently used is "<<device<< " for taskId: "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
		cudaDeviceReset();
	} 
}
