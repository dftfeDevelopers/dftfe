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
// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das (2018), Denis Davydov (2018)
//


/** @file dftUtils.cc
 *  @brief Contains repeatedly used functions in the KSDFT calculations
 *
 */

#include <dftUtils.h>
#include <iostream>
#include <fstream>
#include <dftParameters.h>

namespace dftfe {

namespace dftUtils
{
  double getPartialOccupancy(const double eigenValue,const double fermiEnergy,const double kb,const double T)
  {
    const double factor=(eigenValue-fermiEnergy)/(kb*T);
    return (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
  }

  void transformDomainBoundingVectors(std::vector<std::vector<double> > & domainBoundingVectors,
	                               const dealii::Tensor<2,3,double> & deformationGradient)
  {
      for (unsigned int idim=0; idim<3;++idim)
      {
	  dealii::Tensor<1,3,double> domainVector;
	  for (unsigned int jdim=0; jdim<3;++jdim)
          {
	      domainVector[jdim]=domainBoundingVectors[idim][jdim];
	  }

	  domainVector=deformationGradient*domainVector;

	  for (unsigned int jdim=0; jdim<3;++jdim)
          {
	      domainBoundingVectors[idim][jdim]=domainVector[jdim];
	  }

      }
  }


  void writeDataVTUParallelLowestPoolId(const dealii::DataOut<3> & dataOut,
	                                const MPI_Comm & intrapoolcomm,
				        const MPI_Comm & interpoolcomm,
	                                const std::string & fileName)
  {
    const unsigned int poolId=dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const unsigned int minPoolId=dealii::Utilities::MPI::min(poolId,interpoolcomm);

    if (poolId==minPoolId)
    {
      std::string fileNameVTU=fileName+".vtu";
      dataOut.write_vtu_in_parallel(fileNameVTU.c_str(),intrapoolcomm);
    }
  }


  Pool::Pool(const MPI_Comm &mpi_communicator,
             const unsigned int npool)
  {
    const unsigned int n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    AssertThrow(n_mpi_processes % npool == 0,
                dealii::ExcMessage("Number of mpi processes must be a multiple of NUMBER OF POOLS"));
    const unsigned int poolSize= n_mpi_processes/npool;
    const unsigned int taskId = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // FIXME: any and all terminal output should be optional
    if (taskId == 0 && dftParameters::verbosity>=2)
      {
        std::cout<<"Number of pools: "<<npool<<std::endl;
        std::cout<<"Pool size (number of MPI processes for domain decomposition): "<<poolSize<<std::endl;
      }
    MPI_Barrier(mpi_communicator);

    const unsigned int color1 = taskId%poolSize ;
    MPI_Comm_split(mpi_communicator,
                   color1,
                   0,
                   &interpoolcomm);
    MPI_Barrier(mpi_communicator);

    const unsigned int color2 = taskId / poolSize ;
    MPI_Comm_split(mpi_communicator,
                   color2,
                   0,
                   &intrapoolcomm);

    // FIXME: why do we need a duplicate?
    MPI_Comm_dup(intrapoolcomm , &mpi_comm_replica);

    // FIXME: output should be optional
    for (unsigned int i=0; i<n_mpi_processes; ++i)
      {
        if (taskId==i)
	{
           if (dftParameters::verbosity==1)
             std::cout << " My global id is " << taskId << " , pool id is " << dealii::Utilities::MPI::this_mpi_process(interpoolcomm)  <<
                    " , intrapool id is " << dealii::Utilities::MPI::this_mpi_process(intrapoolcomm) << std::endl;
	}
        MPI_Barrier(mpi_communicator);
      }
  }

  MPI_Comm &Pool::get_replica_comm()
  {
    return mpi_comm_replica;
  }

  MPI_Comm &Pool::get_interpool_comm()
  {
    return interpoolcomm;
  }

  MPI_Comm &Pool::get_intrapool_comm()
  {
    return intrapoolcomm;
  }

}

}
