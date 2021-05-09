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
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das, Denis Davydov
//


/** @file dftUtils.cc
 *  @brief Contains repeatedly used functions in the KSDFT calculations
 *
 */

#include <dftParameters.h>
#include <dftUtils.h>

#include <fstream>
#include <iostream>

namespace dftfe
{
  namespace dftUtils
  {
    double
    getCompositeGeneratorVal(const double rc,
                             const double r,
                             const double a0,
                             const double power)
    {
      if (r <= rc)
        return 1.0;
      else
        return std::exp(-std::pow((r - rc) / a0, power));
    }

    double
    getPartialOccupancy(const double eigenValue,
                        const double fermiEnergy,
                        const double kb,
                        const double T)
    {
      const double factor = (eigenValue - fermiEnergy) / (kb * T);
      return (factor >= 0) ? std::exp(-factor) / (1.0 + std::exp(-factor)) :
                             1.0 / (1.0 + std::exp(factor));
    }

    dealii::BoundingBox<3>
    createBoundingBoxForSphere(const dealii::Point<3> &center,
                               const double            sphereRadius)
    {
      dealii::Tensor<1, 3, double> tempDisp;
      tempDisp[0] = sphereRadius;
      tempDisp[1] = sphereRadius;
      tempDisp[2] = sphereRadius;
      std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
        boundaryPoints;
      boundaryPoints.first  = center - tempDisp;
      boundaryPoints.second = center + tempDisp;
      dealii::BoundingBox<3> boundingBox(boundaryPoints);
      return boundingBox;
    }

    void
    cross_product(const std::vector<double> &a,
                  const std::vector<double> &b,
                  std::vector<double> &      crossProductVector)
    {
      std::vector<double> crossProduct(a.size(), 0.0);
      crossProduct[0] = a[1] * b[2] - a[2] * b[1];
      crossProduct[1] = a[2] * b[0] - a[0] * b[2];
      crossProduct[2] = a[0] * b[1] - a[1] * b[0];

      crossProductVector = crossProduct;
    }

    void
    transformDomainBoundingVectors(
      std::vector<std::vector<double>> &  domainBoundingVectors,
      const dealii::Tensor<2, 3, double> &deformationGradient)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
        {
          dealii::Tensor<1, 3, double> domainVector;
          for (unsigned int jdim = 0; jdim < 3; ++jdim)
            {
              domainVector[jdim] = domainBoundingVectors[idim][jdim];
            }

          domainVector = deformationGradient * domainVector;

          for (unsigned int jdim = 0; jdim < 3; ++jdim)
            {
              domainBoundingVectors[idim][jdim] = domainVector[jdim];
            }
        }
    }

    void
    printCurrentMemoryUsage(const MPI_Comm &mpiComm, const std::string message)
    {
#ifdef USE_PETSC
      PetscLogDouble bytes;
      PetscMemoryGetCurrentUsage(&bytes);
      const double       maxBytes = dealii::Utilities::MPI::max(bytes, mpiComm);
      const unsigned int taskId =
        dealii::Utilities::MPI::this_mpi_process(mpiComm);
      if (taskId == 0)
        std::cout << std::endl
                  << message +
                       ", Current maximum memory usage across all processors: "
                  << maxBytes / 1.0e+6 << " MB." << std::endl
                  << std::endl;
      MPI_Barrier(mpiComm);
#endif
    }

    void
    writeDataVTUParallelLowestPoolId(const dealii::DoFHandler<3> &dofHandler,
                                     const dealii::DataOut<3> &   dataOut,
                                     const MPI_Comm &             domainComm,
                                     const MPI_Comm &             kPointComm,
                                     const MPI_Comm &             bandGroupComm,
                                     const std::string &          folderName,
                                     const std::string &          fileName)
    {
      const unsigned int poolId =
        dealii::Utilities::MPI::this_mpi_process(kPointComm);
      const unsigned int bandGroupId =
        dealii::Utilities::MPI::this_mpi_process(bandGroupComm);
      const unsigned int minPoolId =
        dealii::Utilities::MPI::min(poolId, kPointComm);
      const unsigned int minBandGroupId =
        dealii::Utilities::MPI::min(bandGroupId, bandGroupComm);


      unsigned int n_mpi_processes;
      if (poolId == minPoolId && bandGroupId == minBandGroupId)
        {
          /*std::vector<dealii::types::subdomain_id>
            partition_int(dofHandler.get_triangulation().n_active_cells());
            dealii::GridTools::get_subdomain_association(dofHandler.get_triangulation(),
            partition_int);
            const Vector<double> partitioning(partition_int.begin(),
            partition_int.end());
            dataOut.add_data_vector(partitioning,"partitioning");
            dataOut.build_patches();*/

          const unsigned int this_mpi_process =
            dealii::Utilities::MPI::this_mpi_process(domainComm);
          n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(domainComm);
          std::string outFileName =
            folderName + "/" + fileName + "_" +
            dealii::Utilities::to_string(this_mpi_process) + ".vtu";
          std::ofstream output(outFileName);
          dataOut.write_vtu(output);
        }


      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::vector<std::string> filenames;
          for (unsigned int i = 0; i < n_mpi_processes; ++i)
            filenames.push_back(fileName + "_" +
                                dealii::Utilities::to_string(i) + ".vtu");
          const std::string visit_master_filename =
            (folderName + "/" + fileName + "_master.visit");
          std::ofstream visit_master(visit_master_filename.c_str());
          dealii::DataOutBase::write_visit_record(visit_master, filenames);
          const std::string pvtu_master_filename =
            (folderName + "/" + fileName + "_master.pvtu");
          std::ofstream pvtu_master(pvtu_master_filename.c_str());
          dataOut.write_pvtu_record(pvtu_master, filenames);
          /* static std::vector<std::pair<double, std::string>> times_and_names;
             times_and_names.push_back(
             std::pair<double, std::string>(present_time,
             pvtu_master_filename)); std::ofstream pvd_output("solution.pvd");
             dealii::DataOutBase::write_pvd_record(pvd_output,
             times_and_names);*/
        }
    }

    void
    createBandParallelizationIndices(
      const MPI_Comm &           interBandGroupComm,
      const unsigned int         numBands,
      std::vector<unsigned int> &bandGroupLowHighPlusOneIndices)
    {
      bandGroupLowHighPlusOneIndices.clear();
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int wfcBlockSizeBandGroup = numBands / numberBandGroups;
      AssertThrow(
        wfcBlockSizeBandGroup != 0,
        dealii::ExcMessage(
          "DFT-FE Error: NPBAND is more than either total number of bands or total number of top states in case of spectrum splitting."));
      bandGroupLowHighPlusOneIndices.resize(numberBandGroups * 2);
      for (unsigned int i = 0; i < numberBandGroups; i++)
        {
          bandGroupLowHighPlusOneIndices[2 * i] = i * wfcBlockSizeBandGroup;
          bandGroupLowHighPlusOneIndices[2 * i + 1] =
            (i + 1) * wfcBlockSizeBandGroup;
        }
      bandGroupLowHighPlusOneIndices[2 * numberBandGroups - 1] = numBands;
    }

    Pool::Pool(const MPI_Comm &mpi_communicator, const unsigned int npool)
    {
      const unsigned int n_mpi_processes =
        dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
      AssertThrow(
        n_mpi_processes % npool == 0,
        dealii::ExcMessage(
          "DFT-FE Error: Total number of mpi processes must be a multiple of npool. Please check that total number of mpi processes is a multiple of NPKPT*NPBAND."));
      const unsigned int poolSize = n_mpi_processes / npool;
      const unsigned int taskId =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

      // FIXME: any and all terminal output should be optional
      if (taskId == 0 && dftParameters::verbosity > 4)
        {
          std::cout << "Number of pools: " << npool << std::endl;
          std::cout << "Pool size: " << poolSize << std::endl;
        }
      MPI_Barrier(mpi_communicator);

      const unsigned int color1 = taskId % poolSize;
      MPI_Comm_split(mpi_communicator, color1, 0, &interpoolcomm);
      MPI_Barrier(mpi_communicator);

      const unsigned int color2 = taskId / poolSize;
      MPI_Comm_split(mpi_communicator, color2, 0, &intrapoolcomm);

      // FIXME: output should be optional
      for (unsigned int i = 0; i < n_mpi_processes; ++i)
        {
          if (taskId == i)
            {
              if (dftParameters::verbosity > 4)
                std::cout
                  << " My global id is " << taskId << " , pool id is "
                  << dealii::Utilities::MPI::this_mpi_process(interpoolcomm)
                  << " , intrapool id is "
                  << dealii::Utilities::MPI::this_mpi_process(intrapoolcomm)
                  << std::endl;
            }
          MPI_Barrier(mpi_communicator);
        }
    }

    MPI_Comm &
    Pool::get_interpool_comm()
    {
      return interpoolcomm;
    }

    MPI_Comm &
    Pool::get_intrapool_comm()
    {
      return intrapoolcomm;
    }

  } // namespace dftUtils

} // namespace dftfe
