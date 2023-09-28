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
/** @file restartUtils.cc
 *
 *  @author Sambit Das
 */

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/numerics/solution_transfer.h>

namespace dftfe
{
  //
  void
  triangulationManager::saveSupportTriangulations(std::string path)
  {
    if (d_serialTriangulationUnmoved.n_global_active_cells() != 0 &&
        this_mpi_process == 0)
      {
        const std::string filename1 = path + "/serialUnmovedTria.chk";
        if (std::ifstream(filename1))
          {
            dftUtils::moveFile(filename1, filename1 + ".old");
            dftUtils::moveFile(filename1 + ".info", filename1 + ".info.old");
          }

        d_serialTriangulationUnmoved.save(filename1.c_str());
      }
  }

  //
  void
  triangulationManager::loadSupportTriangulations(std::string path)
  {
    if (d_serialTriangulationUnmoved.n_global_active_cells() != 0)
      {
        const std::string filename1 = path + "/serialUnmovedTria.chk";
        dftUtils::verifyCheckpointFileExists(filename1);
        try
          {
            d_serialTriangulationUnmoved.load(filename1.c_str(), false);
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                "DFT-FE Error: Cannot open checkpoint file- serialUnmovedTria.chk or read the triangulation stored there."));
          }
      }
  }

  //
  void
  triangulationManager::saveTriangulationsSolutionVectors(
    std::string                                           path,
    const unsigned int                                    feOrder,
    const unsigned int                                    nComponents,
    const std::vector<const distributedCPUVec<double> *> &solutionVectors,
    const MPI_Comm &                                      interpoolComm,
    const MPI_Comm &                                      interBandGroupComm)
  {
    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolComm);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    const unsigned int minPoolId =
      dealii::Utilities::MPI::min(poolId, interpoolComm);
    const unsigned int minBandGroupId =
      dealii::Utilities::MPI::min(bandGroupId, interBandGroupComm);

    if (poolId == minPoolId && bandGroupId == minBandGroupId)
      {
        dealii::FESystem<3>   FE(dealii::FE_Q<3>(
                                 dealii::QGaussLobatto<1>(feOrder + 1)),
                               nComponents); // linear shape function
        dealii::DoFHandler<3> dofHandler(d_parallelTriangulationUnmoved);
        dofHandler.distribute_dofs(FE);

        dealii::parallel::distributed::
          SolutionTransfer<3, distributedCPUVec<double>>
            solTrans(dofHandler);
        // assumes solution vectors are ghosted
        solTrans.prepare_for_serialization(solutionVectors);

        const std::string filename = path + "/parallelUnmovedTriaSolData.chk";
        if (std::ifstream(filename) && this_mpi_process == 0)
          {
            dftUtils::moveFile(filename, filename + ".old");
            dftUtils::moveFile(filename + ".info", filename + ".info.old");
          }
        MPI_Barrier(mpi_communicator);

        d_parallelTriangulationUnmoved.save(filename.c_str());

        saveSupportTriangulations(path);
      }
  }

  //
  //
  void
  triangulationManager::loadTriangulationsSolutionVectors(
    std::string                               path,
    const unsigned int                        feOrder,
    const unsigned int                        nComponents,
    std::vector<distributedCPUVec<double> *> &solutionVectors)
  {
    loadSupportTriangulations(path);
    const std::string filename = path + "/parallelUnmovedTriaSolData.chk";
    dftUtils::verifyCheckpointFileExists(filename);
    try
      {
        d_parallelTriangulationMoved.load(filename.c_str());
        d_parallelTriangulationUnmoved.load(filename.c_str());
      }
    catch (...)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Cannot open checkpoint file- parallelUnmovedTriaSolData.chk or read the triangulation stored there."));
      }

    dealii::FESystem<3>   FE(dealii::FE_Q<3>(
                             dealii::QGaussLobatto<1>(feOrder + 1)),
                           nComponents); // linear shape function
    dealii::DoFHandler<3> dofHandler(d_parallelTriangulationUnmoved);
    dofHandler.distribute_dofs(FE);
    dealii::parallel::distributed::
      SolutionTransfer<3, typename dftfe::distributedCPUVec<double>>
        solTrans(dofHandler);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandler,
                                                    locally_relevant_dofs);

    const dealii::IndexSet &locally_owned_dofs =
      dofHandler.locally_owned_dofs();
    dealii::IndexSet ghost_indices = locally_relevant_dofs;
    ghost_indices.subtract_set(locally_owned_dofs);

    for (unsigned int i = 0; i < solutionVectors.size(); ++i)
      {
        solutionVectors[i]->reinit(locally_owned_dofs,
                                   ghost_indices,
                                   mpi_communicator);
        solutionVectors[i]->zero_out_ghosts();
      }

    // assumes solution vectors are not ghosted
    solTrans.deserialize(solutionVectors);
  }
} // namespace dftfe
