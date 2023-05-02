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
        dealii::FESystem<3> FE(dealii::FE_Q<3>(
                                 dealii::QGaussLobatto<1>(feOrder + 1)),
                               nComponents); // linear shape function
        dealii::DoFHandler<3>       dofHandler(d_parallelTriangulationUnmoved);
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

    dealii::FESystem<3> FE(dealii::FE_Q<3>(
                             dealii::QGaussLobatto<1>(feOrder + 1)),
                           nComponents); // linear shape function
    dealii::DoFHandler<3>       dofHandler(d_parallelTriangulationUnmoved);
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

  //
  //
  //
  void
  triangulationManager::saveTriangulationsCellQuadData(
    const std::vector<const std::map<dealii::CellId, std::vector<double>> *>
      &             cellQuadDataContainerIn,
    const MPI_Comm &interpoolComm,
    const MPI_Comm &interBandGroupComm)
  {
    /*
       const unsigned int
  poolId=dealii::Utilities::MPI::this_mpi_process(interpoolComm); const unsigned
  int bandGroupId=dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
       const unsigned int
  minPoolId=dealii::Utilities::MPI::min(poolId,interpoolComm); const unsigned
  int
  minBandGroupId=dealii::Utilities::MPI::min(bandGroupId,interBandGroupComm);

       if (poolId==minPoolId && bandGroupId==minBandGroupId)
       {
       const unsigned int containerSize=cellQuadDataContainerIn.size();
       AssertThrow(containerSize!=0,ExcInternalError());

       unsigned int totalQuadVectorSize=0;
       for (unsigned int i=0; i<containerSize;++i)
       {
       const unsigned int
  quadVectorSize=(*cellQuadDataContainerIn[i]).begin()->second.size();
       Assert(quadVectorSize!=0,ExcInternalError());
       totalQuadVectorSize+=quadVectorSize;
       }

       const unsigned int dataSizeInBytes=sizeof(double)*totalQuadVectorSize;
       std::function<void(const typename
  dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
  typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
       void * data)> funcSave =
       [&](const typename
  dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
  typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
       void * data)
       {
       if (cell->active() && cell->is_locally_owned())
       {
       Assert((*cellQuadDataContainerIn[0]).find(cell->id())!=(*cellQuadDataContainerIn[0]).end(),ExcInternalError());

       double* dataStore = reinterpret_cast<double*>(data);

       double tempArray[totalQuadVectorSize];
       unsigned int count=0;
       for (unsigned int i=0; i<containerSize;++i)
       {
       const unsigned int quadVectorSize=
       (*cellQuadDataContainerIn[i]).begin()->second.size();

       for (unsigned int j=0; j<quadVectorSize;++j)
       {
       tempArray[count]=(*cellQuadDataContainerIn[i]).find(cell->id())->second[j];
       count++;
       }
       }

       std::memcpy(dataStore,
       &tempArray[0],
       dataSizeInBytes);
       }
       else
       {
       double* dataStore = reinterpret_cast<double*>(data);
       double tempArray[totalQuadVectorSize];
       std::memcpy(dataStore,
       &tempArray[0],
       dataSizeInBytes);
       }
       };


       const unsigned int offset =
  d_parallelTriangulationUnmoved.register_data_attach(dataSizeInBytes,
       funcSave);

       const std::string filename="parallelUnmovedTriaSolData.chk";
       if (std::ifstream(filename) && this_mpi_process==0)
       {
       dftUtils::moveFile(filename, filename+".old");
       dftUtils::moveFile(filename+".info", filename+".info.old");
       }
    MPI_Barrier(mpi_communicator);
    d_parallelTriangulationUnmoved.save(filename.c_str());

    saveSupportTriangulations();
  }//poolId==minPoolId check
  */
  }

  //
  //
  void
  triangulationManager::loadTriangulationsCellQuadData(
    std::vector<std::map<dealii::CellId, std::vector<double>>>
      &                              cellQuadDataContainerOut,
    const std::vector<unsigned int> &cellDataSizeContainer)
  {
    /*
       loadSupportTriangulations();
       const std::string filename="parallelUnmovedTriaSolData.chk";
       dftUtils::verifyCheckpointFileExists(filename);
       try
       {
       d_parallelTriangulationMoved.load(filename.c_str());
       d_parallelTriangulationUnmoved.load(filename.c_str());
       }
       catch (...)
       {
       AssertThrow(false, dealii::ExcMessage("DFT-FE Error: Cannot open checkpoint file-
    parallelUnmovedTriaSolData.chk or read the triangulation stored there."));
       }

       AssertThrow(cellQuadDataContainerOut.size()!=0,ExcInternalError());
       AssertThrow(cellQuadDataContainerOut.size()==cellDataSizeContainer.size(),ExcInternalError());

       const unsigned int
    totalQuadVectorSize=std::accumulate(cellDataSizeContainer.begin(),
    cellDataSizeContainer.end(), 0);

    //FIXME: The underlying function calls to register_data_attach to
    notify_ready_to_unpack
    //will need to re-evaluated after the dealii github issue #6223 is fixed
    std::function<void(const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
    void * data)> dummyFunc1 =
    [&](const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
    void * data)
    {};
    const  unsigned int offset1 =
    d_parallelTriangulationMoved.register_data_attach(totalQuadVectorSize*sizeof(double),
    dummyFunc1);


    const std::function<void(const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
    const void * data)> funcLoad =
    [&](const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
    const void * data)
    {
    if (cell->active() && cell->is_locally_owned())
    {
    const double* dataStore = reinterpret_cast<const double*>(data);

    double tempArray[totalQuadVectorSize];

    std::memcpy(&tempArray[0],
    dataStore,
    totalQuadVectorSize*sizeof(double));

    unsigned int count=0;
    for (unsigned int i=0; i<cellQuadDataContainerOut.size();++i)
    {
    Assert(cellDataSizeContainer[i]!=0,ExcInternalError());
    cellQuadDataContainerOut[i][cell->id()]=std::vector<double>(cellDataSizeContainer[i]);
    for (unsigned int j=0; j<cellDataSizeContainer[i];++j)
    {
    cellQuadDataContainerOut[i][cell->id()][j]=tempArray[count];
    count++;
    }
    }//container loop
    }
    };

    d_parallelTriangulationMoved.notify_ready_to_unpack(offset1,
    funcLoad);

    //dummy de-serialization for d_parallelTriangulationUnmoved to avoid assert
    fail in call to save
    //FIXME: This also needs to be re-evaluated after the dealii github issue
    #6223 is fixed const  unsigned int offset2 =
    d_parallelTriangulationUnmoved.register_data_attach(totalQuadVectorSize*sizeof(double),
    dummyFunc1);
    const std::function<void(const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
    const void * data)> dummyFunc2 =
      [&](const typename
    dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell, const
    typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
          const void * data)
      {};
    d_parallelTriangulationUnmoved.notify_ready_to_unpack(offset2,
        dummyFunc2);
    */
  }
} // namespace dftfe
