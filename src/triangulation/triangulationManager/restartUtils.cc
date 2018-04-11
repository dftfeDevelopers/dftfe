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
/** @file restartUtils.cc
 *
 *  @author Sambit Das
 */

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/solution_transfer.h>

namespace dftfe {
    void
    triangulationManager::saveTriangulationsSolutionVectors
				 (const dealii::DoFHandler<3> & dofHandler,
				  const std::vector< const dealii::parallel::distributed::Vector<double> * > & solutionVectors,
	                          const MPI_Comm & interpoolComm)
    {

      const unsigned int poolId=dealii::Utilities::MPI::this_mpi_process(interpoolComm);
      const unsigned int minPoolId=dealii::Utilities::MPI::min(poolId,interpoolComm);

      if (poolId==minPoolId)
      {
         dealii::parallel::distributed::SolutionTransfer<3,typename dealii::parallel::distributed::Vector<double> > solTrans(dofHandler);
         //assumes solution vectors are ghosted
         solTrans.prepare_serialization(solutionVectors);

         std::string filename="triangulationPsiDataChk";
         d_parallelTriangulationMoved.save(filename.c_str());
      }
    }

    //
    //
    void
    triangulationManager::loadTriangulationsSolutionVectors
				 (const unsigned int feOrder,
				  const unsigned int nComponents,
				  std::vector< dealii::parallel::distributed::Vector<double> * > & solutionVectors)
    {
      std::string filename="triangulationPsiDataChk";
      d_parallelTriangulationMoved.load(filename.c_str());

      dealii::FESystem<3> FE(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(feOrder+1)), nComponents); //linear shape function
      DoFHandler<3> dofHandler (d_parallelTriangulationMoved);
      dofHandler.distribute_dofs(FE);
      dealii::parallel::distributed::SolutionTransfer<3,typename dealii::parallel::distributed::Vector<double> > solTrans(dofHandler);

      for (unsigned int i=0; i< solutionVectors.size();++i)
            solutionVectors[i]->zero_out_ghosts();

      //assumes solution vectors are not ghosted
      solTrans.deserialize (solutionVectors);
    }

    //
    //
    //
    void
    triangulationManager::saveTriangulationsCellQuadData
	      (const std::vector<const std::map<dealii::CellId, std::vector<double> > *> & cellQuadDataContainerIn,
	       const MPI_Comm & interpoolComm)
    {

      const unsigned int poolId=dealii::Utilities::MPI::this_mpi_process(interpoolComm);
      const unsigned int minPoolId=dealii::Utilities::MPI::min(poolId,interpoolComm);

      if (poolId==minPoolId)
      {
	 const unsigned int containerSize=cellQuadDataContainerIn.size();
         Assert(containerSize!=0,ExcInternalError());

	 for (unsigned int i=0; i<containerSize;++i)
	 {
           const unsigned int quadVectorSize=(*cellQuadDataContainerIn[i]).begin()->second.size();
	   Assert(quadVectorSize!=0,ExcInternalError());

	   unsigned int dataSizeInBytes=sizeof(double)*quadVectorSize;
           unsigned int offset = d_parallelTriangulationMoved.register_data_attach
	          (dataSizeInBytes,
		   [&](const typename dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell,
		       const typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
		       void * data) -> void
		       {
			  if (cell->active() && cell->is_locally_owned())
			  {
			     Assert((*cellQuadDataContainerIn[i]).find(cell->id())!=(*cellQuadDataContainerIn[i]).end(),ExcInternalError());
			     Assert(quadVectorSize==(*cellQuadDataContainerIn[i]).find(cell->id())->second.size(),ExcInternalError());
			     double* dataStore = reinterpret_cast<double*>(data);

			     double tempArray[quadVectorSize];
                             for (unsigned int j=0; j<quadVectorSize;++j)
			        tempArray[j]=(*cellQuadDataContainerIn[i]).find(cell->id())->second[j];

			     std::memcpy(dataStore,
				         &tempArray[0],
					 dataSizeInBytes);
			  }
		          else
			  {
			     double* dataStore = reinterpret_cast<double*>(data);
			     double tempArray[quadVectorSize];
			     std::memcpy(dataStore,
				         &tempArray[0],
					 dataSizeInBytes);
			  }
		       }
		   );

	   pcout<< "offset=" << offset << std::endl;
           std::string filename="triangulationRhoDataChk";
           d_parallelTriangulationMoved.save(filename.c_str());
	 }

      }
    }

    //
    //
    void
    triangulationManager::loadTriangulationsCellQuadData
	       (std::vector<std::map<dealii::CellId, std::vector<double> > *> & cellQuadDataContainerOut,
		const std::vector<unsigned int>  & cellDataSizeContainer,
	        const std::vector<unsigned int>  & offsetContainer)
    {
      std::string filename="triangulationRhoDataChk";
      d_parallelTriangulationMoved.load(filename.c_str());
      const unsigned int containerSize=cellQuadDataContainerOut.size();
      Assert(containerSize!=0,ExcInternalError());

      for (unsigned int i=0; i<containerSize;++i)
      {

	d_parallelTriangulationMoved.notify_ready_to_unpack
	      (offsetContainer[i],[&](const typename dealii::parallel::distributed::Triangulation<3>::cell_iterator &cell,
		     const typename dealii::parallel::distributed::Triangulation<3>::CellStatus status,
		     const void * data) -> void
		   {
		      if (cell->active() && cell->is_locally_owned())
		      {
			 const double* dataStore = reinterpret_cast<const double*>(data);

			 double tempArray[cellDataSizeContainer[i]];

			 std::memcpy(&tempArray[0],
				     dataStore,
				     cellDataSizeContainer[i]*sizeof(double));

			 (*cellQuadDataContainerOut[i])[cell->id()]=std::vector<double>(cellDataSizeContainer[i]);
			 for (unsigned int j=0; j<cellDataSizeContainer[i];++j)
			    (*cellQuadDataContainerOut[i])[cell->id()][j]=tempArray[j];
		      }
		   }
	       );

      }//container loop
    }
}
