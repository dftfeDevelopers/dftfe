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
// @author Sambit Das
//



#include <interpolateFieldsFromPreviousMesh.h>
#include <dftParameters.h>

namespace dftfe
{

namespace vectorTools
{
//
//constructor
//
interpolateFieldsFromPreviousMesh::interpolateFieldsFromPreviousMesh(const MPI_Comm &mpi_comm):
  mpi_communicator (mpi_comm),
  n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm)),
  this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm)),
  pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{
}

void interpolateFieldsFromPreviousMesh::interpolate
                  (const dealii::parallel::distributed::Triangulation<3> & triangulationSerPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParCurrent,
		   const dealii::FESystem<3> & FEPrev,
		   const dealii::FESystem<3> & FECurrent,
		   const std::vector<vectorType*> & fieldsPreviousMesh,
		   std::vector<vectorType*> & fieldsCurrentMesh,
		   const dealii::ConstraintMatrix * constraintsCurrentPtr)
{
  AssertThrow(FEPrev.components==FECurrent.components,dealii::ExcMessage("FEPrev and FECurrent must have the same number of components."));

  const unsigned int dofs_per_cell_current = FECurrent.dofs_per_cell;
  const unsigned int fe_components=FECurrent.components;
  const unsigned int base_indices_per_cell_current = dofs_per_cell_current/fe_components;

  /// compute-time logger
  dealii::TimerOutput computing_timer(pcout,
		                     dftParameters::reproducible_output ||
				     dftParameters::verbosity<2 ? dealii::TimerOutput::never:
				     dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);

  ///////////////////////////////////////////////////////////////////////////////
  //Step1: create maps which will be used for the MPI calls in the later steps///
  ///////////////////////////////////////////////////////////////////////////////

  computing_timer.enter_section("interpolate:step1");
  dealii::DoFHandler<3> dofHandlerUnmovedParPrev(triangulationParPrev);
  dofHandlerUnmovedParPrev.distribute_dofs(FEPrev);

  typename dealii::parallel::distributed::Triangulation<3>::active_cell_iterator cellTria= triangulationSerPrev.begin_active(), endcTria= triangulationSerPrev.end();
  const unsigned int numGlobalCellsPrev=triangulationSerPrev.n_global_active_cells();

  std::map<dealii::CellId, unsigned int> globalCellIdToCellNumSerPrev;
  std::map<dealii::CellId, dealii::Tensor<1, 3, double> > globalCellIdSerCenters;
  unsigned int icell=0;
  for (; cellTria!=endcTria; ++cellTria)
  {
      globalCellIdToCellNumSerPrev[cellTria->id()] = icell;
      globalCellIdSerCenters[cellTria->id()] = cellTria->center();
      icell++ ;
  }
  //
  /// Owner processor ids of previous parallel mesh cells over a vector with ids corresponding to previous
  /// serial mesh cell numbers starting from 0 to total number of cells
  std::vector<unsigned int> cellParPrevToOwnerProcMap(numGlobalCellsPrev);

  /// map between previous serial mesh cell number to active cell iterator
  std::map<unsigned int,typename dealii::DoFHandler<3>::active_cell_iterator> cellNumSerPrevToParPrevCellIter;


  std::map<dealii::CellId, unsigned int> globalCellIdToCellNumParPrev;
  std::vector<unsigned int> ownerProcLocal(numGlobalCellsPrev,0);
  typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandlerUnmovedParPrev.begin_active();
  typename dealii::DoFHandler<3>::active_cell_iterator endc = dofHandlerUnmovedParPrev.end();
  dealii::Tensor<1, 3, double> center_diff;
  for(; cell!=endc; ++cell)
  {
     if (cell->is_locally_owned())
     {
	 globalCellIdToCellNumParPrev[cell->id()] = globalCellIdToCellNumSerPrev[cell->id()];

         AssertThrow((globalCellIdSerCenters[cell->id()]-cell->center()).norm_square()<1.0e-5,dealii::ExcMessage("Cell ids of serial and parallel triangulation are not same for the same physical cells."));
	 /*
	 typename dealii::parallel::distributed::Triangulation<3>::active_cell_iterator cellTriaTemp = triangulationSerPrev.begin_active(), endcTriaTemp = triangulationSerPrev.end();
	 for(; cellTriaTemp!=endcTriaTemp; ++cellTriaTemp)
	 {
	    center_diff = cellTriaTemp->center() - cellTria->center() ;
	    if (center_diff.norms_square() < 1.0E-5 )
	    {
		globalCellIdToCellNumSerPrev[cellTria->id()] = globalCellId[cellTriaTemp->id()] ;
		break;
	    }
	 }
	 */
         ownerProcLocal[globalCellIdToCellNumSerPrev[cell->id()]] = this_mpi_process;
	 cellNumSerPrevToParPrevCellIter[globalCellIdToCellNumSerPrev[cell->id()]]=cell;
      }
  }
  //
  MPI_Allreduce(&ownerProcLocal[0],
		&cellParPrevToOwnerProcMap[0],
		numGlobalCellsPrev,
		MPI_UNSIGNED,
		MPI_SUM,
		mpi_communicator);

  const unsigned int numLocallyOwnedCellsCurrent=triangulationParCurrent.n_locally_owned_active_cells();

  dealii::DoFHandler<3> dofHandlerUnmovedCurrent(triangulationParCurrent);
  dofHandlerUnmovedCurrent.distribute_dofs(FECurrent);
  std::map<dealii::types::global_dof_index, dealii::Point<3> > supportPointsUnmovedCurrent;
  dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3,3>(), dofHandlerUnmovedCurrent, supportPointsUnmovedCurrent);

  computing_timer.exit_section("interpolate:step1");

  ///////////////////////////////////////////////////////////
  //Step2: collect sending information from each processor///
  ///////////////////////////////////////////////////////////

  computing_timer.enter_section("interpolate:step2");
  const dealii::MappingQ1<3> mapping;

  //
  std::vector<std::vector<std::tuple<unsigned int,unsigned int,unsigned int> >> mappedGroup(numLocallyOwnedCellsCurrent);

  //<current mesh cell number<owning processor of mapped points of current mesh cell base indices in previous serial mesh<previous serial mesh cell numbers belonging to owning processor containing the mapped points>>>
  std::vector<std::vector<std::vector<unsigned int> >> mappedGroupSend0(numLocallyOwnedCellsCurrent,std::vector<std::vector<unsigned int> >(n_mpi_processes));

  //<current mesh cell no<owning processor of mapped points of current mesh cell base indices in previous serial mesh<size of mapped points inside each previous serial mesh cell number corresponding to mappedGroupSend0>>>
  std::vector<std::vector<std::vector<unsigned int> >> mappedGroupSend2(numLocallyOwnedCellsCurrent,std::vector<std::vector<unsigned int> >(n_mpi_processes));

  //<current mesh cell no<owning processor of mapped points of current mesh cell base indices in previous serial mesh<x,y,z reference coordinate component vectors<reference coordinate component value of each mapped dof of current mesh cell no in the owning processor>>>>
  std::vector<std::vector<std::vector<std::vector<double>>> > mappedGroupSend1(numLocallyOwnedCellsCurrent,std::vector<std::vector<std::vector<double>>>(n_mpi_processes,std::vector<std::vector<double>>(3)));

  //<current mesh cell no<owning processor of mapped points of current mesh cell base indices in previous serial mesh<send buffer sizes for mappedGroupSend0 and Send2>>
  std::vector<std::vector<unsigned int> > send0_buf_size(numLocallyOwnedCellsCurrent,std::vector<unsigned int>(n_mpi_processes,0));

  //<current mesh cell no<owning processor of mapped points of current mesh cell base indices in previous serial mesh<send buffer sizes for mappedGroupSend1>>
  std::vector<std::vector<unsigned int> > send1_buf_size(numLocallyOwnedCellsCurrent,std::vector<unsigned int>(n_mpi_processes,0));

  std::vector<unsigned int> countPointPerProc(n_mpi_processes);
  std::vector<bool> dofsTouched(dofHandlerUnmovedCurrent.n_dofs(),
				   false);
  cell = dofHandlerUnmovedCurrent.begin_active();endc = dofHandlerUnmovedCurrent.end();
  unsigned int iLocalCellCurrent=0;
  for (; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
	  std::map<dealii::CellId,std::vector<std::pair<std::vector<double>,unsigned int>>>  cellSerPrevToMappedDofData;

	  std::vector<dealii::types::global_dof_index> cell_dof_indices(dofs_per_cell_current);
	  cell->get_dof_indices(cell_dof_indices);

	  std::pair<typename dealii::parallel::distributed::Triangulation<3>::active_cell_iterator, dealii::Point<3> > mappedCellSerPrev;
	  unsigned int count=0;
	  for(unsigned int ibase = 0; ibase< base_indices_per_cell_current; ++ibase)
	  {
	         const dealii::types::global_dof_index globalDofId=cell_dof_indices[FECurrent.component_to_system_index(0,ibase)];

		 if (!dofsTouched[globalDofId])
                     dofsTouched[globalDofId]=true;
		 else
		     continue;

		 const dealii::Point<3> & p =supportPointsUnmovedCurrent[globalDofId];

		 if (count==0)
		 {
	            mappedCellSerPrev = dealii::GridTools::find_active_cell_around_point(mapping,triangulationSerPrev,p );
		 }
		 else
		 {
		   double dist = 1.0E+06 ;
		   try
		   {
		      dealii::Point<3> p_cell =  mapping.transform_real_to_unit_cell( mappedCellSerPrev.first, p ) ;
		      dist = dealii::GeometryInfo<3>::distance_to_unit_cell(p_cell);
		      if (dist < 1.0E-10)
			mappedCellSerPrev.second = p_cell ;
		   }
		   catch (dealii::MappingQ1<3>::ExcTransformationFailed)
		   {

		   }
		   if (dist > 1.0E-10)
		      mappedCellSerPrev = dealii::GridTools::find_active_cell_around_point(mapping, triangulationSerPrev, p) ;
		 }

		 const dealii::Point<3> pointTemp =dealii::GeometryInfo<3>::project_to_unit_cell(mappedCellSerPrev.second);

		 std::vector<double> mappedPoint(3);
		 mappedPoint[0] = pointTemp.operator()(0);
		 mappedPoint[1] = pointTemp.operator()(1);
		 mappedPoint[2] = pointTemp.operator()(2);

		 const std::pair<std::vector<double>,unsigned int> pairTemp = std::make_pair(mappedPoint,ibase);
		 cellSerPrevToMappedDofData[mappedCellSerPrev.first->id()].push_back(pairTemp);
		 count++;
	  }//base_indices_per_cell loop

	  std::fill(countPointPerProc.begin(),countPointPerProc.end(),0);
	  for(std::map<dealii::CellId,std::vector<std::pair<std::vector<double>,unsigned int> >>::iterator iter = cellSerPrevToMappedDofData.begin(); iter != cellSerPrevToMappedDofData.end(); ++iter)
	  {
	     const std::vector<std::pair<std::vector<double>,unsigned int> > & value = iter->second;
	     const dealii::CellId & key = iter->first;

             Assert(globalCellIdToCellNumSerPrev.find(key)!=globalCellIdToCellNumSerPrev.end(),dealii::ExcMessage("Invalid cellid key."));

             Assert(globalCellIdToCellNumSerPrev[key]<cellParPrevToOwnerProcMap.size(),dealii::ExcMessage("Invalid serial cell number."));

	     const unsigned int ownerProcIdMappedCellSerPrev =
		       cellParPrevToOwnerProcMap[globalCellIdToCellNumSerPrev[key]] ;
	     mappedGroupSend0[iLocalCellCurrent][ownerProcIdMappedCellSerPrev].push_back(globalCellIdToCellNumSerPrev[key]) ;
	     mappedGroupSend2[iLocalCellCurrent][ownerProcIdMappedCellSerPrev].push_back(value.size()) ;
	     send0_buf_size[iLocalCellCurrent][ownerProcIdMappedCellSerPrev]++;
	     //
	     for (unsigned int i=0; i<value.size(); ++i)
	     {
	       const std::vector<double> & mappedPoint = value[i].first ;
	       const unsigned int baseIndexId = value[i].second;
	       //
	       const std::tuple<unsigned int, unsigned int,unsigned int> tupleTemp2 = std::make_tuple(ownerProcIdMappedCellSerPrev,baseIndexId,countPointPerProc[ownerProcIdMappedCellSerPrev]);
	       mappedGroup[iLocalCellCurrent].push_back(tupleTemp2);

	       countPointPerProc[ownerProcIdMappedCellSerPrev] ++;
	       //
	       mappedGroupSend1[iLocalCellCurrent][ownerProcIdMappedCellSerPrev][0].push_back(mappedPoint[0]);
	       mappedGroupSend1[iLocalCellCurrent][ownerProcIdMappedCellSerPrev][1].push_back(mappedPoint[1]);
	       mappedGroupSend1[iLocalCellCurrent][ownerProcIdMappedCellSerPrev][2].push_back(mappedPoint[2]);

	       send1_buf_size[iLocalCellCurrent][ownerProcIdMappedCellSerPrev]++;
	     }//loop over all mapped dof points inside each serial prev cell
          }//loop over serial prev cells
	  iLocalCellCurrent++;
      }//locally_owned
  }//cell_loop
    //
  MPI_Barrier(mpi_communicator);
  computing_timer.exit_section("interpolate:step2");

  ////////////////////////////////////////////////////
  //Step3: Gather mapped points from all processors///
  ////////////////////////////////////////////////////

  computing_timer.enter_section("interpolate:step3");
  /// <sending processor<data size in d_recvData0 from sending processor> >
  std::vector<int> recv_size0(n_mpi_processes, 0);

  /// <sending processor<data size in d_recvData1 accumulated over all three components (d_recvData1[0],d_recvData1[1],d_recvData1[2]) from sending processor> >
  std::vector<int> recv_size1(n_mpi_processes, 0);

  std::vector<int> mpi_offsets0(n_mpi_processes, 0);

  std::vector<int> mpi_offsets1(n_mpi_processes, 0);

  /// <serial mesh cell numbers belonging to receiving processor which contain mapped points from all sending processors>
  std::vector<unsigned int> recvData0;

  /// <size of mapped point data attached to each entry of d_recvData0>
  std::vector<unsigned int> recvData2;

  /// <x,y,z component of reference coordinate<accumulation from all sending processors the reference coordinate component of mapped points in receiving processor from locally owned cells from that sending processor> >
  std::vector<std::vector<double>> recvData1(3);
     //
  for ( unsigned int proc = 0; proc < n_mpi_processes; ++proc)
  {
	unsigned int send_size1 = 0;
	unsigned int send_size0 = 0;
        std::vector<unsigned int> send_data0, send_data2;
        std::vector<std::vector<double>> send_data1(3);
	std::vector<double>  send_data, recvData;

	cell = dofHandlerUnmovedCurrent.begin_active();
	endc = dofHandlerUnmovedCurrent.end();
	iLocalCellCurrent=0;
	for (; cell!=endc; ++cell)
	 {
	  if (cell->is_locally_owned())
	  {
	     for (unsigned int iPoint = 0; iPoint < send1_buf_size[iLocalCellCurrent][proc]; ++iPoint)
	     {
		 //
		 send_data1[0].push_back(mappedGroupSend1[iLocalCellCurrent][proc][0][iPoint])  ;
		 send_data1[1].push_back(mappedGroupSend1[iLocalCellCurrent][proc][1][iPoint])  ;
		 send_data1[2].push_back(mappedGroupSend1[iLocalCellCurrent][proc][2][iPoint])  ;
	     }
	     send_size1 += send1_buf_size[iLocalCellCurrent][proc];
	     //
	     for (unsigned int i = 0; i < send0_buf_size[iLocalCellCurrent][proc]; ++i)
	     {
		 //
		 send_data0.push_back(mappedGroupSend0[iLocalCellCurrent][proc][i])  ;
		 send_data2.push_back(mappedGroupSend2[iLocalCellCurrent][proc][i])  ;
	     }
	     send_size0 += send0_buf_size[iLocalCellCurrent][proc];
	     //
	     iLocalCellCurrent++;
	   }//locally owned
	 }//cell loop
	 //
	 //
	 MPI_Gather(&send_size0,1,MPI_UNSIGNED, &(recv_size0[0]),1, MPI_UNSIGNED,proc,mpi_communicator);
	 MPI_Gather(&send_size1,1,MPI_UNSIGNED, &(recv_size1[0]),1, MPI_UNSIGNED,proc,mpi_communicator);
	 //
	 if (proc==this_mpi_process)
	 {
	    //
	    for(unsigned int i = 1; i < n_mpi_processes; i++)
	    {
	       mpi_offsets0[i] = recv_size0[i-1]+ mpi_offsets0[i-1];
	       mpi_offsets1[i] = recv_size1[i-1]+ mpi_offsets1[i-1];
	    }
	    //
	    const unsigned int recvDataSize0 = std::accumulate(&recv_size0[0], &recv_size0[n_mpi_processes], 0);
	    recvData0.resize(recvDataSize0,0);
	    recvData2.resize(recvDataSize0,0);
	    //
	    const unsigned int recvDataSize1 = std::accumulate(&recv_size1[0], &recv_size1[n_mpi_processes], 0);
	    recvData.resize(recvDataSize1,0.0) ;
	    recvData1.resize(3,std::vector<double>(recvDataSize1,0.0));
	 }
	 //
	 MPI_Gatherv(&(send_data0[0]),send_size0,MPI_UNSIGNED, &(recvData0[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_UNSIGNED,proc,mpi_communicator);
	 MPI_Gatherv(&(send_data2[0]),send_size0,MPI_UNSIGNED, &(recvData2[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_UNSIGNED,proc,mpi_communicator);
	 for (unsigned int ipol=0; ipol<3; ++ipol)
	 {
	      send_data = send_data1[ipol] ;
	      MPI_Gatherv(&(send_data[0]),send_size1,MPI_DOUBLE, &(recvData[0]),&(recv_size1[0]), &(mpi_offsets1[0]),  MPI_DOUBLE,proc,mpi_communicator);
	      if (proc==this_mpi_process)
		  recvData1[ipol] = recvData ;
	 }
  }//n_mpi_processes loop

  MPI_Barrier(mpi_communicator);
  mappedGroupSend0.clear();
  mappedGroupSend1.clear();
  mappedGroupSend2.clear();
  computing_timer.exit_section("interpolate:step3");

  ///////////////////////////////////////////////////////////
  //Step4: Interpolate previous fields to all mapped points//
  ///////////////////////////////////////////////////////////

  computing_timer.enter_section("interpolate:step4");
  AssertThrow(fieldsPreviousMesh.size()==fieldsCurrentMesh.size(),dealii::ExcMessage("Size of fieldsPreviousMesh and fieldsCurrentMesh are no the same."));
  const unsigned int fieldsBlockSize=fieldsPreviousMesh.size();

  for(unsigned int ifield = 0; ifield < fieldsBlockSize; ++ifield)
    fieldsPreviousMesh[ifield]->update_ghost_values();

  const unsigned int totPoints = recvData1[0].size() ;
  std::vector<double> fieldsValuesSendData(totPoints*fieldsBlockSize*fe_components, 0.0);
  unsigned int numPointsDone = 0, numGroupsDone = 0 ;
  for ( unsigned int proc = 0; proc < n_mpi_processes; ++proc)
  {
     //
     for ( unsigned int iGroup = 0; iGroup < recv_size0[proc] ; ++iGroup )
     {
        const unsigned int serCellNumOfMappedPointsGroup= recvData0[numGroupsDone + iGroup];
	const unsigned int numPointsInGroup = recvData2[numGroupsDone + iGroup];

	std::vector<dealii::Point<3>> pointsList(numPointsInGroup);
	for (unsigned int ipoint=0; ipoint<numPointsInGroup; ++ipoint)
	{
          const double px = recvData1[0][numPointsDone+ipoint];
          const double py = recvData1[1][numPointsDone+ipoint];
          const double pz = recvData1[2][numPointsDone+ipoint];
	  //
          const dealii::Point<3> pointTemp(px, py, pz) ;
          pointsList[ipoint] = pointTemp;
	} // loop on points
        //
        const dealii::Quadrature<3> quadRule(pointsList);
        dealii::FEValues<3> feValues(mapping,FEPrev, quadRule, dealii::update_values);

        AssertThrow(cellNumSerPrevToParPrevCellIter[serCellNumOfMappedPointsGroup]->is_locally_owned(),
                    dealii::ExcMessage("Cell not available here"));

        feValues.reinit(cellNumSerPrevToParPrevCellIter[serCellNumOfMappedPointsGroup]);

	std::vector<double> tempInterpolatedField1Comp(numPointsInGroup);
        std::vector<dealii::Vector<double> > tempInterpolatedField(numPointsInGroup,dealii::Vector<double>(fe_components));

	for(unsigned int ifield = 0; ifield < fieldsBlockSize; ++ifield)
	{
	   if (fe_components==1)
	   {
	       feValues.get_function_values(*(fieldsPreviousMesh[ifield]), tempInterpolatedField1Comp);
	       for (unsigned int ipoint=0; ipoint<numPointsInGroup; ipoint++)
	          fieldsValuesSendData[numPointsDone*fieldsBlockSize
				      +ipoint*fieldsBlockSize
				      +ifield]
				      =tempInterpolatedField1Comp[ipoint];
	   }
	   else
	   {

	       feValues.get_function_values(*(fieldsPreviousMesh[ifield]), tempInterpolatedField);
	       for (unsigned int ipoint=0; ipoint<numPointsInGroup; ipoint++)
		   for (unsigned int icomp=0; icomp<fe_components; icomp++)
	             fieldsValuesSendData[numPointsDone*fieldsBlockSize*fe_components
				          +ipoint*fieldsBlockSize*fe_components
				          +ifield*fe_components
					  +icomp]
				          =tempInterpolatedField[ipoint][icomp];

	   }
	}//field loop
        numPointsDone += numPointsInGroup ;
     }// loop on group
     numGroupsDone += recv_size0[proc] ;
   }// loop on proc
   computing_timer.exit_section("interpolate:step4");

   ///////////////////////////////////////////////////////////////
   //Step5: scatter interpolated data back to sending processors//
   ///////////////////////////////////////////////////////////////

   computing_timer.enter_section("interpolate:step5");

   std::vector<std::vector<std::vector<double>>> fieldsValuesRecvData(numLocallyOwnedCellsCurrent,std::vector<std::vector<double>>(n_mpi_processes));
   std::vector<int> fieldsValuesRecvSize(n_mpi_processes,0);
   cell = dofHandlerUnmovedCurrent.begin_active();
   endc = dofHandlerUnmovedCurrent.end();
   iLocalCellCurrent=0;
   for (; cell!=endc; ++cell)
	if (cell->is_locally_owned())
	{
	  for (unsigned int proc=0; proc<n_mpi_processes; ++proc)
	  {
	       fieldsValuesRecvSize[proc] += fieldsBlockSize*fe_components*send1_buf_size[iLocalCellCurrent][proc] ;
	       fieldsValuesRecvData[iLocalCellCurrent][proc].resize(fieldsBlockSize*fe_components*send1_buf_size[iLocalCellCurrent][proc]) ;
	  }//procs loop
	  iLocalCellCurrent++;
	}//locally owned cell loop

   std::vector<int> mpiOffsetsFieldsValuesSend(n_mpi_processes);
   std::vector<int> fieldsValuesSendSize(n_mpi_processes);
   for(unsigned int proc = 0; proc < n_mpi_processes; proc++)
   {
       fieldsValuesSendSize[proc] = fieldsBlockSize*fe_components*recv_size1[proc] ;
       mpiOffsetsFieldsValuesSend[proc] = fieldsBlockSize*fe_components*mpi_offsets1[proc] ;
   }

   for (unsigned int sendProc = 0; sendProc<n_mpi_processes; ++sendProc)
   {
	std::vector<double> recvData(fieldsValuesRecvSize[sendProc]);

	MPI_Scatterv(&(fieldsValuesSendData[0]),&(fieldsValuesSendSize[0]), &(mpiOffsetsFieldsValuesSend[0]), MPI_DOUBLE, &(recvData[0]), fieldsValuesRecvSize[sendProc], MPI_DOUBLE,sendProc,mpi_communicator);

	unsigned int offset = 0;
	iLocalCellCurrent=0;
        cell = dofHandlerUnmovedCurrent.begin_active();
        endc = dofHandlerUnmovedCurrent.end();
	for (; cell!=endc; ++cell)
	  if (cell->is_locally_owned())
	  {
		for ( unsigned int i=0; i<fieldsValuesRecvData[iLocalCellCurrent][sendProc].size(); ++i)
		     fieldsValuesRecvData[iLocalCellCurrent][sendProc][i] = recvData[offset + i];
		offset += fieldsValuesRecvData[iLocalCellCurrent][sendProc].size();
		iLocalCellCurrent++;
	  }//locally owned cell loop
   }//loop on proc
   computing_timer.exit_section("interpolate:step5");

   ////////////////////////////////////////////////////////////////////////////////////////////
   //Step6: set values on fieldsCurrentMesh using interpolated data received after scattering//
   ////////////////////////////////////////////////////////////////////////////////////////////

   computing_timer.enter_section("interpolate:step6");
   const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner
                   =fieldsCurrentMesh[0]->get_partitioner();

   cell = dofHandlerUnmovedCurrent.begin_active();endc = dofHandlerUnmovedCurrent.end();
   iLocalCellCurrent=0;
   for (; cell!=endc; ++cell)
	if (cell->is_locally_owned())
	{
	      std::vector<dealii::types::global_dof_index> cell_dof_indices(dofs_per_cell_current);
	      cell->get_dof_indices(cell_dof_indices);

	      for (unsigned int i=0; i<mappedGroup[iLocalCellCurrent].size(); i++)
	      {
		  const unsigned int proc = std::get<0>(mappedGroup[iLocalCellCurrent][i]);
		  const unsigned int baseIndexId = std::get<1>(mappedGroup[iLocalCellCurrent][i]);
		  const unsigned int pointCount = std::get<2>(mappedGroup[iLocalCellCurrent][i]);
		  for(unsigned int ifield = 0; ifield < fieldsBlockSize; ++ifield)
		     for (unsigned int icomp=0; icomp<fe_components; icomp++)
		     {
			   const dealii::types::global_dof_index globalDofId=cell_dof_indices[FECurrent.component_to_system_index(icomp,baseIndexId)];
			   if (partitioner->in_local_range(globalDofId))
			        (*(fieldsCurrentMesh[ifield])).local_element(partitioner->global_to_local(globalDofId))=fieldsValuesRecvData[iLocalCellCurrent][proc][
					                pointCount*fieldsBlockSize*fe_components
				                        +ifield*fe_components
					                +icomp];
		     }//loop over components
	      }//loop over mapped points originating from iLocalCellCurrent
	      iLocalCellCurrent++;
        }//locally owned loop over cells

  if (constraintsCurrentPtr!=NULL)
     for(unsigned int ifield = 0; ifield < fieldsBlockSize; ++ifield)
        constraintsCurrentPtr->distribute(*(fieldsCurrentMesh[ifield]));

  computing_timer.exit_section("interpolate:step6");
}

}
}
