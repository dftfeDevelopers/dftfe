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
//================================================================================================================================================
//================================================================================================================================================
//    This is the source file for generating and communicating mapping tables between real space points and their symmetry transformed points.
//	            Only relevant for calculations using multiple k-points and when USE GROUP SYMMETRY = true
//
//                                              Author : Krishnendu Ghosh, krisg@umich.edu
//
//================================================================================================================================================
//================================================================================================================================================
//
#include "../../include/dftParameters.h"
#include "../../include/symmetry.h"
#include "../../include/dft.h"
#include "symmetrizeRho.cc"
//
namespace dftfe {
//================================================================================================================================================
//							Class constructor
//================================================================================================================================================
template<unsigned int FEOrder>
symmetryClass<FEOrder>::symmetryClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica,const MPI_Comm &_interpoolcomm):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  mpi_communicator (mpi_comm_replica),
  interpoolcomm (_interpoolcomm),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}
//================================================================================================================================================
//					Wiping out mapping tables; needed between relaxation steps
//================================================================================================================================================
template<unsigned int FEOrder>
void symmetryClass<FEOrder>::clearMaps()
{
 mappedGroup.clear() ;
 mappedGroupSend0.clear() ;
 mappedGroupSend1.clear() ;
 mappedGroupSend2.clear() ;
 mappedGroupRecvd0.clear() ;
 mappedGroupRecvd2.clear() ;
 mappedGroupRecvd1.clear() ;
 send_buf_size.clear() ;
 recv_buf_size.clear() ;
 rhoRecvd.clear() ;
 groupOffsets.clear() ;
 if (dftParameters::xc_id==4)
 gradRhoRecvd.clear() ;
}
//================================================================================================================================================
//================================================================================================================================================
//			     The following is the main driver routine to generate and communicate mapping tables
//================================================================================================================================================
//================================================================================================================================================
template<unsigned int FEOrder>
void symmetryClass<FEOrder>::initSymmetry()
{
  //
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (dftPtr->FEEigen, quadrature, update_values | update_gradients| update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
  Point<3> p, ptemp, p0 ;
  MappingQ1<3> mapping;
  char buffer[100];
  //
  std::pair<typename parallel::distributed::Triangulation<3>::active_cell_iterator, Point<3> > mapped_cell;
  std::tuple<int, std::vector<double>, int> tupleTemp ;
  std::tuple< int, int, int> tupleTemp2 ;
  std::map<CellId,int> groupId  ;
  std::vector<double> mappedPoint(3) ;
  std::vector<int> countGroupPerProc(dftPtr->n_mpi_processes), countPointPerProc(dftPtr->n_mpi_processes)  ;
  std::vector<std::vector<int>> countPointsPerGroupPerProc(dftPtr->n_mpi_processes) ;
  std::vector<std::vector<int>> tailofGroup(dftPtr->n_mpi_processes);
  //
  unsigned int count = 0, cell_id=0, ownerProcId ;
  unsigned int mappedPointId ;
  std::map<CellId, int> globalCellId_parallel ;
  //
  clearMaps() ;
//================================================================================================================================================
//							Allocate memory for the mapping tables
//================================================================================================================================================
  mappedGroup.resize(numSymm) ;
  mappedGroupSend0.resize(numSymm) ;
  mappedGroupSend1.resize(numSymm) ;
  mappedGroupSend2.resize(numSymm) ;
  mappedGroupRecvd0.resize(numSymm) ;
  mappedGroupRecvd2.resize(numSymm) ;
  mappedGroupRecvd1.resize(numSymm) ;
  send_buf_size.resize(numSymm) ;
  recv_buf_size.resize(numSymm) ;
  rhoRecvd.resize(numSymm) ;
  groupOffsets.resize(numSymm) ;
  if (dftParameters::xc_id==4)
    gradRhoRecvd.resize(numSymm) ;
  //
  const parallel::distributed::Triangulation<3> & triangulationSer = (dftPtr->d_mesh).getSerialMeshUnmoved();
  typename parallel::distributed::Triangulation<3>::active_cell_iterator cellTemp = triangulationSer.begin_active(), endcTemp = triangulationSer.end();
  for (; cellTemp!=endcTemp; ++cellTemp)
    {
      globalCellId[cellTemp->id()] = cell_id;
      cell_id++ ;
    }
  //
  ownerProcGlobal.resize(cell_id) ;
  std::vector<int> ownerProc(cell_id,0);
  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm)
  {
    mappedGroup[iSymm] = std::vector<std::vector<std::tuple<int, int, int> >>(cell_id);
    mappedGroupSend0[iSymm]=std::vector<std::vector<std::vector<int> >>(cell_id);
    mappedGroupSend2[iSymm]=std::vector<std::vector<std::vector<int> >>(cell_id);
    mappedGroupSend1[iSymm]=std::vector<std::vector<std::vector<std::vector<double>>> >(cell_id);
    mappedGroupRecvd0[iSymm]=std::vector<std::vector<int> >(cell_id);
    mappedGroupRecvd2[iSymm]=std::vector<std::vector<int> >(cell_id);
    mappedGroupRecvd1[iSymm]=std::vector<std::vector<std::vector<double>> >(cell_id);
    send_buf_size[iSymm]=std::vector<std::vector<std::vector<int>> >(cell_id);
    recv_buf_size[iSymm]=std::vector<std::vector<std::vector<int>> >(cell_id);
    rhoRecvd[iSymm]=std::vector<std::vector<std::vector<double>> >(cell_id);
    groupOffsets[iSymm]=std::vector<std::vector<std::vector<int>> >(cell_id);
    if (dftParameters::xc_id==4)
    gradRhoRecvd[iSymm]=std::vector<std::vector<std::vector<double>> >(cell_id);
  }
//================================================================================================================================================
//					     Create local and global maps to locate cells on their hosting processors
//================================================================================================================================================
  typename DoFHandler<3>::active_cell_iterator cell = (dftPtr->dofHandlerEigen).begin_active(), endc = (dftPtr->dofHandlerEigen).end();
  dealii::Tensor<1, 3, double> center_diff ;
  for(; cell!=endc; ++cell)
  {
     if (cell->is_locally_owned())
     {

	 cellTemp = triangulationSer.begin_active(), endcTemp = triangulationSer.end();
	 for(; cellTemp!=endcTemp; ++cellTemp) {
	    center_diff = cellTemp->center() - cell->center() ;
	    double pnorm = center_diff[0]*center_diff[0] + center_diff[1]*center_diff[1] + center_diff[2]*center_diff[2] ;
	    if (pnorm < 1.0E-5 ) {
		globalCellId_parallel[cell->id()] = globalCellId[cellTemp->id()] ;
		break;
	     }
	}
         dealIICellId [globalCellId_parallel[cell->id()]] = cell;
         ownerProc[globalCellId_parallel[cell->id()]] = this_mpi_process ;
      }
  }
  //
  MPI_Allreduce(&ownerProc[0],
		&ownerProcGlobal[0],
		cell_id,
		MPI_INT,
		MPI_SUM,
		mpi_communicator) ;
//================================================================================================================================================
//			Now enter each local cell to apply each of the symmetry operations on the quad points relevant to the cell.
//			Then find out which cell and processor the transformed point belongs to.
//			Next create maps of points based on symmetry operation, cell address, and processor id.  
//================================================================================================================================================ 
  cell = (dftPtr->dofHandlerEigen).begin_active(); endc = (dftPtr->dofHandlerEigen).end();
  for (; cell!=endc; ++cell)
    {
    if (cell->is_locally_owned()) 
      {
        for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
          {
	   mappedGroup[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::tuple<int, int, int> >(num_quad_points);
	   mappedGroupRecvd1[iSymm][globalCellId_parallel[cell->id()]]=std::vector<std::vector<double>>(3);
	   rhoRecvd[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<double>>(dftPtr->n_mpi_processes) ;
	  }
	  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
              {
	      count = 0;
	      std::fill(countGroupPerProc.begin(),countGroupPerProc.end(),0);
	      //
	      send_buf_size[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<int>>(dftPtr->n_mpi_processes);
	      //
	      mappedGroupSend0[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<int> >(dftPtr->n_mpi_processes);
	      mappedGroupSend2[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<int> >(dftPtr->n_mpi_processes);
	      mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<std::vector<double>> >(dftPtr->n_mpi_processes);
	      //
	      for(int i = 0; i < dftPtr->n_mpi_processes; ++i) 
		{
		 send_buf_size[iSymm][globalCellId_parallel[cell->id()]][i] = std::vector<int>(3, 0);
		 mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][i] = std::vector<std::vector<double>>(3);
		}
              recv_buf_size[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<int>>(3);
              recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][0] = std::vector<int>(dftPtr->n_mpi_processes);
	      recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][1] = std::vector<int>(dftPtr->n_mpi_processes);
	      recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][2] = std::vector<int>(dftPtr->n_mpi_processes);
	      //
	      groupOffsets[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<int>>(3);
	      groupOffsets[iSymm][globalCellId_parallel[cell->id()]][0] = std::vector<int>(dftPtr->n_mpi_processes);
	      groupOffsets[iSymm][globalCellId_parallel[cell->id()]][1] = std::vector<int>(dftPtr->n_mpi_processes);
	      groupOffsets[iSymm][globalCellId_parallel[cell->id()]][2] = std::vector<int>(dftPtr->n_mpi_processes);
              //
	      fe_values.reinit (cell);
              for(unsigned int q_point=0; q_point<num_quad_points; ++q_point) 
		 {
                 p = fe_values.quadrature_point(q_point) ;
	         p0 = crys2cart(p,-1) ;
	         //
                 ptemp[0] = p0[0]*symmMat[iSymm][0][0] + p0[1]*symmMat[iSymm][0][1] + p0[2]*symmMat[iSymm][0][2] ;
                 ptemp[1] = p0[0]*symmMat[iSymm][1][0] + p0[1]*symmMat[iSymm][1][1] + p0[2]*symmMat[iSymm][1][2] ;
                 ptemp[2] = p0[0]*symmMat[iSymm][2][0] + p0[1]*symmMat[iSymm][2][1] + p0[2]*symmMat[iSymm][2][2] ;
		 //
		 ptemp[0] = ptemp[0] + translation[iSymm][0] ;
		 ptemp[1] = ptemp[1] + translation[iSymm][1] ;
		 ptemp[2] = ptemp[2] + translation[iSymm][2] ;
		 //
		 for (unsigned int i=0; i<3; ++i)
		 {
		    while (ptemp[i] > 0.5)
			ptemp[i] = ptemp[i] - 1.0 ;
		    while (ptemp[i] < -0.5)
			ptemp[i] = ptemp[i] + 1.0 ;
		 }
                 p = crys2cart(ptemp,1) ;
		 //
		 if (q_point==0)
		     mapped_cell = GridTools::find_active_cell_around_point ( mapping, triangulationSer, p ) ;
		 else 
		    {
		     double dist = 1.0E+06 ;
		     try 
		       {
		       Point<3> p_cell =  mapping.transform_real_to_unit_cell( mapped_cell.first, p ) ;
		       dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);
		       if (dist < 1.0E-10)
			  mapped_cell.second = p_cell ;
		       }
		    catch (MappingQ1<3>::ExcTransformationFailed)
		       {

		       } 
		    if (dist > 1.0E-10)
		       mapped_cell = GridTools::find_active_cell_around_point ( mapping, triangulationSer, p ) ;
		     }
		 Point<3> pointTemp = mapped_cell.second;
		 //
		 mappedPoint[0] = pointTemp.operator()(0);
		 mappedPoint[1] = pointTemp.operator()(1);
		 mappedPoint[2] = pointTemp.operator()(2);
		 //
		 ownerProcId = ownerProcGlobal[globalCellId[mapped_cell.first->id()]] ;
		 //
		 tupleTemp = std::make_tuple(ownerProcId,mappedPoint,q_point);
		 cellMapTable[mapped_cell.first->id()].push_back(tupleTemp) ;
		 //
		 //
		 send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][1] = send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][1] + 1;
		 send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][2] = send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][2] + 1;
		 //
	     }
	     std::fill(countPointPerProc.begin(),countPointPerProc.end(),0);
	     for(std::map<CellId,std::vector<std::tuple<int, std::vector<double>, int> >>::iterator iter = cellMapTable.begin(); iter != cellMapTable.end(); ++iter) 
		 {
		 std::vector<std::tuple<int, std::vector<double>, int> > value = iter->second;
		 CellId key = iter->first;
		 ownerProcId = ownerProcGlobal[globalCellId[key]] ;
		 mappedGroupSend0[iSymm][globalCellId_parallel[cell->id()]][ownerProcId].push_back(globalCellId[key]) ;
		 mappedGroupSend2[iSymm][globalCellId_parallel[cell->id()]][ownerProcId].push_back(value.size()) ;
		 send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][0] = send_buf_size[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][0] + 1;
		 //
		 for (unsigned int i=0; i<value.size(); ++i) 
		    {
		    mappedPoint = std::get<1>(value[i]) ;
                    int q_point = std::get<2>(value[i]) ;
		    //
                    tupleTemp2 = std::make_tuple(ownerProcId, 0,countPointPerProc[ownerProcId]);
		    mappedGroup[iSymm][globalCellId_parallel[cell->id()]][q_point] = tupleTemp2 ;
		    countPointPerProc[ownerProcId] += 1 ;
		    //
		    mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][0].push_back(mappedPoint[0]) ;
		    mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][1].push_back(mappedPoint[1]) ;
		    mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][ownerProcId][2].push_back(mappedPoint[2]) ;
		    }
              }
	      cellMapTable.clear() ;
           }  // symmetry loop
        }  // is cell locally owned condition
     }  // cell loop
  //
  MPI_Barrier(mpi_communicator) ;
//================================================================================================================================================
//			      Now first prepare the flattened sending and receiving vectors and then MPI gather.
//		     The essential idea here is that each processor collects the transformed points from all other processors.
//			 In symmetrizeRho.cc each processor locally computes density on its transformed points and 
//			     then scatters them back to the processors from which the points came from. 
//================================================================================================================================================ 
  int recvDataSize0=0, recvDataSize1=0, send_size0, send_size1, send_size2 ;
  std::vector<int> send_data0, send_data2, send_data3;
  std::vector<std::vector<double>> send_data1;
  std::vector<double>  send_data, recvdData ;
  mpi_offsets0.resize(dftPtr->n_mpi_processes, 0) ;
  mpi_offsets1.resize(dftPtr->n_mpi_processes, 0) ;
  mpiGrad_offsets1.resize(dftPtr->n_mpi_processes, 0) ;
  recv_size0.resize(dftPtr->n_mpi_processes, 0) ;
  recv_size1.resize(dftPtr->n_mpi_processes, 0) ;
  recvGrad_size1.resize(dftPtr->n_mpi_processes, 0) ;
  recvdData1.resize(3);
  send_data1.resize(3);
  //
  for ( unsigned int proc = 0; proc < dftPtr->n_mpi_processes; ++proc) 
     {
     send_size1 = 0 ; send_size0 = 0 ;
     cell = (dftPtr->dofHandlerEigen).begin_active();
     for (; cell!=endc; ++cell)
    	{
        if (cell->is_locally_owned()) 
	   {
	   for (unsigned int iSymm = 0; iSymm < numSymm; iSymm++)
	      {
	      for (unsigned int iPoint = 0; iPoint < send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1]; ++iPoint) 
		 {
		 //
	         send_data1[0].push_back(mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][proc][0][iPoint])  ;
		 send_data1[1].push_back(mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][proc][1][iPoint])  ;
		 send_data1[2].push_back(mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]][proc][2][iPoint])  ;
		 }
	      send_size1 += send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1] ;
	      //
	      for (unsigned int i = 0; i < send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][0]; ++i) 
		 {
	         send_data0.push_back(mappedGroupSend0[iSymm][globalCellId_parallel[cell->id()]][proc][i])  ;
		 send_data2.push_back(mappedGroupSend2[iSymm][globalCellId_parallel[cell->id()]][proc][i])  ;
		 send_data3.push_back(iSymm)  ;
		 }
	      send_size0 += send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][0] ;
	      //
	      rhoRecvd[iSymm][globalCellId_parallel[cell->id()]][proc].resize((1+dftParameters::spinPolarized)*send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1]) ; // to be used later to recv symmetrized rho
	      }
	   }
	}
     //
     //
     MPI_Gather(&send_size0,1,MPI_INT, &(recv_size0[0]),1, MPI_INT,proc,mpi_communicator);
     MPI_Gather(&send_size1,1,MPI_INT, &(recv_size1[0]),1, MPI_INT,proc,mpi_communicator);
     //
     if (proc==this_mpi_process)
	{
	for(int i = 1; i < dftPtr->n_mpi_processes; i++) 
	   {
           mpi_offsets0[i] = recv_size0[i-1]+ mpi_offsets0[i-1];
	   mpi_offsets1[i] = recv_size1[i-1]+ mpi_offsets1[i-1];
	   }
	//
        recvDataSize0 = std::accumulate(&recv_size0[0], &recv_size0[dftPtr->n_mpi_processes], 0);
        recvdData0.resize(recvDataSize0,0);
	recvdData2.resize(recvDataSize0,0);
	recvdData3.resize(recvDataSize0,0);
        //
        recvDataSize1 = std::accumulate(&recv_size1[0], &recv_size1[dftPtr->n_mpi_processes], 0);
	recvdData.resize(recvDataSize1,0.0) ;
	for (unsigned int ipol=0; ipol<3; ++ipol)
           recvdData1[ipol].resize(recvDataSize1,0.0) ;
	}
	//
     MPI_Gatherv(&(send_data0[0]),send_size0,MPI_INT, &(recvdData0[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);
     MPI_Gatherv(&(send_data2[0]),send_size0,MPI_INT, &(recvdData2[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);
     MPI_Gatherv(&(send_data3[0]),send_size0,MPI_INT, &(recvdData3[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);
     for (unsigned int ipol=0; ipol<3; ++ipol) 
	{
	send_data = send_data1[ipol] ;
	MPI_Gatherv(&(send_data[0]),send_size1,MPI_DOUBLE, &(recvdData[0]),&(recv_size1[0]), &(mpi_offsets1[0]),  MPI_DOUBLE,proc,mpi_communicator);
	if (proc==this_mpi_process)
	   recvdData1[ipol] = recvdData ;
	}
     send_data0.clear() ; send_data.clear() ; recvdData.clear() ;
     send_data1[0].clear(); send_data1[1].clear(); send_data1[2].clear();
     send_data2.clear() ;
     send_data3.clear();
     }
  //
  MPI_Barrier(mpi_communicator) ;
//================================================================================================================================================
//			     Prepare the receiving vectors on which computed density is to be received in symmetrizeRho.cc
//		We do this here instead of doing in symmetrizeRho.cc, because symmetrizeRho.cc is to be called during each SCF iteration
//							So this better be a one time cost
//================================================================================================================================================ 
  cell = (dftPtr->dofHandlerEigen).begin_active();
  totPoints = 0;
  recv_size.resize(dftPtr->n_mpi_processes,0) ;
  //
  for (; cell!=endc; ++cell)
     {
     if (cell->is_locally_owned()) 
	{
        for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm)
	   {
	   rhoRecvd[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<double>>(dftPtr->n_mpi_processes) ;
	   for (unsigned int proc=0; proc<dftPtr->n_mpi_processes; ++proc)
	      {
	      recv_size[proc] = recv_size[proc] + (1+dftParameters::spinPolarized)*send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1] ;
	      rhoRecvd[iSymm][globalCellId_parallel[cell->id()]][proc].resize((1+dftParameters::spinPolarized)*send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1]) ;
	      }
	   }
        }
     }
  //
  for(int i = 0; i < dftPtr->n_mpi_processes; i++) 
     {
     recv_size1[i] = (1 + dftParameters::spinPolarized)*recv_size1[i] ;
     mpi_offsets1[i] = (1 + dftParameters::spinPolarized)*mpi_offsets1[i] ;
     }
  //
  if (dftParameters::xc_id==4)
     {
     cell = (dftPtr->dofHandlerEigen).begin_active();
     for(int i = 0; i < dftPtr->n_mpi_processes; i++) 
        {
        recvGrad_size1[i] = 3*recv_size1[i] ;
        mpiGrad_offsets1[i] = 3*mpi_offsets1[i] ;
        }
     //
     for (; cell!=endc; ++cell)
        {
        if (cell->is_locally_owned()) 
           {
           for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm)
	      {
	      gradRhoRecvd[iSymm][globalCellId_parallel[cell->id()]] = std::vector<std::vector<double>>(dftPtr->n_mpi_processes) ;
	      for (unsigned int proc=0; proc<dftPtr->n_mpi_processes; ++proc)
	         gradRhoRecvd[iSymm][globalCellId_parallel[cell->id()]][proc].resize((1+dftParameters::spinPolarized)*3*send_buf_size[iSymm][globalCellId_parallel[cell->id()]][proc][1]) ;
	      }
           }
        }
     }
  //
}
//================================================================================================================================================
//================================================================================================================================================
//			           Just a quick snippet to go back and forth between crystal and cartesian coordinates.
//			              flag==1 takes crystal to cartesian and flag==-1 does the other way around.
//================================================================================================================================================
//================================================================================================================================================ 
template<unsigned int FEOrder>
Point<3> symmetryClass<FEOrder>::crys2cart(Point<3> p, int flag)
{
  Point<3> ptemp ;
  if (flag==1){
    ptemp[0] = p[0]*(dftPtr->d_domainBoundingVectors)[0][0] + p[1]*(dftPtr->d_domainBoundingVectors)[1][0] + p[2]*(dftPtr->d_domainBoundingVectors)[2][0] ;
    ptemp[1] = p[0]*(dftPtr->d_domainBoundingVectors)[0][1] + p[1]*(dftPtr->d_domainBoundingVectors)[1][1] + p[2]*(dftPtr->d_domainBoundingVectors)[2][1] ;
    ptemp[2] = p[0]*(dftPtr->d_domainBoundingVectors)[0][2] + p[1]*(dftPtr->d_domainBoundingVectors)[1][2] + p[2]*(dftPtr->d_domainBoundingVectors)[2][2] ;
  }
  if (flag==-1){
    ptemp[0] = p[0]*(dftPtr->d_reciprocalLatticeVectors)[0][0] + p[1]*(dftPtr->d_reciprocalLatticeVectors)[0][1] + p[2]*(dftPtr->d_reciprocalLatticeVectors)[0][2] ;
    ptemp[1] = p[0]*(dftPtr->d_reciprocalLatticeVectors)[1][0] + p[1]*(dftPtr->d_reciprocalLatticeVectors)[1][1] + p[2]*(dftPtr->d_reciprocalLatticeVectors)[1][2] ;
    ptemp[2] = p[0]*(dftPtr->d_reciprocalLatticeVectors)[2][0] + p[1]*(dftPtr->d_reciprocalLatticeVectors)[2][1] + p[2]*(dftPtr->d_reciprocalLatticeVectors)[2][2] ;
    ptemp = 1.0 / (2.0*M_PI) * ptemp ;
  }

  return ptemp;
}
//================================================================================================================================================
template class symmetryClass<1>;
template class symmetryClass<2>;
template class symmetryClass<3>;
template class symmetryClass<4>;
template class symmetryClass<5>;
template class symmetryClass<6>;
template class symmetryClass<7>;
template class symmetryClass<8>;
template class symmetryClass<9>;
template class symmetryClass<10>;
template class symmetryClass<11>;
template class symmetryClass<12>;
//=================================================================================================================================================
}
