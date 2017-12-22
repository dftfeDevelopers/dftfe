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
// @author Krishnendu Ghosh (2017)
//

//source file for initializing space group symmetries, generating and communicating mapping tables

template<unsigned int FEOrder>
void dftClass<FEOrder>::initSymmetry()
{
//
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FEEigen, quadrature, update_values | update_gradients| update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
  Point<3> p, ptemp, p0 ;
  MappingQ1<3> mapping;
  char buffer[100];
 //
 typename DoFHandler<3>::active_cell_iterator cell = dofHandlerEigen.begin_active(), endc = dofHandlerEigen.end();
 std::pair<typename DoFHandler<3>::active_cell_iterator, Point<3> > mapped_cell;
 std::tuple<int, std::vector<double>, int> tupleTemp ;
 std::tuple< int, int, int> tupleTemp2 ;
 std::map<CellId,int> groupId  ;
 std::vector<double> mappedPoint(3) ;
 std::vector<int> countGroupPerProc(n_mpi_processes), countPointPerProc(n_mpi_processes)  ;
 std::vector<std::vector<int>> countPointsPerGroupPerProc(n_mpi_processes) ;
 std::vector<std::vector<int>> tailofGroup(n_mpi_processes);
 //
 unsigned int n_cell= 0, count = 0, cell_id=0, ownerProcId ;
 typename DoFHandler<3>::active_cell_iterator cellTemp = dofHandlerEigen.begin_active();
  for (; cellTemp!=endc; ++cellTemp) 
      n_cell++ ;
 std::vector<int> ownerProc(n_cell,0);
 unsigned int mappedPointId ;
 //
 ownerProcGlobal.resize(n_cell) ;
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
 if (xc_id==4)
 gradRhoRecvd.resize(numSymm) ;
 //
 for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm){
    mappedGroup[iSymm] = std::vector<std::vector<std::tuple<int, int, int> >>(triangulation.n_global_active_cells());
    mappedGroupSend0[iSymm]=std::vector<std::vector<std::vector<int> >>(triangulation.n_global_active_cells());
    mappedGroupSend2[iSymm]=std::vector<std::vector<std::vector<int> >>(triangulation.n_global_active_cells());
    mappedGroupSend1[iSymm]=std::vector<std::vector<std::vector<std::vector<double>>> >(triangulation.n_global_active_cells());
    mappedGroupRecvd0[iSymm]=std::vector<std::vector<int> >(triangulation.n_global_active_cells());
    mappedGroupRecvd2[iSymm]=std::vector<std::vector<int> >(triangulation.n_global_active_cells());
    mappedGroupRecvd1[iSymm]=std::vector<std::vector<std::vector<double>> >(triangulation.n_global_active_cells());
    send_buf_size[iSymm]=std::vector<std::vector<std::vector<int>> >(triangulation.n_global_active_cells());
    recv_buf_size[iSymm]=std::vector<std::vector<std::vector<int>> >(triangulation.n_global_active_cells());
    rhoRecvd[iSymm]=std::vector<std::vector<std::vector<double>> >(triangulation.n_global_active_cells());
    groupOffsets[iSymm]=std::vector<std::vector<std::vector<int>> >(triangulation.n_global_active_cells());
    if (xc_id==4)
    gradRhoRecvd[iSymm]=std::vector<std::vector<std::vector<double>> >(triangulation.n_global_active_cells());
  }
 //
 unsigned int n_vertices = sort_vertex(dofHandlerEigen) ;
 //unsigned int min_index=0 ;
 //vertex2cell.resize(n_vertices) ;
 //
 cellTemp = dofHandlerEigen.begin_active();
  for (; cellTemp!=endc; ++cellTemp) 
    {
      dealIICellId [cell_id] = cellTemp;
      globalCellId[cellTemp->id()] = cell_id;
      /*double pmin = 1.0E6 ;
      for (unsigned int i = 0; i<8 ; ++i) {
          p0 = cellTemp->vertex(i) ;
	  double psum = p0.operator()(0) + p0.operator()(1) + p0.operator()(2) ;
	  if (psum < pmin) {
	     pmin = psum ;
	     min_index = i ;
	  }
      }
      unsigned int cell_vertex = cellTemp->vertex_index(min_index) ;
      vertex2cell[cell_vertex] = cellTemp ;
      pcout << "cell " << cellTemp << " vertex id " << cell_vertex << std::endl ;
      */
      if (cellTemp->is_locally_owned())
        ownerProc[cell_id] = this_mpi_process ;
      cell_id++ ;
    }
  //
  MPI_Allreduce(&ownerProc[0],
    &ownerProcGlobal[0],
    n_cell,
    MPI_INT,
    MPI_SUM,
    mpi_communicator) ;
  //
  /*Point<3> p1(0.0155, 0.0155, 0.0155) ;
  unsigned int vertex_id = find_cell (p1) ;
  pcout << " vertex id of given point " << vertex_id << std::endl ;
  pcout << " cell_iterator of the cell containing the point " << vertex2cell[vertex_id] << std::endl ;
  mapped_cell = GridTools::find_active_cell_around_point ( mapping, dofHandlerEigen, p1 ) ;
  pcout << " cell_iterator of the cell containing the point from dealii routine " << mapped_cell.first << std::endl ;*/
  //
  int exception = 0 ;
  cell = dofHandlerEigen.begin_active();
  for (; cell!=endc; ++cell) 
    {
          for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) {
		mappedGroup[iSymm][globalCellId[cell->id()]] = std::vector<std::tuple<int, int, int> >(num_quad_points);
		mappedGroupRecvd1[iSymm][globalCellId[cell->id()]]=std::vector<std::vector<double>>(3);
		rhoRecvd[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<double>>(n_mpi_processes) ;
	  }

	  //
	  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) {
	     count = 0;
	     std::fill(countGroupPerProc.begin(),countGroupPerProc.end(),0);
	     typename DoFHandler<3>::active_cell_iterator cellTemp = dofHandlerEigen.begin_active();
	     for (; cellTemp!=endc; ++cellTemp) 
		   groupId[cellTemp->id()]=-1; 
	     //
	     send_buf_size[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<int>>(n_mpi_processes);
	     //
	     mappedGroupSend0[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<int> >(n_mpi_processes);
	     mappedGroupSend2[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<int> >(n_mpi_processes);
	     mappedGroupSend1[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<std::vector<double>> >(n_mpi_processes);
	     //
	     for(int i = 0; i < n_mpi_processes; ++i) {
		 send_buf_size[iSymm][globalCellId[cell->id()]][i] = std::vector<int>(3, 0);
		 mappedGroupSend1[iSymm][globalCellId[cell->id()]][i] = std::vector<std::vector<double>>(3);
		}
             recv_buf_size[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<int>>(3);
             recv_buf_size[iSymm][globalCellId[cell->id()]][0] = std::vector<int>(n_mpi_processes);
	     recv_buf_size[iSymm][globalCellId[cell->id()]][1] = std::vector<int>(n_mpi_processes);
	     recv_buf_size[iSymm][globalCellId[cell->id()]][2] = std::vector<int>(n_mpi_processes);
	     //
	     groupOffsets[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<int>>(3);
	     groupOffsets[iSymm][globalCellId[cell->id()]][0] = std::vector<int>(n_mpi_processes);
	     groupOffsets[iSymm][globalCellId[cell->id()]][1] = std::vector<int>(n_mpi_processes);
	     groupOffsets[iSymm][globalCellId[cell->id()]][2] = std::vector<int>(n_mpi_processes);
             //
	     //
	     if (cell->is_locally_owned()) {
		fe_values.reinit (cell);
             for(unsigned int q_point=0; q_point<num_quad_points; ++q_point) {
                 p = fe_values.quadrature_point(q_point) ;
	         p0 = crys2cart(p,-1) ;
	         //
		 //ptemp[0] = p0[0]*symmMat[iSymm][0][0] + p0[1]*symmMat[iSymm][1][0] + p0[2]*symmMat[iSymm][2][0] ;
                 //ptemp[1] = p0[0]*symmMat[iSymm][0][1] + p0[1]*symmMat[iSymm][1][1] + p0[2]*symmMat[iSymm][2][1] ;
                 //ptemp[2] = p0[0]*symmMat[iSymm][0][2] + p0[1]*symmMat[iSymm][1][2] + p0[2]*symmMat[iSymm][2][2] ; 
	         //
                 ptemp[0] = p0[0]*symmMat[iSymm][0][0] + p0[1]*symmMat[iSymm][0][1] + p0[2]*symmMat[iSymm][0][2] ;
                 ptemp[1] = p0[0]*symmMat[iSymm][1][0] + p0[1]*symmMat[iSymm][1][1] + p0[2]*symmMat[iSymm][1][2] ;
                 ptemp[2] = p0[0]*symmMat[iSymm][2][0] + p0[1]*symmMat[iSymm][2][1] + p0[2]*symmMat[iSymm][2][2] ; 
		 //
		 ptemp[0] = ptemp[0] + translation[iSymm][0] ;
		 ptemp[1] = ptemp[1] + translation[iSymm][1] ;
		 ptemp[2] = ptemp[2] + translation[iSymm][2] ;
		 //
		 for (unsigned int i=0; i<3; ++i){
		    while (ptemp[i] > 0.5)
			ptemp[i] = ptemp[i] - 1.0 ;
		   while (ptemp[i] < -0.5)
			ptemp[i] = ptemp[i] + 1.0 ;
		 }
                 p = crys2cart(ptemp,1) ;
                 //mapped_cell = GridTools::find_active_cell_around_point ( mapping, dofHandlerEigen, p ) ;
		 if (q_point==0){
			  //vertex_id = find_cell (p) ;
			  //std::cout << this_mpi_process << " " << vertex_id << "  " << p.operator()(0) << p.operator()(1) << p.operator()(2) << std::endl ;
		          //mapped_cell.first = vertex2cell[vertex_id ] ;
			  //mapped_cell = find_active_cell_around_point_custom ( mapping, dofHandlerEigen, p ) ;
			  mapped_cell = GridTools::find_active_cell_around_point ( mapping, dofHandlerEigen, p ) ;
		}
		else {
		  double dist = 1.0E+06 ;
		 try {
		    Point<3> p_cell =  mapping.transform_real_to_unit_cell( mapped_cell.first, p ) ;
		    dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);
		    if (dist < 1.0E-10)
			mapped_cell.second = p_cell ;				
		 }
		 catch (MappingQ1<3>::ExcTransformationFailed)
		   {

		   }	
		 if (dist > 1.0E-10)
		    mapped_cell = GridTools::find_active_cell_around_point ( mapping, dofHandlerEigen, p ) ;
		 }
	        /* if (exception==1) {
		     mapped_cell = GridTools::find_active_cell_around_point ( mapping, dofHandlerEigen, p ) ;
		     pcout << " entered exception loop "  << p[0] << "  " << p[1] << "  " << p[2] << std::endl ;
		     pcout << " epoint inside cell "  << mapped_cell.second[0] << "  " << mapped_cell.second[1] << "  " << mapped_cell.second[2] << std::endl ;
		     exception = 0;
		 }*/
		 Point<3> pointTemp = mapped_cell.second;
		 //Point<3> pointTemp = GeometryInfo<3>::project_to_unit_cell (mapped_cell.second);
		 //		 
		 mappedPoint[0] = pointTemp.operator()(0);
		 mappedPoint[1] = pointTemp.operator()(1);
		 mappedPoint[2] = pointTemp.operator()(2);
		//
	        for (unsigned int i = 0; i<3; ++i) {
		     if (mappedPoint[i] < 1.0E-10){
			pcout << mappedPoint[i] << "  " << p[0] << "  " << p[1] << "  " << p[2] << std::endl ;
			mappedPoint[i] = double (0.0) ;
			}
		      if (mappedPoint[i] > ( 1.00-1.0E-10) ) {
			pcout << mappedPoint[i] << "  " << p[0] << "  " << p[1] << "  " << p[2] << std::endl ;
			mappedPoint[i] = double (1.0) ;
			}
                }
		ownerProcId = ownerProcGlobal[globalCellId[mapped_cell.first->id()]] ;
		//
		tupleTemp = std::make_tuple(ownerProcId,mappedPoint,q_point);
		cellMapTable[mapped_cell.first->id()].push_back(tupleTemp) ;
		//
		//
		send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][1] = send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][1] + 1;
		send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][2] = send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][2] + 1;
		//
		/*if (groupId[mapped_cell.first->id()]==-1) {
                   mappedPointId = mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].size() ;
		   tupleTemp2 = std::make_tuple(ownerProcId, groupId[mapped_cell.first->id()],mappedPointId);
		   mappedGroup[iSymm][globalCellId[cell->id()]][q_point] = tupleTemp2 ;
		   //
		   //tupleTemp = std::make_tuple(mapped_cell.first,mappedPoint,q_point);
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].push_back(mappedPoint[0]) ;
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][1].push_back(mappedPoint[1]) ;
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][2].push_back(mappedPoint[2]) ;
		   groupId[mapped_cell.first->id()] = countGroupPerProc[ownerProcId];
		   //pcout << groupId[mapped_cell.first->id()] << std::endl ;
		   tailofGroup[ownerProcId].push_back(mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].size());
		   mappedGroupSend0[iSymm][globalCellId[cell->id()]][ownerProcId].push_back(globalCellId[mapped_cell.first->id()]) ;
		   send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][0] = send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][0] + 1;
		   countPointsPerGroupPerProc[ownerProcId].push_back(1);
		   countGroupPerProc[ownerProcId]++;		   
		   count++ ;
		   }
		else
		   {
		    //tupleTemp = std::make_tuple(mapped_cell.first,mappedPoint,q_point);
		    //
		    //mappedPointId = mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].size() ;
		    //tupleTemp2 = std::make_tuple(ownerProcId, groupId[mapped_cell.first->id()],mappedPointId);
		    //mappedGroup[iSymm][globalCellId[cell->id()]][q_point] = tupleTemp2 ;
		    //
		    //mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].push_back(mappedPoint[0]) ;
		    //mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][1].push_back(mappedPoint[1]) ;
		    //mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][2].push_back(mappedPoint[2]) ;
		    //
		    //pcout << groupId[mapped_cell.first->id()] << std::endl ;
		    std::vector<double> :: iterator itr0 = mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].begin() + tailofGroup[ownerProcId][groupId[mapped_cell.first->id()]] ;
		    std::vector<double> :: iterator itr1 = mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][1].begin() + tailofGroup[ownerProcId][groupId[mapped_cell.first->id()]] ;
		    std::vector<double> :: iterator itr2 = mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][2].begin() + tailofGroup[ownerProcId][groupId[mapped_cell.first->id()]] ;
		    //
		    mappedPointId = tailofGroup[ownerProcId][groupId[mapped_cell.first->id()]] ;
		    tupleTemp2 = std::make_tuple(ownerProcId, groupId[mapped_cell.first->id()],mappedPointId);
		    mappedGroup[iSymm][globalCellId[cell->id()]][q_point] = tupleTemp2 ;
		    //
                    if (itr0!=mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].end()) {
			//
			//std::cout << " entered 0" << std::endl ;
			for (unsigned int iq=0; iq < q_point ; ++iq) {
			     if (std::get<2>(mappedGroup[iSymm][globalCellId[cell->id()]][iq])>=mappedPointId && std::get<0>(mappedGroup[iSymm][globalCellId[cell->id()]][iq])==ownerProcId) {
				std::get<2>(mappedGroup[iSymm][globalCellId[cell->id()]][iq]) = std::get<2>(mappedGroup[iSymm][globalCellId[cell->id()]][iq]) + 1 ;
				 std::cout << " entered 1 " << tailofGroup[ownerProcId][groupId[mapped_cell.first->id()]] << std::endl ;
				}
			}
		    }
                    mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].insert(itr0, 1, mappedPoint[0]) ;
		    mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][1].insert(itr1, 1, mappedPoint[1]) ;
		    mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][2].insert(itr2, 1, mappedPoint[2]) ;
		    //		    
		    //
		    for (unsigned int iGroup=groupId[mapped_cell.first->id()] ; iGroup <= countGroupPerProc[ownerProcId] ; ++iGroup)
		    	tailofGroup[ownerProcId][iGroup] += 1 ;
		    //
		    countPointsPerGroupPerProc[ownerProcId][groupId[mapped_cell.first->id()]] += 1;
		    } */ 
	     }
	     std::fill(countPointPerProc.begin(),countPointPerProc.end(),0);
	     for(std::map<CellId,std::vector<std::tuple<int, std::vector<double>, int> >>::iterator iter = cellMapTable.begin(); iter != cellMapTable.end(); ++iter) {
		 std::vector<std::tuple<int, std::vector<double>, int> > value = iter->second;
		 CellId key = iter->first;
		 ownerProcId = ownerProcGlobal[globalCellId[key]] ;
		 mappedGroupSend0[iSymm][globalCellId[cell->id()]][ownerProcId].push_back(globalCellId[key]) ;
		 mappedGroupSend2[iSymm][globalCellId[cell->id()]][ownerProcId].push_back(value.size()) ;
		 send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][0] = send_buf_size[iSymm][globalCellId[cell->id()]][ownerProcId][0] + 1;
		 //
		 for (unsigned int i=0; i<value.size(); ++i) {
		   //ownerProcId = std::get<0>(value[i]) ; 
		   mappedPoint = std::get<1>(value[i]) ; 
                   int q_point = std::get<2>(value[i]) ; 
		   //
                   tupleTemp2 = std::make_tuple(ownerProcId, 0,countPointPerProc[ownerProcId]);
		   mappedGroup[iSymm][globalCellId[cell->id()]][q_point] = tupleTemp2 ;
		   countPointPerProc[ownerProcId] += 1 ;
		   //
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][0].push_back(mappedPoint[0]) ;
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][1].push_back(mappedPoint[1]) ;
		   mappedGroupSend1[iSymm][globalCellId[cell->id()]][ownerProcId][2].push_back(mappedPoint[2]) ;
		 }	
             }
	     /*for (unsigned int proc=0; proc < n_mpi_processes; ++proc){
	        for ( unsigned int i=0; i < countPointsPerGroupPerProc[proc].size(); ++i)  
	            mappedGroupSend2[iSymm][globalCellId[cell->id()]][proc].push_back(countPointsPerGroupPerProc[proc][i]) ;
	        countPointsPerGroupPerProc[proc].clear();
		tailofGroup[proc].clear();
	    }*/
	    cellMapTable.clear() ;
          }
	  //tailofGroup.clear() ;
        }
     }
     //  
     /*
     /////////////////////////////////////////////////////////////////         OPTIMIZED PORTION (NOT WORKING YET, NEEDS MORE WORK)  ///////////////////////////////////////////////// 
     pcout << " check 0.1 " << std::endl ;
     int recvDataSize0=0, recvDataSize1=0, send_size0, send_size1, send_size2 ;
     std::vector<int> send_data0, send_data2, send_data3;
     std::vector<std::vector<double>> send_data1;
     std::vector<double>  send_data, recvdData ;
     mpi_offsets0.resize(n_mpi_processes, 0) ;
     mpi_offsets1.resize(n_mpi_processes, 0) ;
     recv_size0.resize(n_mpi_processes, 0) ;
     recv_size1.resize(n_mpi_processes, 0) ;
     recvSize.resize(n_mpi_processes, 0) ;
     recvdData1.resize(3);
     send_data1.resize(3);
     //
     for ( unsigned int proc = 0; proc < n_mpi_processes; ++proc) {
	send_size1 = 0 ; send_size0 = 0 ;
	cell = dofHandlerEigen.begin_active();
  	for (; cell!=endc; ++cell) 
    	 {
	 for (unsigned int iSymm = 0; iSymm < numSymm; iSymm++) 
	    {
	     for (unsigned int iPoint = 0; iPoint < send_buf_size[iSymm][globalCellId[cell->id()]][proc][1]; ++iPoint) {
		 //
	         send_data1[0].push_back(mappedGroupSend1[iSymm][globalCellId[cell->id()]][proc][0][iPoint])  ;
		 send_data1[1].push_back(mappedGroupSend1[iSymm][globalCellId[cell->id()]][proc][1][iPoint])  ;
		 send_data1[2].push_back(mappedGroupSend1[iSymm][globalCellId[cell->id()]][proc][2][iPoint])  ;
		 }
	     send_size1 += send_buf_size[iSymm][globalCellId[cell->id()]][proc][1] ;
	     //
	     for (unsigned int i = 0; i < send_buf_size[iSymm][globalCellId[cell->id()]][proc][0]; ++i) {
		 //
	         send_data0.push_back(mappedGroupSend0[iSymm][globalCellId[cell->id()]][proc][i])  ;
		 send_data2.push_back(mappedGroupSend2[iSymm][globalCellId[cell->id()]][proc][i])  ;
		 send_data3.push_back(iSymm)  ;
		 }
	     send_size0 += send_buf_size[iSymm][globalCellId[cell->id()]][proc][0] ;
	     //
	     rhoRecvd[iSymm][globalCellId[cell->id()]][proc].resize((1+spinPolarized)*send_buf_size[iSymm][globalCellId[cell->id()]][proc][1]) ; // to be used later to recv symmetrized rho
	     }
	 }
	 //
	 recvSize[proc] = send_size1 ; // to be used later to recv symmetrized rho
         //
	 MPI_Gather(&send_size0,1,MPI_INT, &(recv_size0[0]),1, MPI_INT,proc,mpi_communicator);
	 MPI_Gather(&send_size1,1,MPI_INT, &(recv_size1[0]),1, MPI_INT,proc,mpi_communicator);
	 //
	 if (proc==this_mpi_process)
	  {
 	  //  
	   for(int i = 1; i < n_mpi_processes; i++) {
  	       mpi_offsets0[i] = recv_size0[i-1]+ mpi_offsets0[i-1];
	       mpi_offsets1[i] = recv_size1[i-1]+ mpi_offsets1[i-1];
	   }
	   //
           recvDataSize0 = std::accumulate(&recv_size0[0], &recv_size0[n_mpi_processes], 0);
           recvdData0.resize(recvDataSize0,0);
	   recvdData2.resize(recvDataSize0,0);
	   recvdData3.resize(recvDataSize0,0);
           //
           recvDataSize1 = std::accumulate(&recv_size1[0], &recv_size1[n_mpi_processes], 0);
	   recvdData.resize(recvDataSize1,0.0) ;
	   for (unsigned int ipol=0; ipol<3; ++ipol)
             recvdData1[ipol].resize(recvDataSize1,0.0) ;
	  }
	  //
	  MPI_Gatherv(&(send_data0[0]),send_size0,MPI_INT, &(recvdData0[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);
	  MPI_Gatherv(&(send_data2[0]),send_size0,MPI_INT, &(recvdData2[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);	
	  MPI_Gatherv(&(send_data3[0]),send_size0,MPI_INT, &(recvdData3[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,proc,mpi_communicator);
	  for (unsigned int ipol=0; ipol<3; ++ipol) {
	      send_data = send_data1[ipol] ;
	      MPI_Gatherv(&(send_data[0]),send_size1,MPI_DOUBLE, &(recvdData[0]),&(recv_size1[0]), &(mpi_offsets1[0]),  MPI_DOUBLE,proc,mpi_communicator); 
	      if (proc==this_mpi_process)
	          recvdData1[ipol] = recvdData ;
	      }	
          send_data0.clear() ; send_data.clear() ; recvdData.clear() ;
          send_data1[0].clear(); send_data1[1].clear(); send_data1[2].clear(); 
	  send_data2.clear() ;
	  send_data3.clear();
     } */
		 

     pcout << " check 0.1 " << std::endl ;
     int recvDataSize0=0, recvDataSize1=0, send_size0, send_size1, send_size2 ;
     std::vector<int> recv_size0(n_mpi_processes,0), recv_size1(n_mpi_processes,0), recv_size2(n_mpi_processes,0);
     std::vector<int> sendData0, sendData2;
     std::vector<std::vector<double>> sendData1, recvdData1;
     std::vector<double>  sendData, recvdData ;
     std::vector<int> mpi_offsets0(n_mpi_processes, 0), mpi_offsets1(n_mpi_processes, 0),  mpi_offsets2(n_mpi_processes, 0);
     std::vector<int> recvdData0, recvdData2;
   cell = dofHandlerEigen.begin_active();
   for (; cell!=endc; cell++) 
     {
          for (unsigned int iSymm = 0; iSymm < numSymm; iSymm++) 
	    {
     	      for (int recvProc = 0; recvProc < n_mpi_processes; recvProc++) {
			
		  //
		  send_size0 = send_buf_size[iSymm][globalCellId[cell->id()]][recvProc][0] ;
	          MPI_Gather(&send_size0,1,MPI_INT, &(recv_size0[0]),1, MPI_INT,recvProc,mpi_communicator);
		  recv_buf_size[iSymm][globalCellId[cell->id()]][0] = recv_size0 ;
		  //
		  send_size1 = send_buf_size[iSymm][globalCellId[cell->id()]][recvProc][1] ;
		  MPI_Gather(&send_size1,1,MPI_INT, &(recv_size1[0]),1, MPI_INT,recvProc,mpi_communicator);
		  recv_buf_size[iSymm][globalCellId[cell->id()]][1] = recv_size1 ;
		  //   	
		 // 
                 if (recvProc==this_mpi_process)
		    {
		     groupOffsets[iSymm][globalCellId[cell->id()]][0][0] = 0 ;  
		     groupOffsets[iSymm][globalCellId[cell->id()]][1][0] = 0 ;  
		     //  
		    for(int i = 1; i < n_mpi_processes; i++) {
  		     groupOffsets[iSymm][globalCellId[cell->id()]][0][i] = recv_buf_size[iSymm][globalCellId[cell->id()]][0][i-1]+ groupOffsets[iSymm][globalCellId[cell->id()]][0][i-1];
		     groupOffsets[iSymm][globalCellId[cell->id()]][1][i] = recv_buf_size[iSymm][globalCellId[cell->id()]][1][i-1]+ groupOffsets[iSymm][globalCellId[cell->id()]][1][i-1];
		     }
		    //
		      mpi_offsets0 = groupOffsets[iSymm][globalCellId[cell->id()]][0] ;
		      mpi_offsets1 = groupOffsets[iSymm][globalCellId[cell->id()]][1] ;
		     //
		      //
		      recvDataSize0 = std::accumulate(&recv_size0[0], &recv_size0[n_mpi_processes], 0);
		      recvdData0.resize(recvDataSize0,0);
		      mappedGroupRecvd0[iSymm][globalCellId[cell->id()]].resize(recvDataSize0) ;
 		      //
		      for ( unsigned int k=0; k<3 ; k++)
		      mappedGroupRecvd1[iSymm][globalCellId[cell->id()]][k].resize(recvDataSize1) ;
		      //
		      recvDataSize1 = std::accumulate(&recv_size1[0], &recv_size1[n_mpi_processes], 0);
		      recvdData.resize(recvDataSize1,0.0) ;
		      recvdData2.resize(recvDataSize0,0);
		      mappedGroupRecvd2[iSymm][globalCellId[cell->id()]].resize(recvDataSize0); 
		    }
		  //
		  for (unsigned int i=0; i < send_size0; ++i)
		  sendData0.push_back(mappedGroupSend0[iSymm][globalCellId[cell->id()]][recvProc][i]) ;
		  //
		  for (unsigned int i=0; i < send_size0; ++i)
		  sendData2.push_back(mappedGroupSend2[iSymm][globalCellId[cell->id()]][recvProc][i]) ;
		  //
		  MPI_Gatherv(&(sendData0[0]),send_size0,MPI_INT, &(recvdData0[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,recvProc,mpi_communicator);
		  //
		 sendData.resize (mappedGroupSend1[iSymm][globalCellId[cell->id()]][recvProc][0].size()) ;
		 for (unsigned int ipol=0; ipol<3; ++ipol){
		   sendData = mappedGroupSend1[iSymm][globalCellId[cell->id()]][recvProc][ipol];
		   MPI_Gatherv(&(sendData[0]),send_size1,MPI_DOUBLE, &(recvdData[0]),&(recv_size1[0]), &(mpi_offsets1[0]),  MPI_DOUBLE,recvProc,mpi_communicator);
		    if (recvProc==this_mpi_process)
		    mappedGroupRecvd1[iSymm][globalCellId[cell->id()]][ipol] = recvdData ;
		   }
		 //
		   MPI_Gatherv(&(sendData2[0]),send_size0,MPI_INT, &(recvdData2[0]),&(recv_size0[0]), &(mpi_offsets0[0]), MPI_INT,recvProc,mpi_communicator);
		 //
		   if (recvProc==this_mpi_process)
		    {
		     //
			mappedGroupRecvd0[iSymm][globalCellId[cell->id()]] = recvdData0 ;
		     //
			mappedGroupRecvd2[iSymm][globalCellId[cell->id()]] = recvdData2 ;
		    }
	
		sendData0.clear();  recvdData0.clear();
		sendData2.clear();  recvdData2.clear();
		sendData.clear();  recvdData.clear();
	    }
	}
     }
    MPI_Barrier(mpi_communicator) ;
    //
   cell = dofHandlerEigen.begin_active();
   totPoints = 0;
   mpi_scatter_offset.resize(n_mpi_processes,0);
   send_scatter_size.resize(n_mpi_processes,0);
   recv_size.resize(n_mpi_processes,0) ;
   //
   for (; cell!=endc; ++cell) 
    {
     for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm)
	{
	 rhoRecvd[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<double>>(n_mpi_processes) ;
	 for (unsigned int proc=0; proc<n_mpi_processes; ++proc){
		totPoints += (1+spinPolarized)*recv_buf_size[iSymm][globalCellId[cell->id()]][1][proc] ;
		mpi_scatter_offset[proc] += (1+spinPolarized)*groupOffsets[iSymm][globalCellId[cell->id()]][1][proc];
		send_scatter_size[proc] += (1+spinPolarized)*recv_buf_size[iSymm][globalCellId[cell->id()]][1][proc];
		recv_size [proc] += (1+spinPolarized)*send_buf_size[iSymm][globalCellId[cell->id()]][proc][1] ;
		rhoRecvd[iSymm][globalCellId[cell->id()]][proc].resize((1+spinPolarized)*send_buf_size[iSymm][globalCellId[cell->id()]][proc][1]) ;
		}
	 }
    }
   //
   if (xc_id==4){
   cell = dofHandlerEigen.begin_active();
   mpi_scatterGrad_offset.resize(n_mpi_processes,0);
   send_scatterGrad_size.resize(n_mpi_processes,0);
   //
   for (; cell!=endc; ++cell) 
    {
     for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm)
	{
	 gradRhoRecvd[iSymm][globalCellId[cell->id()]] = std::vector<std::vector<double>>(n_mpi_processes) ;
	 for (unsigned int proc=0; proc<n_mpi_processes; ++proc){
		mpi_scatterGrad_offset[proc] += (1+spinPolarized)*3*groupOffsets[iSymm][globalCellId[cell->id()]][1][proc];
		send_scatterGrad_size[proc] += (1+spinPolarized)*3*recv_buf_size[iSymm][globalCellId[cell->id()]][1][proc];
		gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc].resize((1+spinPolarized)*3*send_buf_size[iSymm][globalCellId[cell->id()]][proc][1]) ;
		}
	 }
    }
  } 
     
}    
template<unsigned int FEOrder>
Point<3> dftClass<FEOrder>::crys2cart(Point<3> p, int flag)
{
  Point<3> ptemp ;
  if (flag==1){
    ptemp[0] = p[0]*d_latticeVectors[0][0] + p[1]*d_latticeVectors[1][0] + p[2]*d_latticeVectors[2][0] ;
    ptemp[1] = p[0]*d_latticeVectors[0][1] + p[1]*d_latticeVectors[1][1] + p[2]*d_latticeVectors[2][1] ;
    ptemp[2] = p[0]*d_latticeVectors[0][2] + p[1]*d_latticeVectors[1][2] + p[2]*d_latticeVectors[2][2] ;
  }
  if (flag==-1){
    ptemp[0] = p[0]*d_reciprocalLatticeVectors[0][0] + p[1]*d_reciprocalLatticeVectors[0][1] + p[2]*d_reciprocalLatticeVectors[0][2] ;
    ptemp[1] = p[0]*d_reciprocalLatticeVectors[1][0] + p[1]*d_reciprocalLatticeVectors[1][1] + p[2]*d_reciprocalLatticeVectors[1][2] ;
    ptemp[2] = p[0]*d_reciprocalLatticeVectors[2][0] + p[1]*d_reciprocalLatticeVectors[2][1] + p[2]*d_reciprocalLatticeVectors[2][2] ;
    ptemp = 1.0 / (2.0*M_PI) * ptemp ;
  }

  return ptemp;
}
template<unsigned int FEOrder>
void dftClass<FEOrder>:: test_spg_get_ir_reciprocal_mesh()
{
  double lattice[3][3], position[atomLocations.size()][3];
  int num_atom = atomLocations.size();
  int types[num_atom] ;
  int mesh[3] = {nkx, nky, nkz};
  int is_shift[] = {ceil(dkx), ceil(dky), ceil(dkz)};
  int grid_address[nkx * nky * nkz][3];
  int grid_mapping_table[nkx * nky * nkz];
  int max_size = 50;
  //
  for (unsigned int i=0; i<3; ++i) {
     for (unsigned int j=0; j<3; ++j)
         lattice[i][j] = d_latticeVectors[i][j];
  }
  //
  std::set<unsigned int>::iterator it = atomTypes.begin();  
  for (unsigned int i=0; i<atomLocations.size(); ++i){
      std::advance(it, i);
      types[i] = (int)(*it);
      for (unsigned int j=0; j<3; ++j)
      position[i][j] = atomLocationsFractional[i][j+2] ;
   }
  //
  pcout << "*** Testing irreducible BZ with SPG ***" << std:: endl;

  int num_ir = spg_get_ir_reciprocal_mesh(grid_address,
					  grid_mapping_table,
					  mesh,
					  is_shift,
					  0,
					  lattice,
					  position,
					  types,
					  num_atom,
					  1e-5);

  char buffer[100];
  //
  std::vector<int> v(grid_mapping_table, grid_mapping_table + sizeof grid_mapping_table / sizeof grid_mapping_table[0]);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  //
  pcout << "Number of irreducible BZ points using SPG " << num_ir <<std:: endl;
  for (unsigned int i=0; i<v.size(); ++i) {
     sprintf(buffer, "  %5u:  %12.5f  %12.5f %12.5f\n", i+1, float(grid_address[v[i]][0])/float(nkx), float(grid_address[v[i]][1])/float(nky), float(grid_address[v[i]][2])/float(nkz));
     pcout << buffer;
  }
   pcout << "*** Make sure previously given k-points are same as the one above ***" << std:: endl;
}
 //
  template<unsigned int FEOrder>
  unsigned int dftClass<FEOrder>:: find_closest_vertex_custom (const DoFHandler<3> &mesh,
                       const Point<3>        &p)
 {
    // first get the underlying
    // triangulation from the
    // mesh and determine vertices
    // and used vertices
    //const Triangulation<3> &tria = mesh.get_triangulation();

    //const std::vector< Point<3> > &vertices = tria.get_vertices();
    //const std::vector< bool       > &used     = tria.get_used_vertices();

    // At the beginning, the first
    // used vertex is the closest one
    //std::vector<bool>::const_iterator first =
    //  std::find(used.begin(), used.end(), true);

    // Assert that at least one vertex
    // is actually used
    //Assert(first != used.end(), ExcInternalError());

    //unsigned int best_vertex = std::distance(used.begin(), first);
    //double       best_dist   = (p - vertices[best_vertex]).norm_square();
     unsigned int best_vertex ;
     double       best_dist   = 1.0E6 ;
    //
    unsigned int indx = bisectionSearch( vertices_x_unique, p.operator()(0) );
    //
    // For all remaining vertices, test
    // whether they are any closer
    for (unsigned int i = 0; i < index_list_x[indx].size(); i++) {
	unsigned int j = index_list_x[indx][i] ;
      //if (used[j]) { 
//	if ( std::abs(p.operator()(0)-vertices[j].operator()(0)) < 0.5 
//		 && std::abs(p.operator()(1)-vertices[j].operator()(1)) < 0.5 && std::abs(p.operator()(1)-vertices[j].operator()(1)) < 0.5)
 //       {
          double dist = (p - vertices[j]).norm_square();
          if (dist < best_dist)
            {
              best_vertex = j;
              best_dist   = dist;
            }
        //}
    //}
    }
    return best_vertex;
  }
  //
   void find_active_cell_around_point_internal_custom
    (const DoFHandler<3> &mesh,
     std::set<typename DoFHandler<3>::active_cell_iterator> &searched_cells,
     std::set<typename DoFHandler<3>::active_cell_iterator> &adjacent_cells)
    {
      typedef typename DoFHandler<3>::active_cell_iterator cell_iterator;

      // update the searched cells
      searched_cells.insert(adjacent_cells.begin(), adjacent_cells.end());
      // now we to collect all neighbors
      // of the cells in adjacent_cells we
      // have not yet searched.
      std::set<cell_iterator> adjacent_cells_new;

      typename std::set<cell_iterator>::const_iterator
      cell = adjacent_cells.begin(),
      endc = adjacent_cells.end();
      for (; cell != endc; ++cell)
        {
          std::vector<cell_iterator> active_neighbors;
          GridTools::get_active_neighbors<DoFHandler<3> >(*cell, active_neighbors);
          for (unsigned int i=0; i<active_neighbors.size(); ++i)
            if (searched_cells.find(active_neighbors[i]) == searched_cells.end())
              adjacent_cells_new.insert(active_neighbors[i]);
        }
      adjacent_cells.clear();
      adjacent_cells.insert(adjacent_cells_new.begin(), adjacent_cells_new.end());
      if (adjacent_cells.size() == 0)
        {
          // we haven't found any other cell that would be a
          // neighbor of a previously found cell, but we know
          // that we haven't checked all cells yet. that means
          // that the domain is disconnected. in that case,
          // choose the first previously untouched cell we
          // can find
          cell_iterator it = mesh.begin_active();
          for ( ; it!=mesh.end(); ++it)
            if (searched_cells.find(it) == searched_cells.end())
              {
                adjacent_cells.insert(it);
                break;
              }
        }
    }
  //
  template<unsigned int FEOrder>
  std::pair<typename DoFHandler<3>::active_cell_iterator, Point<3> > 
  dftClass<FEOrder>:: find_active_cell_around_point_custom (const Mapping<3>  &mapping,
                                 const DoFHandler<3> &mesh,
                                 const Point<3>        &p)
  {
    //typedef typename dealii::internal::ActiveCellIterator<3, 3, typename DoFHandler<3> >::type active_cell_iterator;

    // The best distance is set to the
    // maximum allowable distance from
    // the unit cell; we assume a
    // max. deviation of 1e-10
    double best_distance = 1e-10;
    int    best_level = -1;
    std::pair<DoFHandler<3>::active_cell_iterator, Point<3> > best_cell;

    // Find closest vertex and determine
    // all adjacent cells
    std::vector<DoFHandler<3>::active_cell_iterator> adjacent_cells_tmp
      = GridTools::find_cells_adjacent_to_vertex(mesh,
                                      find_closest_vertex_custom(mesh, p));

    // Make sure that we have found
    // at least one cell adjacent to vertex.
    Assert(adjacent_cells_tmp.size()>0, ExcInternalError());

    // Copy all the cells into a std::set
    std::set<DoFHandler<3>::active_cell_iterator> adjacent_cells (adjacent_cells_tmp.begin(),
                                                   adjacent_cells_tmp.end());
    std::set<DoFHandler<3>::active_cell_iterator> searched_cells;

    // Determine the maximal number of cells
    // in the grid.
    // As long as we have not found
    // the cell and have not searched
    // every cell in the triangulation,
    // we keep on looking.
    const unsigned int n_active_cells = mesh.get_triangulation().n_active_cells();
    bool found = false;
    unsigned int cells_searched = 0;
    while (!found && cells_searched < n_active_cells)
      {
        typename std::set<DoFHandler<3>::active_cell_iterator>::const_iterator
        cell = adjacent_cells.begin(),
        endc = adjacent_cells.end();
        for (; cell != endc; ++cell)
          {
            try
              {
                const Point<3> p_cell = mapping.transform_real_to_unit_cell(*cell, p);

                // calculate the infinity norm of
                // the distance vector to the unit cell.
                const double dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);

                // We compare if the point is inside the
                // unit cell (or at least not too far
                // outside). If it is, it is also checked
                // that the cell has a more refined state
                if ((dist < best_distance)
                    ||
                    ((dist == best_distance)
                     &&
                     ((*cell)->level() > best_level)))
                  {
                    found         = true;
                    best_distance = dist;
                    best_level    = (*cell)->level();
                    best_cell     = std::make_pair(*cell, p_cell);
                  }
              }
            catch (typename MappingQGeneric<3>::ExcTransformationFailed &)
              {
                // ok, the transformation
                // failed presumably
                // because the point we
                // are looking for lies
                // outside the current
                // cell. this means that
                // the current cell can't
                // be the cell around the
                // point, so just ignore
                // this cell and move on
                // to the next
              }
          }

        // update the number of cells searched
        cells_searched += adjacent_cells.size();

        // if we have not found the cell in
        // question and have not yet searched every
        // cell, we expand our search to
        // all the not already searched neighbors of
        // the cells in adjacent_cells. This is
        // what find_active_cell_around_point_internal
        // is for.
        if (!found && cells_searched < n_active_cells)
          {
            find_active_cell_around_point_internal_custom(mesh, searched_cells, adjacent_cells);
          }
      }

    AssertThrow (best_cell.first.state() == IteratorState::valid,
                 GridTools::ExcPointNotFound<3>(p));

    return best_cell;
  } 
 
  //
 
  template<unsigned int FEOrder>
  unsigned int dftClass<FEOrder>::  sort_vertex (const DoFHandler<3> &mesh)
  {
    // first get the underlying
    // triangulation from the
    // mesh and determine vertices
    // and used vertices
    const Triangulation<3> &tria = mesh.get_triangulation();

    vertices = tria.get_vertices();
    const std::vector< bool       > &used     = tria.get_used_vertices();
    std::vector<double> vertices_x, vertices_y, vertices_z ;
    std::vector<unsigned int> index ;
    unsigned int n_vertices = vertices.size() ;
    //
    for (unsigned int j = 0; j < vertices.size(); j++) {
    	if (used[j]) {
	Point<3> v = vertices[j] ;
	vertices_x.push_back(v.operator()(0)) ; 
    	vertices_y.push_back(v.operator()(1)) ; 
    	vertices_z.push_back(v.operator()(2)) ; 
	index.push_back(j) ;
	}
        //pcout << vertices[j].operator()(0) << vertices[j].operator()(1) << vertices[j].operator()(2) << std::endl ;
    }
    std::vector<unsigned int> idx(vertices_x.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&vertices_x](unsigned int i1, unsigned int i2) {return vertices_x[i1] < vertices_x[i2];});
    //
    std::vector<unsigned int> idy(vertices_y.size());
    std::iota(idy.begin(), idy.end(), 0);
    std::sort(idy.begin(), idy.end(), [&vertices_y](unsigned int i1, unsigned int i2) {return vertices_y[i1] < vertices_y[i2];});
    //
    std::vector<unsigned int> idz(vertices_z.size());
    std::iota(idz.begin(), idz.end(), 0);
    std::sort(idz.begin(), idz.end(), [&vertices_z](unsigned int i1, unsigned int i2) {return vertices_z[i1] < vertices_z[i2];});
    //
    std::sort(vertices_x.begin(), vertices_x.end()) ;
    std::sort(vertices_y.begin(), vertices_y.end()) ;
    std::sort(vertices_z.begin(), vertices_z.end()) ;
    //
    std::vector<unsigned int> temp ;
    //
    //pcout << "**********  x direction ***********" << std::endl ;
    double current_value = vertices_x[0] ;
    vertices_x_unique.push_back(current_value) ;
    for (unsigned int i = 0; i < vertices_x.size() ; ++i ) {
	if ( vertices_x[i] > current_value ) {
	    std::sort(temp.begin(),temp.end()) ;
	    index_list_x.push_back(temp) ;
	    temp.clear();
	    current_value = vertices_x[i];
	    vertices_x_unique.push_back(current_value) ;
	    //pcout << current_value << std::endl ;
	}
	//else
	temp.push_back ( index[idx[i]] ) ;
    }
    std::sort(temp.begin(),temp.end()) ;
    index_list_x.push_back(temp) ;
    temp.clear();
    //
    //pcout << "**********  y direction ***********" << std::endl ;
    current_value = vertices_y[0] ;
    vertices_y_unique.push_back(current_value) ;
    for (unsigned int i = 0; i < vertices_y.size() ; ++i ) {
	if ( vertices_y[i] > current_value ) {
	    std::sort(temp.begin(),temp.end()) ;
	    index_list_y.push_back(temp) ;
	    temp.clear();
	    current_value = vertices_y[i];
	    vertices_y_unique.push_back(current_value) ;
	    //pcout << current_value << std::endl ;
	}
	//else
	temp.push_back ( index[idy[i]] ) ;
    }
    std::sort(temp.begin(),temp.end()) ;
    index_list_y.push_back(temp) ;
    temp.clear();
    //
    //pcout << "**********  z direction ***********" << std::endl ;
    current_value = vertices_z[0] ;
    vertices_z_unique.push_back(current_value) ;
    for (unsigned int i = 0; i < vertices_z.size() ; ++i ) {
	if ( vertices_z[i] > current_value ) {
	    std::sort(temp.begin(),temp.end()) ;
	    index_list_z.push_back(temp) ;
	    temp.clear();
	    current_value = vertices_z[i];
	    vertices_z_unique.push_back(current_value) ;
	    //pcout << current_value << std::endl ;
	}
	//else
	temp.push_back ( index[idz[i]] ) ;
    }
    std::sort(temp.begin(),temp.end()) ;
    index_list_z.push_back(temp) ;
    temp.clear();
    //
    //std::sort(index_list_x.begin(),index_list_x.end()) ;
    //std::sort(index_list_y.begin(),index_list_y.end()) ;
    //std::sort(index_list_z.begin(),index_list_z.end()) ;
    return(n_vertices) ;
    }
   //
   template<unsigned int FEOrder>
   unsigned int dftClass<FEOrder>::  find_cell (Point<3> p)
   {
   //
   //std::vector<double>::iterator firstx = vertices_x_unique.begin(), lastx = vertices_x_unique.end();
   //std::vector<double>::iterator iterx = find( firstx, lastx, p.operator()(0) );
   unsigned int indx = bisectionSearch( vertices_x_unique, p.operator()(0) );
   //
   //std::vector<double>::iterator firsty = vertices_y_unique.begin(), lasty = vertices_y_unique.end();
   unsigned int indy = bisectionSearch( vertices_y_unique, p.operator()(1) );
   //
   //std::vector<double>::iterator firstz = vertices_z_unique.begin(), lastz = vertices_z_unique.end();
   unsigned int indz = bisectionSearch( vertices_z_unique, p.operator()(2) );
   //
   //pcout << "indx " << indx << " indy " << indy << " indz  " << indz << std::endl ;
   //
   std::vector<unsigned int> indexList_xy (index_list_x[indx].size()) ;
   std::vector<unsigned int>::iterator iterxy = std::set_intersection( index_list_x[indx].begin(), index_list_x[indx].end(), 
							   index_list_y[indy].begin(), index_list_y[indy].end(), indexList_xy.begin() );
   //for (unsigned int i = 0; i < indexList_xy.size() ; ++i)
   //	pcout << indexList_xy[i] << std::endl ;
   //
   indexList_xy.resize(iterxy-indexList_xy.begin()); 
   //
   //pcout << "indexList_xy.size " << indexList_xy.size() << std::endl ;
   /*if (indexList_xy.size()<21){
	pcout << "index_list_x[indx].size() " << index_list_x[indx].size() << std::endl ;
	pcout << "index_list_y[indy].size() " << index_list_y[indy].size() << std::endl ;
	for (unsigned int i=0; i< index_list_x[indx].size(); ++i)
	    pcout << index_list_x[indx][i] << "  " << index_list_y[indy][i] << std::endl ;
   }*/
   //
   std::vector<unsigned int> indexList_xyz (indexList_xy.size()) ;
   std::vector<unsigned int>::iterator iterxyz = std::set_intersection( indexList_xy.begin(), indexList_xy.end(), 
   							   index_list_z[indz].begin(), index_list_z[indz].end(), indexList_xyz.begin() );
   //pcout << "indexList_xyz.size " << indexList_xyz.size() << std::endl ;
   indexList_xyz.resize(iterxyz-indexList_xyz.begin()); 
   //pcout << "indexList_xyz.size " << indexList_xyz.size() << std::endl ;
   return(indexList_xyz[0]) ;
   }
  //
  template<unsigned int FEOrder>
  unsigned int dftClass<FEOrder>:: bisectionSearch(std::vector<double> &arr, double x)
   {
   unsigned int n = arr.size();
   unsigned int start_point = 0 ;
   if (x > arr[n-1] )
	return(999999) ;
   //
   while ( n > 2) {

       unsigned int check_point = start_point + ceil(double(n)/double(2)) -1 ;
	if ( x > arr[check_point] ) 
	    start_point = check_point ;
	n = n - ceil(double(n)/double(2)) + 1 ;
    }
   //
    return(start_point) ; 
  }
  /*void test ()
  {
    call_another_test() ;   
  }*/

