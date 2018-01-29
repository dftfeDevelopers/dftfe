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
// @author Phani Motamarri (2017)
//

#ifndef symmetry_H_
#define symmetry_H_
#include "headers.h"
#include "constants.h"
#include "dft.h"

#include <iostream>
#include <iomanip> 
#include <numeric>
#include <sstream>
#include <complex>
#include <deque>

using namespace dealii;

template <unsigned int FEOrder>
class symmetryClass
{
 template <unsigned int T>
 friend class dftClass;

 template <unsigned int T>
 friend class eigenClass;

 public:
  /**
   * symmetryClass constructor
   */
  symmetryClass(dftClass<FEOrder>* _dftPtr,  MPI_Comm &mpi_comm_replica, MPI_Comm &interpoolcomm);


  /**
   * symmetryClass destructor
   */
  //~symmetryClass();

 

  void test_spg_get_ir_reciprocal_mesh();
  void initSymmetry();
  void computeAndSymmetrize_rhoOut();
  void computeLocalrhoOut();
  Point<3> crys2cart(Point<3> p, int i);
  
  
 private:
  dftClass<FEOrder>* dftPtr;
   //FE data structres
  dealii::FE_Q<3>   FE;
  //compute-time logger
  dealii::TimerOutput computing_timer;
   //parallel objects
  MPI_Comm mpi_communicator, interpoolcomm;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;
  //
  std::vector<std::vector<std::vector<double> >> symmMat;
  unsigned int numSymm;
  std::map<CellId,std::vector<std::tuple<int, std::vector<double>, int> >>  cellMapTable ;
  //std::vector<std::map<CellId,std::vector<std::vector<std::tuple<typename DoFHandler<3>::active_cell_iterator, Point<3>, int> >>>> mappedGroup ;
  std::vector<std::vector<std::vector<std::tuple<int, int, int> >>> mappedGroup ;
  // Communication vectors required for rho-symmetrization
  std::vector<std::vector<std::vector<std::vector<int> >>> mappedGroupSend0;
  std::vector<std::vector<std::vector<std::vector<int> >>> mappedGroupSend2;
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>> >> mappedGroupSend1;
  std::vector<std::vector<std::vector<int> >> mappedGroupRecvd0;
  std::vector<std::vector<std::vector<int> >> mappedGroupRecvd2;
  std::vector<std::vector<std::vector<std::vector<double>>> > mappedGroupRecvd1;
  std::vector<std::vector<std::vector<std::vector<int>> >> send_buf_size;
  std::vector<std::vector<std::vector<std::vector<int>> >> recv_buf_size;
  std::vector<std::vector<std::vector<std::vector<double>>> > rhoRecvd, gradRhoRecvd;
  std::vector<std::vector<std::vector<std::vector<int>> >> groupOffsets;
  std::map<int,typename DoFHandler<3>::active_cell_iterator> dealIICellId ;
  std::map<CellId, int> globalCellId ;
  std::vector<int> ownerProcGlobal;
  std::vector<int> mpi_scatter_offset, send_scatter_size, recv_size, mpi_scatterGrad_offset, send_scatterGrad_size;
  unsigned int totPoints ;
  double translation[500][3];
  std::vector<std::vector<int>> symmUnderGroup ;
  std::vector<int> numSymmUnderGroup ;
  //
  std::vector<typename DoFHandler<3>::active_cell_iterator> vertex2cell ;
  std::vector<double> vertices_x_unique, vertices_y_unique, vertices_z_unique ;
  std::vector<std::vector<unsigned int>> index_list_x, index_list_y, index_list_z ;
  //
  unsigned int bisectionSearch(std::vector<double> &arr, double x) ;
  unsigned int sort_vertex (const DoFHandler<3> &mesh)  ;        
  unsigned int find_cell (Point<3> p) ;  
  //
  std::pair<typename DoFHandler<3>::active_cell_iterator, Point<3> > 
  find_active_cell_around_point_custom (const Mapping<3>  &mapping,
                                 const DoFHandler<3> &mesh,
                                 const Point<3>        &p) ;
  unsigned int find_closest_vertex_custom (const DoFHandler<3> &mesh,
                       const Point<3>        &p) ;
  std::vector< Point<3> > vertices ;
  //
  std::vector<int> mpi_offsets0, mpi_offsets1, mpiGrad_offsets1 ;
  std::vector<int> recvdData0, recvdData2, recvdData3;  
  std::vector<std::vector<double>> recvdData1;
  std::vector<int> recv_size0, recv_size1, recvGrad_size1;

};

#endif
