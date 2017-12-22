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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//source file for all mesh reading/generation functions

//Generate triangulation.
template<unsigned int FEOrder>
void dftClass<FEOrder>::mesh(){
  computing_timer.enter_section("mesh"); 
if (meshFileName!="") {
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);
  //Read mesh in UCD format generated from Cubit
  std::ifstream f(meshFileName.c_str());
  gridin.read_ucd(f);
#ifdef ENABLE_PERIODIC_BC
  //mark faces
 typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
 cell = triangulation.begin_active(), endc = triangulation.end();
  for(; cell!=endc; ++cell) 
    {
      for(unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if(cell->face(f)->at_boundary())
	    {
	      if (std::abs(face_center[0]+(domainSizeX/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(1);
	      else if (std::abs(face_center[0]-(domainSizeX/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(2);
	      else if (std::abs(face_center[1]+(domainSizeY/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(3);
	      else if (std::abs(face_center[1]-(domainSizeY/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(4);
	      else if (std::abs(face_center[2]+(domainSizeZ/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(5);
	      else if (std::abs(face_center[2]-(domainSizeZ/2.0))<1.0e-5)
		cell->face(f)->set_boundary_id(6);
	    }
	}
    }

  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<3>::cell_iterator> > periodicity_vector;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
    }
  triangulation.add_periodicity(periodicity_vector);
#endif 
 }
else {
    Point<3> P1(domainSizeX/2.0,domainSizeY/2.0,domainSizeZ/2.0);
    Point<3> P2(-domainSizeX/2.0,-domainSizeY/2.0,-domainSizeZ/2.0);
   
    GridGenerator::hyper_rectangle (triangulation, P1, P2,false);

 typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
#ifdef ENABLE_PERIODIC_BC
  //mark faces
 cell = triangulation.begin_active(), endc = triangulation.end();
  for(; cell!=endc; ++cell) 
    {
      for(unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if(cell->face(f)->at_boundary())
	    {
	      if (std::abs(face_center[0]+(domainSizeX/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(1);
	      else if (std::abs(face_center[0]-(domainSizeX/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(2);
	      else if (std::abs(face_center[1]+(domainSizeY/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(3);
	      else if (std::abs(face_center[1]-(domainSizeY/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(4);
	      else if (std::abs(face_center[2]+(domainSizeZ/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(5);
	      else if (std::abs(face_center[2]-(domainSizeZ/2.0))<1.0e-8)
		cell->face(f)->set_boundary_id(6);
	    }
	}
    }

  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<3>::cell_iterator> > periodicity_vector;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
    }
  triangulation.add_periodicity(periodicity_vector);
#endif  
  triangulation.refine_global(n_refinement_steps); 

  //
  char buffer1[100];
  sprintf(buffer1, "Adaptive meshing:\nbase mesh number of elements: %u\n", triangulation.n_global_active_cells());
  pcout << buffer1;
 
}
 computing_timer.exit_section("mesh"); 
}
