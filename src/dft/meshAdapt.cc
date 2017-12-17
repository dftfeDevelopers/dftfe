//source file for all mesh reading/generation functions
//

#define approxMaxNumberElements 2000
#define maxRefinementLevels 10

//
//Generate triangulation.
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::mesh()
{
  computing_timer.enter_section("mesh"); 
#ifdef meshFileName1
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);

  //Read mesh in UCD format generated from Cubit
  std::ifstream f(meshFileName.c_str());
  gridin.read_ucd(f);
#else

  //
  //Adaptive mesh generation
  //
  double Ln = domainSizeX/2.0, Ln1 = innerDomainSize/2.0, L1 = outerBallRadius, L0 = innerBallRadius; 
  double hn = meshSizeOuterDomain, hn1=meshSizeInnerDomain, h1=meshSizeOuterBall, h0=meshSizeInnerBall;
  double br = baseRefinementLevel;

  //
  //get the number of image charges
  //  
  const int numberImageCharges = d_imageIds.size();

  //Generate outer box. Here an assumption is made that the box is always centered at the origin. 
  GridGenerator::hyper_cube (triangulation, -Ln, Ln);

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


  triangulation.refine_global(br); //Base refinement is chosen as 4, but this can be changed.

  //
  char buffer1[100];
  sprintf(buffer1, "Adaptive meshing:\nbase mesh number of elements: %u\n", triangulation.n_global_active_cells());
  pcout << buffer1;

  //	
  //Multilayer refinement
  //
  dealii::Point<3> origin;
  unsigned int numLevels=0;
  bool refineFlag=true;

   while(refineFlag)
    {
      refineFlag=false;
      cell = triangulation.begin_active();
      endc = triangulation.end();
      for ( ; cell != endc; ++cell)
	{
	  if (cell->is_locally_owned())
	    {
	      dealii::Point<3> center(cell->center());  
	      double h=cell->minimum_vertex_distance();

	      //loop over all atoms
	      double distanceToClosestAtom = Ln;
	      Point<3> closestAtom;
	      for (unsigned int n=0; n<atomLocations.size(); n++)
		{
		  Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
		  if(center.distance(atom) < distanceToClosestAtom)
		    {
		      distanceToClosestAtom = center.distance(atom);
		      closestAtom = atom;
		    }
		}

	      for (unsigned int iImageCharge=0; iImageCharge < numberImageCharges; ++iImageCharge)
		{
		  Point<3> imageAtom(d_imagePositions[iImageCharge][0],d_imagePositions[iImageCharge][1],d_imagePositions[iImageCharge][2]);
		  if(center.distance(imageAtom) < distanceToClosestAtom)
		    {
		      distanceToClosestAtom = center.distance(imageAtom);
		      closestAtom = imageAtom;
		    }
		}

	      //check for location of the cell with respect to the multiple layers of refinement
	      bool inInnerBall=false, inOuterBall=false, inInnerXYZ=false, inOuterXYZ=false;
	      /*if (distanceToClosestAtom <= L0) {inInnerBall=true;}
	      else if (distanceToClosestAtom <= L1) {inOuterBall=true;}
	      else if (distanceToClosestAtom <= Ln1) {inInnerXYZ=true;}
	      else if (distanceToClosestAtom <= Ln) {inOuterXYZ=true;}

	      //check for multilayer refinement
	      bool cellRefineFlag=false;
	      if (inInnerBall &&  (h>h0))  {cellRefineFlag=true;}       //innerBall refinement
	      else if (inOuterBall &&  (h>h1))  {cellRefineFlag=true;} //outerBall refinement
	      else if (inInnerXYZ  &&  (h>hn1)) {cellRefineFlag=true;} //innerXYZ refinement
	      else if (inOuterXYZ &&  (h>hn)) {cellRefineFlag=true;}   //outerXYZ refinement*/

	      bool cellRefineFlag = false;

	      MappingQ1<3,3> mapping;
	      try
		{
		  Point<3> p_cell = mapping.transform_real_to_unit_cell(cell,closestAtom);
                  double dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);
		  
		  if(dist < 1e-08 && h > h0)
		    cellRefineFlag = true;
		}
	      catch(MappingQ1<3>::ExcTransformationFailed)
		{
		  
		}
	      //set refine flags
	      if(cellRefineFlag)
		{
		  refineFlag=true; //This sets the global refinement sweep flag
		  cell->set_refine_flag();
		}
	    }
	}
  
      
    
      //Refine
      refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
      if (refineFlag)
	{
	  //if ((triangulation.n_global_active_cells()<approxMaxNumberElements) & (numLevels<maxRefinementLevels))
	  if(numLevels<maxRefinementLevels)
	    {
	      char buffer2[100];
	      sprintf(buffer2, "refinement in progress, level: %u\n", numLevels);
	      pcout << buffer2;
	      triangulation.execute_coarsening_and_refinement();
	      numLevels++;
	    }
	  else
	    {
	      refineFlag=false;
	    }
	}
    }

  //compute some adaptive mesh metrics
  double minElemLength=Ln;
  cell = triangulation.begin_active();
  endc = triangulation.end();
  for ( ; cell != endc; ++cell){
    if (cell->is_locally_owned()){
      if (cell->minimum_vertex_distance()<minElemLength) minElemLength = cell->minimum_vertex_distance();
    }
  }
  Utilities::MPI::min(minElemLength, mpi_communicator);

  //print out adaptive mesh metrics
  pcout << "Refinement levels executed: " << numLevels << std::endl;
  char buffer[100];
  sprintf(buffer, "Adaptivity summary:\n numCells: %u, numLevels: %u, h_min: %5.2e\n", triangulation.n_global_active_cells(), numLevels, minElemLength);
  pcout << buffer;
#endif
  //
  computing_timer.exit_section("mesh"); 
}

//adaptive mesh controls
/*#define outerXYZ 20.0 //This corresponds to h_(n) region which is a box
  #define innerXYZ 5.0 //10.0  //This corresponds to h_(n-1) region which is a box
  #define outerBall 0.0 //This corresponds to h_(1) region which is a sphere
  #define innerBall 1.0 //This corresponds to h_(0) region which is a sphere
  #define meshHn  2.50 //5.0
  #define meshHn1 1.25 //2.5
  #define meshH1 0.10
  #define meshH0 0.05*/
