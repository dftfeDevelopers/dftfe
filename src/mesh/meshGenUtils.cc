namespace meshGenUtils
{
void cross_product(std::vector<double> &a,
		   std::vector<double> &b,
		   std::vector<double> &crossProduct)
{
  crossProduct.resize(a.size(),0.0);

  crossProduct[0] = a[1]*b[2]-a[2]*b[1];
  crossProduct[1] = a[2]*b[0]-a[0]*b[2];
  crossProduct[2] = a[0]*b[1]-a[1]*b[0];

}

void computePeriodicFaceNormals(std::vector<std::vector<double> > & latticeVectors,
				std::vector<std::vector<double> > & periodicFaceNormals)
{

  //resize periodic face normal
  periodicFaceNormals.resize(3);


  //evaluate cross product between first two lattice vectors 
  cross_product(latticeVectors[0],
		latticeVectors[1],
		periodicFaceNormals[2]);

  //evaluate cross product between second two lattice vectors 
  cross_product(latticeVectors[1],
		latticeVectors[2],
		periodicFaceNormals[0]);

  
  //evaluate cross product between third and first two lattice vectors 
  cross_product(latticeVectors[2],
		latticeVectors[0],
		periodicFaceNormals[1]);

}

void computeOffsetVectors(std::vector<std::vector<double> > & latticeVectors,
			  std::vector<Tensor<1,3> > & offsetVectors)
{
  //create unitVectorsXYZ
  std::vector<std::vector<double> > unitVectorsXYZ;
  unitVectorsXYZ.resize(3);

  for(int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3,0.0);
      unitVectorsXYZ[i][i] = 1.0;
    }


  //resize offset vectors
  offsetVectors.resize(3);

  for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 3; ++j)
	{
	  offsetVectors[i][j] = unitVectorsXYZ[i][j] - latticeVectors[i][j];
	}
    }

}


double getCosineAngle(Tensor<1,3> & Vector1,
		      std::vector<double> & Vector2)
{
  
  double dotProduct = Vector1[0]*Vector2[0] + Vector1[1]*Vector2[1] + Vector1[2]*Vector2[2];
  double lengthVector1Sq = (Vector1[0]*Vector1[0] + Vector1[1]*Vector1[1] + Vector1[2]*Vector1[2]);
  double lengthVector2Sq = (Vector2[0]*Vector2[0] + Vector2[1]*Vector2[1] + Vector2[2]*Vector2[2]);

  double angle = dotProduct/sqrt(lengthVector1Sq*lengthVector2Sq);

  return angle;
  
}


double getCosineAngle(std::vector<double> & Vector1,
		      std::vector<double> & Vector2)
{
  
  double dotProduct = Vector1[0]*Vector2[0] + Vector1[1]*Vector2[1] + Vector1[2]*Vector2[2];
  double lengthVector1Sq = (Vector1[0]*Vector1[0] + Vector1[1]*Vector1[1] + Vector1[2]*Vector1[2]);
  double lengthVector2Sq = (Vector2[0]*Vector2[0] + Vector2[1]*Vector2[1] + Vector2[2]*Vector2[2]);

  double angle = dotProduct/sqrt(lengthVector1Sq*lengthVector2Sq);

  return angle;
  
}



void markPeriodicFacesNonOrthogonal(Triangulation<3,3> &triangulation, 
				    std::vector<std::vector<double> > & latticeVectors)
{

  std::vector<std::vector<double> > periodicFaceNormals;
  std::vector<Tensor<1,3> > offsetVectors;

  //compute periodic face normals from lattice vector information
  computePeriodicFaceNormals(latticeVectors,
			     periodicFaceNormals);

  //compute offset vectors such that lattice vectors plus offset vectors are alligned along spatial x,y,z directions 
  computeOffsetVectors(latticeVectors,
		       offsetVectors);

  std::cout<<"Periodic Face Normals 1: "<<periodicFaceNormals[0][0]<<" "<<periodicFaceNormals[0][1]<<" "<<periodicFaceNormals[0][2]<<std::endl;
  std::cout<<"Periodic Face Normals 2: "<<periodicFaceNormals[1][0]<<" "<<periodicFaceNormals[1][1]<<" "<<periodicFaceNormals[1][2]<<std::endl;
  std::cout<<"Periodic Face Normals 3: "<<periodicFaceNormals[2][0]<<" "<<periodicFaceNormals[2][1]<<" "<<periodicFaceNormals[2][2]<<std::endl;

  QGauss<2>  quadratureFace_formula(2); FESystem<3>  FE(FE_Q<3>(QGaussLobatto<1>(2)), 1);
  FEFaceValues<3> feFace_values(FE, quadratureFace_formula, update_normal_vectors);

  typename Triangulation<3,3>::active_cell_iterator cell, endc;
  //
  //mark faces
  //
  cell = triangulation.begin_active(), endc = triangulation.end();
  for(;cell!=endc; ++cell) 
    {
      for(unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if (cell->face(f)->at_boundary())
	    {

	      feFace_values.reinit(cell,f);
	      Tensor<1,3> faceNormalVector = feFace_values.normal_vector(0);

	      //std::cout<<"Face normal vector: "<<faceNormalVector[0]<<" "<<faceNormalVector[1]<<" "<<faceNormalVector[2]<<std::endl;
	      std::cout<<"Angle : "<<getCosineAngle(faceNormalVector,periodicFaceNormals[0])<<" "<<getCosineAngle(faceNormalVector,periodicFaceNormals[1])<<" "<<getCosineAngle(faceNormalVector,periodicFaceNormals[2])<<std::endl;
	      
	      if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[0]) + 1.0) < 1.0e-05)
		{
		  cell->face(f)->set_boundary_id(1);
		  std::cout<<"Boundary 1"<<std::endl;
		}
	      else if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[0]) - 1.0) < 1.0e-05)
		{
		  cell->face(f)->set_boundary_id(2);
		  std::cout<<"Boundary 2"<<std::endl;
		}
	      else if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[1]) + 1.0) < 1.0e-05) 
		{
		  cell->face(f)->set_boundary_id(3);
		  std::cout<<"Boundary 3"<<std::endl;
		}
	      else if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[1]) - 1.0) < 1.0e-05) 
		{
		  cell->face(f)->set_boundary_id(4);
		  std::cout<<"Boundary 4"<<std::endl;
		}
	      else if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[2]) + 1.0) < 1.0e-05) 
		{
		  cell->face(f)->set_boundary_id(5);
		  std::cout<<"Boundary 5"<<std::endl;
		}
	      else if(std::abs(getCosineAngle(faceNormalVector,periodicFaceNormals[2]) - 1.0) < 1.0e-05) 
		{
		  cell->face(f)->set_boundary_id(6);
		  std::cout<<"Boundary 6"<<std::endl;
		}
	      else
		std::cout<<"Domain is not periodic: "<<std::endl;
	      
	    }
	}
    }

  //pcout << "Done with Boundary Flags\n";
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<3,3>::cell_iterator> > periodicity_vector;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector, offsetVectors[i]);
      //GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);      
    }
  triangulation.add_periodicity(periodicity_vector);
  std::cout << "Periodic Facepairs: " << periodicity_vector.size() << std::endl;
}

void markPeriodicFaces(Triangulation<3,3> &triangulation)
{

  double domainSizeX = dftParameters::domainSizeX,domainSizeY = dftParameters::domainSizeY,domainSizeZ=dftParameters::domainSizeZ;

  std::cout << "domainSizeX: "<<domainSizeX<<",domainSizeY: "<<domainSizeY<<",domainSizeZ: "<<domainSizeZ<< std::endl;

  typename Triangulation<3,3>::active_cell_iterator cell, endc;

  //
  //mark faces
  //
  cell = triangulation.begin_active(), endc = triangulation.end();
  for(; cell!=endc; ++cell) 
    {
      for(unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if(cell->face(f)->at_boundary())
	    {
	      if (std::abs(face_center[0]+(domainSizeX/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(1);
	      else if (std::abs(face_center[0]-(domainSizeX/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(2);
	      else if (std::abs(face_center[1]+(domainSizeY/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(3);
	      else if (std::abs(face_center[1]-(domainSizeY/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(4);
	      else if (std::abs(face_center[2]+(domainSizeZ/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(5);
	      else if (std::abs(face_center[2]-(domainSizeZ/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(6);
	    }
	}
    }

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<3,3>::cell_iterator> > periodicity_vector;
  for(int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
    }

  triangulation.add_periodicity(periodicity_vector);
}

}
