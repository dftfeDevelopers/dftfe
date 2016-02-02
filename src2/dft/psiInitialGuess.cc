#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>

void dftClass::determineOrbitalFilling(){
  //create stencil following orbital filling order
  std::vector<unsigned int> level;
  std::vector<std::vector<unsigned int> > stencil;
  //1s
  level.clear(); level.push_back(0); level.push_back(0); stencil.push_back(level);
  //2s
  level.clear(); level.push_back(1); level.push_back(0); stencil.push_back(level);
  //2p
  level.clear(); level.push_back(1); level.push_back(1); stencil.push_back(level);
  //3s
  level.clear(); level.push_back(2); level.push_back(0); stencil.push_back(level);
  //3p
  level.clear(); level.push_back(2); level.push_back(1); stencil.push_back(level);
  //4s
  level.clear(); level.push_back(3); level.push_back(0); stencil.push_back(level);
  //3d
  level.clear(); level.push_back(2); level.push_back(2); stencil.push_back(level);
  //4p
  level.clear(); level.push_back(3); level.push_back(1); stencil.push_back(level);
  //5s
  level.clear(); level.push_back(4); level.push_back(0); stencil.push_back(level);
  //4d
  level.clear(); level.push_back(3); level.push_back(2); stencil.push_back(level);
  //5p
  level.clear(); level.push_back(4); level.push_back(1); stencil.push_back(level);
  //6s
  level.clear(); level.push_back(5); level.push_back(0); stencil.push_back(level);
  //4f
  level.clear(); level.push_back(3); level.push_back(3); stencil.push_back(level);
  //5d
  level.clear(); level.push_back(4); level.push_back(2); stencil.push_back(level);
  //6p
  level.clear(); level.push_back(5); level.push_back(1); stencil.push_back(level);
  //7s
  level.clear(); level.push_back(6); level.push_back(0); stencil.push_back(level);
  //5f
  level.clear(); level.push_back(4); level.push_back(3); stencil.push_back(level);
  //6d
  level.clear(); level.push_back(5); level.push_back(2); stencil.push_back(level);
  //7p
  level.clear(); level.push_back(6); level.push_back(1); stencil.push_back(level);
  //8s
  level.clear(); level.push_back(7); level.push_back(0); stencil.push_back(level);

  
  //loop over atoms
  for (unsigned int z=0; z<atomLocations.size(); z++){
    unsigned int Z=atomLocations[z][0];
    pcout << "Z:" << Z << std::endl;
    //load PSI radial value files
    if (radValues.count(Z)==0){
      char psiFile[256];
      sprintf(psiFile, "../../../data/electronicStructure/z%u/psi.inp", Z);
      std::vector<std::vector<double> > values;
      readFile(numPSIColumns, values, psiFile);
      //
      int numRows = values.size()-1;
      pcout << "psiRows: " << numRows << std::endl;

      std::vector<double> xData(numRows), yData(numRows);
      //x
      for(int irow = 0; irow < numRows; ++irow){
	xData[irow]= values[irow][0];
      }
      outerValues[Z] = xData[numRows-1];
      pcout << outerValues[Z] << std::endl;
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);	
      //y's
      for (unsigned int i=0; i<numPSIColumns-1; i++){
	for(int irow = 0; irow < numRows; ++irow){
	  yData[irow] = values[irow][i+1];
	}
	//
	alglib::real_1d_array y;
	y.setcontent(numRows,&yData[0]);
	alglib::ae_int_t natural_bound_type = 0;
	alglib::spline1dinterpolant* spline=new alglib::spline1dinterpolant;
	alglib::spline1dbuildcubic(x, y, numRows,
				   natural_bound_type,
				   0.0,
				   natural_bound_type,
				   0.0,
				   *spline);
	radValues[Z].push_back(spline);
      }
    }
    //check if additional wave functions requested
    unsigned int additionalLevels=0;
    if (additionalWaveFunctions.count(Z)!=0) {
      additionalLevels=additionalWaveFunctions[Z];
    } 
    unsigned int totalLevels=((unsigned int)std::ceil(Z/2.0))+additionalLevels;
    pcout << totalLevels << std::endl;
    //fill levels
    unsigned int levels=0;
    for (std::vector<std::vector<unsigned int> >::iterator it=stencil.begin(); it <stencil.end(); it++){
      unsigned int n=(*it)[0], l=(*it)[1];
      //m loop
      for (int m=-l; m<= (int) l; m++){
	orbital temp;
	temp.Z=Z; temp.n=n; temp.l=l; temp.m=m; temp.psiID=levels;
	//NOTE: change
	if (levels>2) temp.psiID=2;
	//
	waveFunctionsVector.push_back(temp); levels++;
	pcout << " n:" << n + 1 << " l:" << l << " m:" << m << " psi:" << temp.psiID << std::endl;
	if (levels>=totalLevels) break;
      }
      if (levels>=totalLevels) break;
    }
  }
}

//
void dftClass::readPSIRadialValues(){
  //loop over nodes to set PSI initial guess
  unsigned int dofs_per_cell = matrix_free_data.get_dofs_per_cell();
  //get support points
  std::map<types::global_dof_index, Point<3> > support_points;
  MappingQ<3> mapQ(1);
  DoFTools::map_dofs_to_support_points(mapQ, dofHandler, support_points); 
  IndexSet locallyOwnedSet;
  DoFTools::extract_locally_owned_dofs(dofHandler, locallyOwnedSet);
  std::vector<unsigned int> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  std::vector<std::vector<double> > local_dof_values(numEigenValues, std::vector<double>(locallyOwnedDOFs.size()));
  //loop over nodes
  for(unsigned int dof=0; dof<locallyOwnedDOFs.size(); dof++){
    unsigned int dofID= locallyOwnedDOFs[dof];
    Point<3> node = support_points[dofID];
    for(unsigned int atom=0; atom<atomLocations.size(); atom++){
      Point<3> atomCoord(atomLocations[atom][1],atomLocations[atom][2],atomLocations[atom][3]);
      //
      double x =node[0]-atomCoord[0];
      double y =node[1]-atomCoord[1];
      double z =node[2]-atomCoord[2];
      //
      double r = sqrt(x*x + y*y + z*z);
      double theta = acos(z/r);
      double phi = atan2(y,x);
      //
      if (r==0){theta=0; phi=0;}
      double R;
      unsigned int waveFunction=0;
      for (std::vector<orbital>::iterator it=waveFunctionsVector.begin(); it<waveFunctionsVector.end(); it++){
	R=0.0;
	if (r<=outerValues[it->Z]) R = alglib::spline1dcalc(*radValues[it->Z][it->psiID],r);
	//
	if (it->m >= 0){
	  local_dof_values[waveFunction][dof] =  R*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
	}
	else{
	  local_dof_values[waveFunction][dof] =  R*boost::math::spherical_harmonic_i(it->l,-(it->m),theta,phi);	  
	}
	waveFunction++;
      }
    }
  }
  //
  for (unsigned int i=0; i<eigenVectors.size(); ++i){
    constraintsNone.distribute_local_to_global(local_dof_values[i], locallyOwnedDOFs, *eigenVectors[i]);
  }
  //multiply by M^0.5
  for (unsigned int i=0; i<eigenVectors.size(); ++i){
    for (unsigned int j=0; j<eigenVectors[i]->local_size(); j++){
      if (std::abs(eigen.massVector.local_element(j))>1.0e-15){
	eigenVectors[i]->local_element(j)/=eigen.massVector.local_element(j);
      }
    }  
    eigenVectors[i]->update_ghost_values();
    pcout << "norm: " << eigenVectors[i]->l2_norm() << std::endl;
  }
}

//
void dftClass::readPSI(){
  computing_timer.enter_section("dftClass init PSI"); 
  //
  readPSIRadialValues();
  //
  computing_timer.exit_section("dftClass init PSI"); 
}
