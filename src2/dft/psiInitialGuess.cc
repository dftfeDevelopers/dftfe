#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>

//load PSI radial value files
void dftClass::loadPSIFiles(unsigned int Z, unsigned int n, unsigned int l){
  //
  if (radValues[Z][n].count(l)>0) return;
  //
  char psiFile[256];
  if(isPseudopotential)
    sprintf(psiFile, "../../../data/electronicStructure/pseudoPotential/z%u/psi%u%u.inp", Z, n, l);
  else
    sprintf(psiFile, "../../../data/electronicStructure/allElectron/z%u/psi%u%u.inp", Z, n, l);
  std::vector<std::vector<double> > values;
  readFile(2, values, psiFile);
  //
  int numRows = values.size()-1;
  std::vector<double> xData(numRows), yData(numRows);
  //x
  for(int irow = 0; irow < numRows; ++irow){
    xData[irow]= values[irow][0];
  }
  outerValues[Z][n][l] = xData[numRows-1];
  alglib::real_1d_array x;
  x.setcontent(numRows,&xData[0]);	
  //y
  for(int irow = 0; irow < numRows; ++irow){
    yData[irow] = values[irow][1];
  }
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
  //pcout << "send: Z:" << Z << " n:" << n << " l:" << l << " numRows:" << numRows << std::endl; 
  radValues[Z][n][l]=spline;
}

//
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
  //
  
  //loop over atoms
  for (unsigned int z=0; z<atomLocations.size(); z++){
    unsigned int Z=atomLocations[z][0];
        
    //check if additional wave functions requested
    unsigned int additionalLevels=0;
    if (additionalWaveFunctions.count(Z)!=0) {
      additionalLevels=additionalWaveFunctions[Z];
    } 
    unsigned int totalLevels=((unsigned int)std::ceil(Z/2.0))+additionalLevels;
    numElectrons+=Z;
    numBaseLevels+=(unsigned int)std::ceil(Z/2.0);
    numLevels+=totalLevels;
    
    //fill levels
    bool printLevels=false;
    if (radValues.count(Z)==0){
      printLevels=true;
      pcout << "Z:" << Z << std::endl;
    }
    unsigned int levels=0;
    for (std::vector<std::vector<unsigned int> >::iterator it=stencil.begin(); it <stencil.end(); it++){
      unsigned int n=(*it)[0], l=(*it)[1];
      //load PSI files
      loadPSIFiles(Z, n, l);
      //m loop
      for (int m=-l; m<= (int) l; m++){
	orbital temp;
	temp.atomID=z;
	temp.Z=Z; temp.n=n; temp.l=l; temp.m=m; temp.psi=radValues[Z][n][l];
	waveFunctionsVector.push_back(temp); levels++;
	if (printLevels) pcout << " n:" << n  << " l:" << l << " m:" << m << std::endl;
	if (levels>=totalLevels) break;
      }
      if (levels>=totalLevels) break;
    }
  }
  pcout << "total num electrons: " << numElectrons << std::endl;
  pcout << "total num base levels: " << numBaseLevels << std::endl;
  pcout << "total num levels: " << numLevels << std::endl;
  pcout << "************************************" << std::endl;
}

//
void dftClass::readPSIRadialValues(){
  //loop over nodes to set PSI initial guess
  //get support points
  std::map<types::global_dof_index, Point<3> > support_points;
  MappingQ<3> mapQ(1);
  DoFTools::map_dofs_to_support_points(mapQ, dofHandler, support_points); 
  IndexSet locallyOwnedSet;
  DoFTools::extract_locally_owned_dofs(dofHandler, locallyOwnedSet);
  std::vector<unsigned int> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  std::vector<std::vector<double> > local_dof_values(numEigenValues, std::vector<double>(locallyOwnedDOFs.size(), 0.0));
  //loop over nodes
  bool pp=false;
  for(unsigned int dof=0; dof<locallyOwnedDOFs.size(); dof++){
    unsigned int dofID= locallyOwnedDOFs[dof];
    Point<3> node = support_points[dofID];
    //loop over wave functions
    unsigned int waveFunction=0;
    for (std::vector<orbital>::iterator it=waveFunctionsVector.begin(); it<waveFunctionsVector.end(); it++){
      //find coordinates of atom correspoding to this wave function

      Point<3> atomCoord(atomLocations[it->atomID][2],atomLocations[it->atomID][3],atomLocations[it->atomID][4]);
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
      //radial part
      double R=0.0;
      if (r<=outerValues[it->Z][it->n][it->l]) R = alglib::spline1dcalc(*(it->psi),r);
      if (!pp){
	//pcout << "atom: " << it->atomID << " Z:" << it->Z << " n:" << it->n << " l:" << it->l << " m:" << it->m << " x:" << atomCoord[0] << " y:" << atomCoord[1] << " z:" << atomCoord[2] << " Ro:" << outerValues[it->Z][it->n][it->l] << std::endl; 
      }
      //spherical part
      if (it->m > 0){
	local_dof_values[waveFunction][dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
      }
      else if (it->m == 0){
	local_dof_values[waveFunction][dof] =  R*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
      }
      else{
	local_dof_values[waveFunction][dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_i(it->l,-(it->m),theta,phi);	  
      }
      waveFunction++;
    }
    pp=true;
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
    char buffer[100];
    sprintf(buffer, "norm %u: l1: %14.8e  l2:%14.8e\n",i, eigenVectors[i]->l1_norm(), eigenVectors[i]->l2_norm());
    pcout << buffer;
    eigenVectors[i]->update_ghost_values();
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
