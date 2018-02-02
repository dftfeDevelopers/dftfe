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

#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>

#include "../../include/dftParameters.h"

using namespace dftParameters ;

template<unsigned int FEOrder>
void dftClass<FEOrder>::loadPSIFiles(unsigned int Z, 
				     unsigned int n, 
				     unsigned int l,
				     unsigned int &fileReadFlag)
{

  if (radValues[Z][n].count(l) > 0) 
    {
      fileReadFlag = 1;
      return;
    }


  //
  //set the paths for the Single-Atom wavefunction data
  //
  char psiFile[256];

  if(isPseudopotential)
    if(pseudoProjector==2)
	sprintf(psiFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/singleAtomData/psi%u%u.inp", dftParameters::currentPath.c_str(), Z, n, l);
    else
        sprintf(psiFile, "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp", dftParameters::currentPath.c_str(), Z, n, l);

  else
    sprintf(psiFile, "%s/data/electronicStructure/allElectron/z%u/singleAtomData/psi%u%u.inp", dftParameters::currentPath.c_str(), Z, n, l);

  std::vector<std::vector<double> > values;

  fileReadFlag = dftUtils::readPsiFile(2, values, psiFile);
  

  //
  //spline fitting for single-atom wavefunctions
  //
  if(fileReadFlag > 0)
    {
      pcout<<"reading data from file: "<<psiFile<<std::endl;
      
      int numRows = values.size()-1;
      std::vector<double> xData(numRows), yData(numRows);

      //x
      for(int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow]= values[irow][0];
	}
      outerValues[Z][n][l] = xData[numRows-1];
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);	

      //y
      for(int irow = 0; irow < numRows; ++irow)
	{
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

      radValues[Z][n][l]=spline;
    }
}

//
//determine orbital ordering
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::determineOrbitalFilling()
{
  //
  //create a stencil following orbital filling order
  //
  std::vector<unsigned int> level;
  std::vector<std::vector<unsigned int> > stencil;

  //1s
  level.clear(); level.push_back(1); level.push_back(0); stencil.push_back(level);
  //2s
  level.clear(); level.push_back(2); level.push_back(0); stencil.push_back(level);
  //2p
  level.clear(); level.push_back(2); level.push_back(1); stencil.push_back(level);
  //3s
  level.clear(); level.push_back(3); level.push_back(0); stencil.push_back(level);
  //3p
  level.clear(); level.push_back(3); level.push_back(1); stencil.push_back(level);
  //4s
  level.clear(); level.push_back(4); level.push_back(0); stencil.push_back(level);
  //3d
  level.clear(); level.push_back(3); level.push_back(2); stencil.push_back(level);
  //4p
  level.clear(); level.push_back(4); level.push_back(1); stencil.push_back(level);
  //5s
  level.clear(); level.push_back(5); level.push_back(0); stencil.push_back(level);
  //4d
  level.clear(); level.push_back(4); level.push_back(2); stencil.push_back(level);
  //5p
  level.clear(); level.push_back(5); level.push_back(1); stencil.push_back(level);
  //6s
  level.clear(); level.push_back(6); level.push_back(0); stencil.push_back(level);
  //4f
  level.clear(); level.push_back(4); level.push_back(3); stencil.push_back(level);
  //5d
  level.clear(); level.push_back(5); level.push_back(2); stencil.push_back(level);
  //6p
  level.clear(); level.push_back(6); level.push_back(1); stencil.push_back(level);
  //7s
  level.clear(); level.push_back(7); level.push_back(0); stencil.push_back(level);
  //5f
  level.clear(); level.push_back(5); level.push_back(3); stencil.push_back(level);
  //6d
  level.clear(); level.push_back(6); level.push_back(2); stencil.push_back(level);
  //7p
  level.clear(); level.push_back(7); level.push_back(1); stencil.push_back(level);
  //8s
  level.clear(); level.push_back(8); level.push_back(0); stencil.push_back(level);
  
  int totalNumberWaveFunctions = numEigenValues;
  unsigned int fileReadFlag = 0;
  unsigned int waveFunctionCount = 0;
  //unsigned int extraWaveFunctionsPerAtom = 0;
  unsigned int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = d_imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  unsigned int errorReadFile = 0;
  //std::vector<unsigned int> numberAtomicWaveFunctions(numberAtoms,0.0);
  //std::vector<unsigned int> levels(numberAtoms,0.0);

  //
  //loop over atoms
  //
  for(unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
    {
      unsigned int Z = atomLocations[iAtom][0];
      unsigned int valenceZ = atomLocations[iAtom][1];
      unsigned int numberAtomFunctions; 

      if(dftParameters::isPseudopotential)
	{
	  //numberAtomFunctions = std::ceil(valenceZ/2.0) + extraWaveFunctionsPerAtom;
	  numElectrons += valenceZ;
	}
      else
	{
	  //numberAtomFunctions = std::ceil(Z/2.0) + extraWaveFunctionsPerAtom;
	  numElectrons += Z;
	}

      //numberAtomicWaveFunctions[iAtom] = numberAtomFunctions;
    }


  for (std::vector<std::vector<unsigned int> >::iterator it = stencil.begin(); it < stencil.end(); it++)
    {
      unsigned int n = (*it)[0], l = (*it)[1];

      for (int m = -l; m <= (int) l; m++)
	{
	  for(unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
	    {
	      unsigned int Z = atomLocations[iAtom][0];

	      //
	      //fill levels
	      //
	      if(radValues.count(Z)==0)
		{
		  pcout << "Z:" << Z << std::endl;
		}
	      
	      //
	      //load PSI files
	      //
	      loadPSIFiles(Z, n, l,fileReadFlag);

	      if(fileReadFlag > 0)
		{
		  orbital temp;
		  temp.atomID = iAtom;
		  temp.Z = Z; temp.n = n; temp.l = l; temp.m = m; temp.psi = radValues[Z][n][l];
		  waveFunctionsVector.push_back(temp); waveFunctionCount++;
		  if(waveFunctionCount >= numEigenValues && waveFunctionCount >= numberGlobalAtoms) break;
		}
	      
	    }

	  if(waveFunctionCount >= numEigenValues && waveFunctionCount >= numberGlobalAtoms) break;
	}

      if(waveFunctionCount >= numEigenValues && waveFunctionCount >= numberGlobalAtoms) break;

      if(fileReadFlag == 0)
	errorReadFile += 1;
    }

  if(errorReadFile == stencil.size())
    {
      std::cerr<< "Error: Require single-atom wavefunctions as initial guess for starting the SCF."<< std::endl;
      std::cerr<< "Error: Could not find single-atom wavefunctions for any atom: "<< std::endl;
      exit(-1);
    }
  
  if(waveFunctionsVector.size() > numEigenValues)
    {
      numEigenValues = waveFunctionsVector.size();
    }
  

  pcout<<"============================================================================================================================="<<std::endl;
  pcout<<"number of electrons: "<<numElectrons<<std::endl;
  pcout<<"number of wavefunctions computed using single atom data to be used as initial guess for starting the SCF: " <<waveFunctionCount<<std::endl;
  pcout<<"============================================================================================================================="<<std::endl;
}

//
template<unsigned int FEOrder>
void dftClass<FEOrder>::readPSIRadialValues(){
 
  IndexSet locallyOwnedSet;
  DoFTools::extract_locally_owned_dofs(dofHandlerEigen, locallyOwnedSet);
  std::vector<unsigned int> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);


#ifdef ENABLE_PERIODIC_BC
  unsigned int numberDofs = locallyOwnedDOFs.size()/2;
#else
  unsigned int numberDofs = locallyOwnedDOFs.size();
#endif

  std::vector<std::vector<double> > local_dof_values(numEigenValues, std::vector<double>(numberDofs, 0.0));
  unsigned int numberGlobalAtoms = atomLocations.size();

  
  //
  //loop over nodes
  //
  bool pp=false;
  for(unsigned int dof=0; dof<numberDofs; dof++)
    {
#ifdef ENABLE_PERIODIC_BC
      unsigned int dofID = local_dof_indicesReal[dof];
#else
      unsigned int dofID = locallyOwnedDOFs[dof];
#endif
      Point<3> node = d_supportPointsEigen[dofID];

      //
      //loop over wave functions
      //
      unsigned int waveFunction=0;
      for (std::vector<orbital>::iterator it = waveFunctionsVector.begin(); it < waveFunctionsVector.end(); it++)
	{

	  //get the imageIdmap information corresponding to globalChargeId
	  //
#ifdef ENABLE_PERIODIC_BC
	  std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMap[it->atomID];
#else
	  std::vector<int> imageIdsList;
	  imageIdsList.push_back(it->atomID);
#endif

	  for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size();++iImageAtomCount)
	    {
	  
	      //
	      //find coordinates of atom correspoding to this wave function and imageAtom
	      //
	      int chargeId = imageIdsList[iImageAtomCount];
	      Point<3> atomCoord;

	      if(chargeId < numberGlobalAtoms)
		{
		  atomCoord[0] = atomLocations[chargeId][2];
		  atomCoord[1] = atomLocations[chargeId][3];
		  atomCoord[2] = atomLocations[chargeId][4];
		}
	      else
		{
		  atomCoord[0] = d_imagePositions[chargeId-numberGlobalAtoms][0];
		  atomCoord[1] = d_imagePositions[chargeId-numberGlobalAtoms][1];
		  atomCoord[2] = d_imagePositions[chargeId-numberGlobalAtoms][2];
		}
	  
	      double x = node[0]-atomCoord[0];
	      double y = node[1]-atomCoord[1];
	      double z = node[2]-atomCoord[2];


	      double r = sqrt(x*x + y*y + z*z);
	      double theta = acos(z/r);
	      double phi = atan2(y,x);


	      if (r==0){theta=0; phi=0;}
	      //radial part
	      double R=0.0;
	      if (r<=outerValues[it->Z][it->n][it->l]) R = alglib::spline1dcalc(*(it->psi),r);
	      //spherical part
	      if (it->m > 0)
		{
		  local_dof_values[waveFunction][dof] +=  R*std::sqrt(2)*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
		}
	      else if (it->m == 0)
		{
		  local_dof_values[waveFunction][dof] +=  R*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
		}
	      else
		{
		  local_dof_values[waveFunction][dof] +=  R*std::sqrt(2)*boost::math::spherical_harmonic_i(it->l,-(it->m),theta,phi);	  
		}
	    }
	  waveFunction++;
	}
    }

  if(waveFunctionsVector.size() < numEigenValues)
    {

      unsigned int nonAtomicWaveFunctions = numEigenValues - waveFunctionsVector.size();
      pcout << "                                                                                             "<<std::endl;
      pcout << "number of wavefunctions generated randomly to be used as initial guess for starting the SCF : " << nonAtomicWaveFunctions << std::endl;

      //
      // assign the rest of the wavefunctions using a standard normal distribution
      //
      boost::math::normal normDist;

      for(unsigned int iWave = waveFunctionsVector.size(); iWave < numEigenValues; ++iWave)
	{
	  for(unsigned int dof=0; dof<numberDofs; dof++)
	    {
	      double z = (-0.5 + (rand()+ 0.0)/(RAND_MAX))*3.0;
	      double value =  boost::math::pdf(normDist, z); 
	      if(rand()%2 == 0)
		value = -1.0*value;
	      //#ifdef ENABLE_PERIODIC_BC
	      //local_dof_values[iWave][localProc_dof_indicesReal[dof]] = value;
	      //#else
	      local_dof_values[iWave][dof] = value;
	      //#endif
	      
	    }
	}

    }

  //char buffer[100];
  for(int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      //sprintf(buffer, "%10u%10u\n", eigenVectors[kPoint].size(), local_dof_values.size() );
      //pcout << buffer;
      for (unsigned int i = 0; i < numEigenValues; ++i)
	{
	  //constraintsNoneEigen.distribute_local_to_global(local_dof_values[i], locallyOwnedDOFs, *eigenVectors[kPoint][i]);
	  //eigenVectors[kPoint][i]->compress(VectorOperation::add);
#ifdef ENABLE_PERIODIC_BC
	 for(unsigned int j = 0; j < numberDofs; ++j)
	    {
	      unsigned int dofID = local_dof_indicesReal[j];
	      if(eigenVectors[kPoint][i]->in_local_range(dofID))
		{
		  if(!constraintsNoneEigen.is_constrained(dofID))
		    (*eigenVectors[kPoint][i])(dofID) = local_dof_values[i][j];
		}
	    }
#else
	  for(unsigned int j = 0; j < numberDofs; ++j)
	    {
	      unsigned int dofID = locallyOwnedDOFs[j];
	      if(eigenVectors[kPoint][i]->in_local_range(dofID))
		{
		  if(!constraintsNoneEigen.is_constrained(dofID))
		    (*eigenVectors[kPoint][i])(dofID) = local_dof_values[i][j];
		}
	    }
#endif
	  eigenVectors[kPoint][i]->compress(VectorOperation::insert);
	  eigenVectors[kPoint][i]->update_ghost_values();
	}
    // pcout<<"check 0.13: "<<std::endl;
    }
  
   // pcout<<"check 1: "<<std::endl;

  //
  //multiply by M^0.5
  //
  for(int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  /*for (unsigned int j = 0; j < eigenVectors[kPoint][i]->local_size(); j++)
	    {
	      if (std::abs(eigenPtr->massVector.local_element(j))>1.0e-15)
		{
		  eigenVectors[kPoint][i]->local_element(j)/=eigenPtr->massVector.local_element(j);
		}
		}*/
	  for(types::global_dof_index j = 0; j < eigenVectors[kPoint][i]->size(); ++j)
	     {
	       if(eigenVectors[kPoint][i]->in_local_range(j))
		  {
		    if(!constraintsNoneEigen.is_constrained(j) && std::abs(eigenPtr->massVector(j))>1.0e-15)
		      (*eigenVectors[kPoint][i])(j) /= eigenPtr->massVector(j);
		  }
	     }


	  char buffer[100];
	  sprintf(buffer, "norm %u: l1: %14.8e  l2:%14.8e\n",i, eigenVectors[kPoint][i]->l1_norm(), eigenVectors[kPoint][i]->l2_norm());
	  //char buffer[100];
	  //sprintf(buffer, "norm %u: l1: %14.8e  l2:%14.8e\n",i, eigenVectors[kPoint][i]->l1_norm(), eigenVectors[kPoint][i]->l2_norm());
	  //pcout << buffer;
	  eigenVectors[kPoint][i]->zero_out_ghosts();
	  eigenVectors[kPoint][i]->compress(VectorOperation::insert);
	  constraintsNoneEigen.distribute(*eigenVectors[kPoint][i]);
	  eigenVectors[kPoint][i]->update_ghost_values();
	}
    }

   //pcout<<"check 2: "<<std::endl;

}

//
template<unsigned int FEOrder>
void dftClass<FEOrder>::readPSI(){
  computing_timer.enter_section("initialize wave functions"); 

  readPSIRadialValues();

  computing_timer.exit_section("initialize wave functions"); 
}
