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

//source file for locating core atom nodes
template<unsigned int FEOrder>
void dftClass<FEOrder>::solveVself()
{
  d_localVselfs.clear();
  d_vselfFieldBins.clear();    
  //phiExt with nuclear charge
  //
  int numberBins = d_boundaryFlag.size();
  int numberGlobalCharges = atomLocations.size();
  
  matrix_free_data.initialize_dof_vector(poissonPtr->phiExt,phiExtDofHandlerIndex);

  poissonPtr->phiExt = 0;

  //pcout<<"size of support points: "<<d_supportPoints.size()<<std::endl;

  std::map<dealii::types::global_dof_index, int>::iterator iterMap;
  d_vselfFieldBins.resize(numberBins);
  for(int iBin = 0; iBin < numberBins; ++iBin)
    {
      int constraintMatrixId = iBin + 2;
      matrix_free_data.initialize_dof_vector(poissonPtr->vselfBinScratch,constraintMatrixId);
      poissonPtr->vselfBinScratch = 0;

      std::map<types::global_dof_index,Point<3> >::iterator iterNodalCoorMap;
      std::map<dealii::types::global_dof_index, int> & vSelfBinNodeMap = d_vselfBinField[iBin];

      //
      //set initial guess to vSelfBinScratch
      //
      for(iterNodalCoorMap = d_supportPoints.begin(); iterNodalCoorMap != d_supportPoints.end(); ++iterNodalCoorMap)
	{
	  if(poissonPtr->vselfBinScratch.in_local_range(iterNodalCoorMap->first))
	    {
	      if(!d_noConstraints.is_constrained(iterNodalCoorMap->first))
		{
		  iterMap = vSelfBinNodeMap.find(iterNodalCoorMap->first);
		  if(iterMap != vSelfBinNodeMap.end())
		    {
		      poissonPtr->vselfBinScratch(iterNodalCoorMap->first) = iterMap->second;
		    }
		}
	      
	    }
	}
 
      poissonPtr->vselfBinScratch.compress(VectorOperation::insert);
      poissonPtr->vselfBinScratch.update_ghost_values();
      d_constraintsVector[constraintMatrixId]->distribute(poissonPtr->vselfBinScratch);
      //
      //call the poisson solver to compute vSelf in each bin
      //
      poissonPtr->solve(poissonPtr->vselfBinScratch,constraintMatrixId);

      std::set<int> & atomsInBinSet = d_bins[iBin];
      std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
      int numberGlobalAtomsInBin = atomsInCurrentBin.size();

      std::vector<int> & imageIdsOfAtomsInCurrentBin = d_imageIdsInBins[iBin];
      int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

      std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlag[iBin];
       

     

      int inNodes =0, outNodes = 0;
      for(iterNodalCoorMap = d_supportPoints.begin(); iterNodalCoorMap != d_supportPoints.end(); ++iterNodalCoorMap)
	{
	  if(poissonPtr->vselfBinScratch.in_local_range(iterNodalCoorMap->first))
	    {
	      if(!d_noConstraints.is_constrained(iterNodalCoorMap->first))
		{
		  //
		  //get the vertex Id
		  //
		  Point<3> nodalCoor = iterNodalCoorMap->second;

		  //
		  //get the boundary flag for iVertex for current bin
		  //
		  int boundaryFlag;
		  iterMap = boundaryNodeMap.find(iterNodalCoorMap->first);
		  if(iterMap != boundaryNodeMap.end())
		    {
		      boundaryFlag = iterMap->second;
		    }
		  else
		    {
		      std::cout<<"Could not find boundaryNode Map for the given dof:"<<std::endl;
		      exit(-1);
		    }

		  //
		  //go through all atoms in a given bin
		  //
		  for(int iCharge = 0; iCharge < numberGlobalAtomsInBin+numberImageAtomsInBin; ++iCharge)
		    {
		      //
		      //get the globalChargeId corresponding to iCharge in the current bin
		      //and add numberGlobalCharges to image atomId
		      int chargeId;
		      if(iCharge < numberGlobalAtomsInBin)
			chargeId = atomsInCurrentBin[iCharge];
		      else
			chargeId = imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]+numberGlobalCharges;

		      //std::cout<<"Charge Id in BinId: "<<chargeId<<" "<<iBin<<std::endl;

		  
		      double vSelf;
		      if(boundaryFlag == chargeId)
			{
			  vSelf = poissonPtr->vselfBinScratch(iterNodalCoorMap->first);
			  inNodes++;
			}
		      else
			{
			  Point<3> atomCoor(0.0,0.0,0.0);
			  double nuclearCharge;
			  if(iCharge < numberGlobalAtomsInBin)
			    {
			      atomCoor[0] = atomLocations[chargeId][2];
			      atomCoor[1] = atomLocations[chargeId][3];
			      atomCoor[2] = atomLocations[chargeId][4];
			  
			      if(dftParameters::isPseudopotential)
				nuclearCharge = atomLocations[chargeId][1];
			      else
				nuclearCharge = atomLocations[chargeId][0];
			  
			    }
			  else
			    {
			      atomCoor[0] = d_imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][0];
			      atomCoor[1] = d_imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][1];
			      atomCoor[2] = d_imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][2];
			      nuclearCharge = d_imageCharges[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]];

			    }

			  const double r = nodalCoor.distance(atomCoor);
			  vSelf = -nuclearCharge/r;
			  outNodes++;
			}

		      //store updated value in phiExt which is sumVself

		      poissonPtr->phiExt(iterNodalCoorMap->first)+= vSelf;

		    }//charge loop

		}//non-hanging node check
	    
	    }//local range loop

	}//Vertexloop

      //
      //store Vselfs for atoms in bin
      //
      for(std::map<unsigned int, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
	{
	  std::vector<double> temp(2,0.0);
	  temp[0] = it->second;//charge;
	  temp[1] = poissonPtr->vselfBinScratch(it->first);//vself 
	  std::cout<< "(only for debugging: peak value of Vself: "<< temp[1] << ")" <<std::endl;
	  d_localVselfs.push_back(temp);
	}
        //
        //store solved vselfBinScratch field
        //
        d_vselfFieldBins[iBin]=poissonPtr->vselfBinScratch;
    }//bin loop

  poissonPtr->phiExt.compress(VectorOperation::insert);
  poissonPtr->phiExt.update_ghost_values();
  d_constraintsVector[phiExtDofHandlerIndex]->distribute(poissonPtr->phiExt); 
  poissonPtr->phiExt.update_ghost_values();
  //
  //print the norms of phiExt (in periodic case L2 norm of phiExt field does not match. check later)
  //
  pcout<<"L2 Norm Value of phiext: "<<poissonPtr->phiExt.l2_norm()<<std::endl;
}
