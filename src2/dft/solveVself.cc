//source file for locating core atom nodes

void dftClass::solveVself()
{ 
  //
  //phiExt with nuclear charge
  //
  int numberBins = d_boundaryFlag.size();
  int numberGlobalCharges = atomLocations.size();
  
  //int constraintMatrixId = 2;
  //poisson.solve(poisson.phiExt,constraintMatrixId);
 

  poisson.phiExt = 0;

  pcout<<"Size of support points: "<<d_supportPoints.size()<<std::endl;

  std::map<dealii::types::global_dof_index, int>::iterator iterMap;
  for(int iBin = 0; iBin < numberBins; ++iBin)
    {
      int constraintMatrixId = iBin + 2;
      matrix_free_data.initialize_dof_vector(poisson.vselfBinScratch,constraintMatrixId);
      poisson.solve(poisson.vselfBinScratch,constraintMatrixId);

      std::set<int> & atomsInBinSet = d_bins[iBin];
      std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
      int numberGlobalAtomsInBin = atomsInCurrentBin.size();

      std::vector<int> & imageIdsOfAtomsInCurrentBin = d_imageIdsInBins[iBin];
      int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

      std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlag[iBin];
      std::map<types::global_dof_index,Point<3> >::iterator iterNodalCoorMap;

      int inNodes =0, outNodes = 0;
      for(iterNodalCoorMap = d_supportPoints.begin(); iterNodalCoorMap != d_supportPoints.end(); ++iterNodalCoorMap)
	{
	  if(poisson.vselfBinScratch.in_local_range(iterNodalCoorMap->first))
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
		      vSelf = poisson.vselfBinScratch(iterNodalCoorMap->first);
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
			  
			  if(isPseudopotential)
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

		  poisson.phiExt(iterNodalCoorMap->first)+= vSelf;

		}//charge loop
	    
	    }

	}//Vertexloop

      //
      //store Vselfs for atoms in bin
      //
      for(std::map<unsigned int, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
	{
	  std::vector<double> temp(2,0.0);
	  temp[0] = it->second;//charge;
	  temp[1] = poisson.vselfBinScratch(it->first);//vself 
	  std::cout<<"Peak Value of Vself: "<<temp[1]<<std::endl;
	  d_localVselfs.push_back(temp);
	}


      //   std::cout << "In: " << inNodes << "  Out: " << outNodes << "\n";
    }//bin loop

    poisson.phiExt.compress(VectorOperation::insert);
    poisson.phiExt.update_ghost_values();

    //
    //print the norms of phiExt (in periodic case L2 norm of phiExt field does not match. check later)
    //
    //pcout<<"Peak Value of phiext: "<<poisson.phiExt.linfty_norm()<<std::endl;
    pcout<<"L2 Norm Value of phiext: "<<poisson.phiExt.l2_norm()<<std::endl;
}
