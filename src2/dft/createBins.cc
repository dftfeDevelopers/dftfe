//source file for locating core atom nodes

void exchangeAtomToGlobalNodeIdMaps(const int totalNumberAtoms,
				    std::map<int,std::set<int> > & atomToGlobalNodeIdMap,
				    unsigned int numMeshPartitions,
				    const MPI_Comm & mpi_communicator)

    {

      
      std::map<int,std::set<int> >::iterator iter;

      for(int iGlobal = 0; iGlobal < totalNumberAtoms; ++iGlobal){

	//
	// for each charge, exchange its global list across all procs
	//
	iter = atomToGlobalNodeIdMap.find(iGlobal);    

	std::vector<int> localAtomToGlobalNodeIdList;

	if(iter != atomToGlobalNodeIdMap.end()){
	  std::set<int>  & localGlobalNodeIdSet = iter->second;
	  std::copy(localGlobalNodeIdSet.begin(),
		    localGlobalNodeIdSet.end(),
		    std::back_inserter(localAtomToGlobalNodeIdList));

	}

	int numberGlobalNodeIdsOnLocalProc = localAtomToGlobalNodeIdList.size();

	int * atomToGlobalNodeIdListSizes = new int[numMeshPartitions];

	MPI_Allgather(&numberGlobalNodeIdsOnLocalProc,
		      1,
		      MPI_INT,
		      atomToGlobalNodeIdListSizes,
		      1,
		      MPI_INT,
		      mpi_communicator);

	int newAtomToGlobalNodeIdListSize = 
	  std::accumulate(&(atomToGlobalNodeIdListSizes[0]),
			  &(atomToGlobalNodeIdListSizes[numMeshPartitions]),
			  0);

	std::vector<int> globalAtomToGlobalNodeIdList(newAtomToGlobalNodeIdListSize);
    
	int * mpiOffsets = new int[numMeshPartitions];

	mpiOffsets[0] = 0;

	for(int i = 1; i < numMeshPartitions; ++i)
	  mpiOffsets[i] = atomToGlobalNodeIdListSizes[i-1]+ mpiOffsets[i-1];

	MPI_Allgatherv(&(localAtomToGlobalNodeIdList[0]),
		       numberGlobalNodeIdsOnLocalProc,
		       MPI_INT,
		       &(globalAtomToGlobalNodeIdList[0]),
		       &(atomToGlobalNodeIdListSizes[0]),
		       &(mpiOffsets[0]),
		       MPI_INT,
		       mpi_communicator);

	//
	// over-write local interaction with items of globalInteractionList
	//
	for(int i = 0 ; i < globalAtomToGlobalNodeIdList.size(); ++i)
	  (atomToGlobalNodeIdMap[iGlobal]).insert(globalAtomToGlobalNodeIdList[i]);

	delete [] atomToGlobalNodeIdListSizes;
	delete [] mpiOffsets;

      }
      return;
 
    }


void dftClass::createAtomBins(std::vector<const ConstraintMatrix * > & constraintsVector)
			     
{

  //
  // access complete list of image charges
  //
  std::vector<int> imageIdsContainer;
  std::vector<std::vector<double > > imagePositions;
  std::vector<double> imageChargeValues;

  imageIdsContainer = d_imageIds;
  imagePositions = d_imagePositions;
  imageChargeValues = d_imageCharges;

  const int numberImageCharges = imageIdsContainer.size();

  pcout<<"Number Image Charges: "<<numberImageCharges<<std::endl;

  const int numberGlobalAtoms = atomLocations.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  

  std::map<int,std::set<int> > atomToGlobalNodeIdMap;

  for(unsigned int iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
    {
      std::set<int> tempNodalSet;
      Point<3> atomCoor;

      if(iAtom < numberGlobalAtoms)
	{
	  atomCoor[0] = atomLocations[iAtom][2];
	  atomCoor[1] = atomLocations[iAtom][3];
	  atomCoor[2] = atomLocations[iAtom][4];
	}
      else
	{
	  //
	  //Fill with ImageAtom Coors
	  //
	  atomCoor[0] = imagePositions[iAtom-numberGlobalAtoms][0];
	  atomCoor[1] = imagePositions[iAtom-numberGlobalAtoms][1];
	  atomCoor[2] = imagePositions[iAtom-numberGlobalAtoms][2];
	}

      // std::cout<<"Atom Coor: "<<atomCoor[0]<<" "<<atomCoor[1]<<" "<<atomCoor[2]<<std::endl;

      DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),endc = dofHandler.end();

      for(; cell!= endc; ++cell)
	{
	  if(cell->is_locally_owned())
	    {
	      int cutOffFlag = 0;
	      for(unsigned int iNode = 0; iNode < vertices_per_cell; ++iNode)
		{
		  
		  Point<3> feNodeGlobalCoord = cell->vertex(iNode);

		  double distance = atomCoor.distance(feNodeGlobalCoord);

		  if(distance < radiusAtomBall)
		    {
		      cutOffFlag = 1;
		      break;
		    }
		  
		}//element node loop

	      if(cutOffFlag == 1)
		{
		  for(unsigned int iNode = 0; iNode < vertices_per_cell; ++iNode)
		    {
		      unsigned int nodeID=cell->vertex_dof_index(iNode,0);
		      tempNodalSet.insert(nodeID);
		    }

		}

	    }//cell locally owned if loop

	}//cell or element loop

      atomToGlobalNodeIdMap[iAtom] = tempNodalSet;

    }//atom loop
 
  //
  //exchange atomToGlobalNodeIdMap across all processors
  //
  exchangeAtomToGlobalNodeIdMaps(totalNumberAtoms,
				 atomToGlobalNodeIdMap,
				 n_mpi_processes,
				 mpi_communicator);

  
  //
  //erase keys which have empty values
  //
  for(int iAtom = numberGlobalAtoms; iAtom < totalNumberAtoms; ++iAtom)
    {
      if(atomToGlobalNodeIdMap[iAtom].empty() == true)
	{
	  atomToGlobalNodeIdMap.erase(iAtom);
	}
    }


  //
  //create interaction maps by finding the intersection of global NodeIds of each atom 
  //  
  std::map<int,std::set<int> > interactionMap;

  for(int iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
    {
      //
      //Add iAtom to the interactionMap corresponding to the key iAtom
      //
      if(iAtom < numberGlobalAtoms)
	interactionMap[iAtom].insert(iAtom);
	  
      //std::cout<<"IAtom: "<<iAtom<<std::endl;

      for(int jAtom = iAtom - 1; jAtom > -1; jAtom--)
	{
	  //std::cout<<"JAtom: "<<jAtom<<std::endl;
	      
	  //
	  //compute intersection between the atomGlobalNodeIdMap of iAtom and jAtom
	  //
	  std::vector<int> nodesIntersection;

	  std::set_intersection(atomToGlobalNodeIdMap[iAtom].begin(),
				atomToGlobalNodeIdMap[iAtom].end(),
				atomToGlobalNodeIdMap[jAtom].begin(),
				atomToGlobalNodeIdMap[jAtom].end(),
				std::back_inserter(nodesIntersection));

	  // std::cout<<"Size of NodeIntersection: "<<nodesIntersection.size()<<std::endl;

	  if(nodesIntersection.size() > 0)
	    {
	      if(iAtom < numberGlobalAtoms && jAtom < numberGlobalAtoms)
		{
		  //
		  //if both iAtom and jAtom are actual atoms in unit-cell/domain,then iAtom and jAtom are interacting atoms
		  //
		  interactionMap[iAtom].insert(jAtom);
		  interactionMap[jAtom].insert(iAtom);
		}
	      else if(iAtom < numberGlobalAtoms && jAtom >= numberGlobalAtoms)
		{
		  //
		  //if iAtom is actual atom in unit-cell and jAtom is imageAtom, find the actual atom for which jAtom is
		  //the image then create the interaction map between that atom and iAtom
		  //
		  int masterAtomId = imageIdsContainer[jAtom - numberGlobalAtoms];
		  if(masterAtomId == iAtom)
		    {
		      std::cout<<"Atom and its own image is interacting decrease radius"<<std::endl;
		      exit(-1);
		    }
		  interactionMap[iAtom].insert(masterAtomId);
		  interactionMap[masterAtomId].insert(iAtom);
		}
	      else if(iAtom >= numberGlobalAtoms && jAtom < numberGlobalAtoms)
		{
		  //
		  //if jAtom is actual atom in unit-cell and iAtom is imageAtom, find the actual atom for which iAtom is
		  //the image and then create interaction map between that atom and jAtom
		  //
		  int masterAtomId = imageIdsContainer[iAtom - numberGlobalAtoms];
		  if(masterAtomId == jAtom)
		    {
		      std::cout<<"Atom and its own image is interacting decrease radius"<<std::endl;
		      exit(-1);
		    }
		  interactionMap[masterAtomId].insert(jAtom);
		  interactionMap[jAtom].insert(masterAtomId);

		}
	      else if(iAtom >= numberGlobalAtoms && jAtom >= numberGlobalAtoms)
		{
		  //
		  //if both iAtom and jAtom are image atoms in unit-cell iAtom and jAtom are interacting atoms
		  //find the actual atoms for which iAtom and jAtoms are images and create interacting maps between them
		  int masteriAtomId = imageIdsContainer[iAtom - numberGlobalAtoms];
		  int masterjAtomId = imageIdsContainer[jAtom - numberGlobalAtoms];
		  if(masteriAtomId == masterjAtomId)
		    {
		      std::cout<<"Two Image Atoms corresponding to same parent Atoms are interacting decrease radius"<<std::endl;
		      exit(-1);
		    }
		  interactionMap[masteriAtomId].insert(masterjAtomId);
		  interactionMap[masterjAtomId].insert(masteriAtomId);
		}

	    }

	}//end of jAtom loop

    }//end of iAtom loop

  std::map<int,std::set<int> >::iterator iter;

  //
  // start by adding atom 0 to bin 0
  //
  (d_bins[0]).insert(0);
  int binCount = 0;
  // iterate from atom 1 onwards
  for(int i = 1; i < numberGlobalAtoms; ++i){

    const std::set<int> & interactingAtoms = interactionMap[i];
    //
    //treat spl case when no atom intersects with another. e.g. simple cubic
    //
    if(interactingAtoms.size() == 0){
      (d_bins[binCount]).insert(i);
      continue;
    }

    bool isBinFound;
    // iterate over each existing bin and see if atom i fits into the bin
    for(iter = d_bins.begin();iter!= d_bins.end();++iter){

      // pick out atoms in this bin
      std::set<int>& atomsInThisBin = iter->second;      
      int index = std::distance(d_bins.begin(),iter);
	
      isBinFound = true;

      // to belong to this bin, this atom must not overlap with any other
      // atom already present in this bin
      for(std::set<int>::iterator iter2 = interactingAtoms.begin(); iter2!= interactingAtoms.end();++iter2){

	int atom = *iter2;
	
	if(atomsInThisBin.find(atom) != atomsInThisBin.end()){
	  isBinFound = false;
	  break;
	}
      }

      if(isBinFound == true){
	(d_bins[index]).insert(i);
	break;
      }            
    }
    // if all current bins have been iterated over w/o a match then
    // create a new bin for this atom
    if(isBinFound == false){
      binCount++;
      (d_bins[binCount]).insert(i);
    }
  }


  const int numberBins = binCount + 1;
  pcout<<"Number Bins: "<<numberBins<<std::endl;

  //std::vector<std::vector<int> > imageIdsInBins;
  d_imageIdsInBins.resize(numberBins);
  d_boundaryFlag.resize(numberBins);
  
  //
  //set constraint matrices for each bin
  //
  for(int iBin = 0; iBin < numberBins; ++iBin)
    {
      std::set<int> & atomsInBinSet = d_bins[iBin];
      std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
      std::vector<Point<3> > atomPositionsInCurrentBin;

      int numberGlobalAtomsInBin = atomsInCurrentBin.size();

      std::vector<int> &imageIdsOfAtomsInCurrentBin = d_imageIdsInBins[iBin];
      std::vector<std::vector<double> > imagePositionsOfAtomsInCurrentBin;

      pcout<<"Bin: "<<iBin<<" Number of Global Atoms: "<<numberGlobalAtomsInBin<<std::endl;

      for(int index = 0; index < numberGlobalAtomsInBin; ++index)
	{
	  int globalChargeIdInCurrentBin = atomsInCurrentBin[index];

	  //std:cout<<"Index: "<<index<<"Global Charge Id: "<<globalChargeIdInCurrentBin<<std::endl;

	  Point<3> atomPosition(atomLocations[globalChargeIdInCurrentBin][2],atomLocations[globalChargeIdInCurrentBin][3],atomLocations[globalChargeIdInCurrentBin][4]); 
	  atomPositionsInCurrentBin.push_back(atomPosition);

	  for(int iImageAtom = 0; iImageAtom < numberImageCharges; ++iImageAtom)
	    {
	      if(imageIdsContainer[iImageAtom] == globalChargeIdInCurrentBin)
		{
		  imageIdsOfAtomsInCurrentBin.push_back(iImageAtom);
		  std::vector<double> imageChargeCoor = imagePositions[iImageAtom];
		  imagePositionsOfAtomsInCurrentBin.push_back(imageChargeCoor);
		}
	    }

	}
      	 
      int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

      //
      //create constraint matrix for current bin
      //
      ConstraintMatrix * constraintsForVselfInBin = new ConstraintMatrix;
      DoFTools::make_hanging_node_constraints (dofHandler, *constraintsForVselfInBin);
     
      
      unsigned int inNodes=0, outNodes=0;
      std::map<types::global_dof_index,Point<3> >::iterator iterMap;
      for(iterMap = d_supportPoints.begin(); iterMap != d_supportPoints.end(); ++iterMap)
	{
	  if(locally_relevant_dofs.is_element(iterMap->first))
	    {
	      int overlapFlag = 0;
	      Point<3> nodalCoor = iterMap->second;
	      std::vector<double> distanceFromNode;

	      for(unsigned int iAtom = 0; iAtom < numberGlobalAtomsInBin+numberImageAtomsInBin; ++iAtom)
		{
		  Point<3> atomCoor;
		  if(iAtom < numberGlobalAtomsInBin)
		    {
		      atomCoor = atomPositionsInCurrentBin[iAtom];
		    }
		  else
		    {
		      atomCoor[0] = imagePositionsOfAtomsInCurrentBin[iAtom - numberGlobalAtomsInBin][0];
		      atomCoor[1] = imagePositionsOfAtomsInCurrentBin[iAtom - numberGlobalAtomsInBin][1];
		      atomCoor[2] = imagePositionsOfAtomsInCurrentBin[iAtom - numberGlobalAtomsInBin][2];
		    }

		 
		  double distance = nodalCoor.distance(atomCoor);

		  if(distance < radiusAtomBall)
		    overlapFlag += 1;

		  if(overlapFlag > 1)
		    {
		      std::cout<< "One of your Bins has a problem. It has interacting atoms" << std::endl;
		      exit(-1);
		    }

		  distanceFromNode.push_back(distance);

		}//atom loop

	      std::vector<double>::iterator minDistanceIter = std::min_element(distanceFromNode.begin(),
									       distanceFromNode.end());

	      std::iterator_traits<std::vector<double>::iterator>::difference_type minDistanceAtomId = std::distance(distanceFromNode.begin(),
														     minDistanceIter);


	      double minDistance = *minDistanceIter;

	      int chargeId;

	      if(minDistanceAtomId < numberGlobalAtomsInBin)
		chargeId = atomsInCurrentBin[minDistanceAtomId];
	      else
		chargeId = imageIdsOfAtomsInCurrentBin[minDistanceAtomId-numberGlobalAtomsInBin]+numberGlobalAtoms;

	      std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlag[iBin];
	   
	      if(minDistance < radiusAtomBall)
		{
		  boundaryNodeMap[iterMap->first]=chargeId;
		  inNodes++;
		}
	      else
		{
		  double atomCharge;
		  
		  if(minDistanceAtomId < numberGlobalAtomsInBin)
		    {
		      if(isPseudopotential)
			atomCharge = atomLocations[chargeId][1];
		      else
			atomCharge = atomLocations[chargeId][0];
		    }
		  else
		    atomCharge = imageChargeValues[imageIdsOfAtomsInCurrentBin[minDistanceAtomId-numberGlobalAtomsInBin]];

		  double potentialValue = -atomCharge/minDistance;
		  constraintsForVselfInBin->add_line(iterMap->first);
		  constraintsForVselfInBin->set_inhomogeneity(iterMap->first,potentialValue);
		  boundaryNodeMap[iterMap->first] = -1;
		  outNodes++;

		}//else loop

	    }//locally relevant dofs
 
	}//nodal loop
      constraintsForVselfInBin->merge(constraintsNone,ConstraintMatrix::MergeConflictBehavior::left_object_wins);
      constraintsForVselfInBin->close();
      constraintsVector.push_back(constraintsForVselfInBin);
      
      //std::cout<<"Size of Constraints: "<<constraintsForVselfInBin->n_constraints()<<std::endl;
      //std::cout << "In: " << inNodes << "  Out: " << outNodes << "\n";

    }//bin loop


  return;

}//
