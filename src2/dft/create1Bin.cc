//source file for locating core atom nodes

void dftClass::createAtomBins(std::vector<const ConstraintMatrix * > & constraintsVector){ 

  double radiusAtomBall = 4.0;
  std::map<types::global_dof_index,Point<3> >::iterator iterMap;
  ConstraintMatrix * constraintsForVselfInBin = new ConstraintMatrix;
  DoFTools::make_hanging_node_constraints (dofHandler, *constraintsForVselfInBin);
  unsigned int inNodes=0, outNodes=0;
  for(iterMap = d_supportPoints.begin(); iterMap != d_supportPoints.end(); ++iterMap)
    {
      if(locally_relevant_dofs.is_element(iterMap->first))
	{
	  int overlapFlag = 0;
	  Point<3> nodalCoor = iterMap->second;
	  std::vector<double> distanceFromNode;

	  for(unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
	    {
	      Point<3> atomCoor(atomLocations[iAtom][2],atomLocations[iAtom][3],atomLocations[iAtom][4]);
	      double distance = nodalCoor.distance(atomCoor);

	      if(distance < radiusAtomBall)
		overlapFlag += 1;

	      if(overlapFlag > 1)
		{
		  std::cerr<< "One of your Bins has a problem. It has interacting atoms" << std::endl;
		  exit(-1);
		}

	      distanceFromNode.push_back(distance);

	    }//atom loop

	  std::vector<double>::iterator minDistanceIter = std::min_element(distanceFromNode.begin(),
									   distanceFromNode.end());

	  std::iterator_traits<std::vector<double>::iterator>::difference_type minDistanceAtomId = std::distance(distanceFromNode.begin(),
														 minDistanceIter);


	  double minDistance = *minDistanceIter;
	   
	  if(minDistance < radiusAtomBall)
	    {
	      inNodes++;
	    }
	  else
	    {
	      double atomCharge;
	      if(isPseudopotential)
		atomCharge = atomLocations[minDistanceAtomId][1];
	      else
		atomCharge = atomLocations[minDistanceAtomId][0];
	      double potentialValue = -atomCharge/minDistance;
	      constraintsForVselfInBin->add_line(iterMap->first);
	      constraintsForVselfInBin->set_inhomogeneity(iterMap->first,potentialValue);
	      outNodes++;
	    }

	}//locally relevant dofs

    }//end of global_dof_index
  constraintsVector.push_back(constraintsForVselfInBin);
  constraintsForVselfInBin->close();
  std::cout<<"Size of Constraints: "<<constraintsForVselfInBin->n_constraints()<<std::endl;
  std::cout << "In: " << inNodes << "  Out: " << outNodes << "\n";
  return;

}//
