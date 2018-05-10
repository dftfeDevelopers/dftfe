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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Sambit Das (2018)
//

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::initLocalPseudoPotential()
{
  pseudoValues.clear();
  //
  //Reading single atom rho initial guess
  //
  std::map<unsigned int, alglib::spline1dinterpolant> pseudoSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > pseudoPotentialData;
  std::map<unsigned int, double> outerMostPointPseudo;


  //
  //loop over atom types
  //
  for(std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char pseudoFile[256];
      if (dftParameters::pseudoProjector==2)
	  sprintf(pseudoFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/locPot.dat", DFT_PATH,*it);
      else
          sprintf(pseudoFile, "%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/locPot.dat", DFT_PATH,*it);

      if (!dftParameters::reproducible_output)
        pcout<<"Reading Local Pseudo-potential data from: " <<pseudoFile<<std::endl;
      dftUtils::readFile(2, pseudoPotentialData[*it], pseudoFile);
      unsigned int numRows = pseudoPotentialData[*it].size()-1;
      std::vector<double> xData(numRows), yData(numRows);
      for(unsigned int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow] = pseudoPotentialData[*it][irow][0];
	  yData[irow] = pseudoPotentialData[*it][irow][1];
	}

      //interpolate pseudopotentials
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t bound_type_l = 0;
      alglib::ae_int_t bound_type_r = 1;
      const double slopeL= (pseudoPotentialData[*it][1][1]-pseudoPotentialData[*it][0][1])/(pseudoPotentialData[*it][1][0]-pseudoPotentialData[*it][0][0]);
      const double slopeR=-pseudoPotentialData[*it][numRows-1][1]/pseudoPotentialData[*it][numRows-1][0];
      spline1dbuildcubic(x, y, numRows, bound_type_l, slopeL, bound_type_r, slopeR, pseudoSpline[*it]);
      outerMostPointPseudo[*it]= xData[numRows-1];
      //
      if(outerMostPointPseudo[*it] < d_pspTail)
	 d_pspTail = outerMostPointPseudo[*it] ;
    }
   if (dftParameters::verbosity==2)
      pcout << " d_pspTail adjusted to " << d_pspTail << std::endl ;
  //
  //Initialize pseudopotential
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  //FEValues<3> fe_values (FE, quadrature_formula, update_values);
  FEValues<3> fe_values (FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points = quadrature_formula.size();

  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = d_imageIds.size();

  //
  //loop over elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  fe_values.reinit(cell);
	  pseudoValues[cell->id()]=std::vector<double>(n_q_points);
	  double * pseudoValuesPtr = &((pseudoValues)[cell->id()][0]);
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
	      const Point<3> & quadPoint=fe_values.quadrature_point(q);
	      double pseudoValueAtQuadPt=0.0;
	      //loop over atoms
	      for (unsigned int n=0; n<atomLocations.size(); n++)
		{
		  Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
		  double distanceToAtom = quadPoint.distance(atom);
		  if(distanceToAtom <= d_pspTail)//outerMostPointPseudo[atomLocations[n][0]])
		    {
		      pseudoValueAtQuadPt += alglib::spline1dcalc(pseudoSpline[atomLocations[n][0]], distanceToAtom);
		    }
		  else
		    {
		      pseudoValueAtQuadPt += (-atomLocations[n][1])/distanceToAtom;
		    }
		}

	      //loop over image charges
	      for(unsigned int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
		{
		  Point<3> imageAtom(d_imagePositions[iImageCharge][0],d_imagePositions[iImageCharge][1],d_imagePositions[iImageCharge][2]);
		  double distanceToAtom = quadPoint.distance(imageAtom);
		  int masterAtomId = d_imageIds[iImageCharge];
		  if(distanceToAtom <= d_pspTail)//outerMostPointPseudo[atomLocations[masterAtomId][0]])
		    {
		      pseudoValueAtQuadPt += alglib::spline1dcalc(pseudoSpline[atomLocations[masterAtomId][0]], distanceToAtom);
		    }
		  else
		    {
		      pseudoValueAtQuadPt += (-atomLocations[masterAtomId][1])/distanceToAtom;
		    }

		}
	      pseudoValuesPtr[q] = pseudoValueAtQuadPt;//1.0
	    }
	}
    }

  //
  //
  //
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::initNonLocalPseudoPotential()
{
  d_pseudoWaveFunctionIdToFunctionIdDetails.clear();
  d_deltaVlIdToFunctionIdDetails.clear();
  d_numberPseudoAtomicWaveFunctions.clear();
  d_numberPseudoPotentials.clear();
  d_nonLocalAtomGlobalChargeIds.clear();
  d_globalChargeIdToImageIdMap.clear();
  d_pseudoWaveFunctionSplines.clear();
  d_deltaVlSplines.clear();
  d_outerMostPointPseudoWaveFunctionsData.clear();
  d_outerMostPointPseudoPotData.clear();
  // Store the Map between the atomic number and the waveFunction details
  // (i.e. map from atomicNumber to a 2D vector storing atom specific wavefunction Id and its corresponding
  // radial and angular Ids)
  // (atomicNumber->[atomicWaveFunctionId][Global Spline Id, l quantum number, m quantum number]
  //
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToWaveFunctionIdDetails;
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToPotentialIdMap;
  //
  // Store the number of unique splines encountered so far
  //
  unsigned int cumulativeSplineId    = 0;
  unsigned int cumulativePotSplineId = 0;



  for(std::set<unsigned int>::iterator it = atomTypes.begin(); it != atomTypes.end(); ++it)
    {
      char pseudoAtomDataFile[256];
      sprintf(pseudoAtomDataFile, "%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/PseudoAtomDat", DFT_PATH, *it);

      unsigned int atomicNumber = *it;

      //pcout<<"Reading data from file: "<<pseudoAtomDataFile<<std::endl;

      //
      // open the testFunctionFileName
      //
      std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);


      //
      // 2D vector to store the function Id details for the current atom type
      // [Atomic wavefunction id](global spline id, l quantum number, m quantum number)
      //
      std::vector<std::vector<int> > atomicFunctionIdDetails;

      //
      // store the number of single-atom waveFunctions associated with the current atomic number
      //
      unsigned int numberAtomicWaveFunctions;

      //
      // read number of single-atom wavefunctions
      //
      if(readPseudoDataFileNames.is_open())
	readPseudoDataFileNames >> numberAtomicWaveFunctions;

      //
      // resize atomicFunctionIdDetails
      //
      atomicFunctionIdDetails.resize(numberAtomicWaveFunctions);

      //
      // Skip the rest in the first line and proceed to next line
      //
      readPseudoDataFileNames.ignore();
      if (dftParameters::verbosity==2)
         pcout << "Number of Pseudo Wave Functions for atom with Z: " << atomicNumber<<" is "<<numberAtomicWaveFunctions << std::endl;

      //
      //string to store each line of the file
      //
      std::string readLine;

      //
      // set to store the radial(spline) function Ids
      //
      std::set<int> radFunctionIds;

      //
      //
      for(unsigned int i = 0; i < numberAtomicWaveFunctions; ++i)
	{

	  std::vector<int>  & radAndAngularFunctionId = atomicFunctionIdDetails[i];

	  radAndAngularFunctionId.resize(3,0);

	  //
	  // get the next line
	  //
	  std::getline(readPseudoDataFileNames, readLine);
	  std::istringstream lineString(readLine);

	  unsigned int count = 0;
	  int Id;
	  double mollifierRadius;
	  std::string dummyString;
	  while(lineString >> dummyString)
	    {
	      if(count < 3)
		{

		  Id = atoi(dummyString.c_str());
		  //
		  // insert the radial(spline) Id to the splineIds set
		  //
		  if(count == 0)
		    radFunctionIds.insert(Id);

		  radAndAngularFunctionId[count] = Id;

		}
	      else
		{

		  std::cerr<<"Invalid argument in the SingleAtomData file"<<std::endl;
		  exit(-1);
		}

	      count++;

	    }

	  //
	  // Add the cumulativeSplineId to radialId
	  //
	  radAndAngularFunctionId[0] += cumulativeSplineId;

	  if (dftParameters::verbosity==2)
	      pcout << "Radial and Angular Functions Ids: " << radAndAngularFunctionId[0] << " " << radAndAngularFunctionId[1] << " " << radAndAngularFunctionId[2] << std::endl;

	}

      //
      // map the atomic number to atomicNumberToFunctionIdDetails
      //
      atomicNumberToWaveFunctionIdDetails[atomicNumber] = atomicFunctionIdDetails;

      //
      // update cumulativeSplineId
      //
      cumulativeSplineId += radFunctionIds.size();

      //
      // store the splines for the current atom type
      //
      std::vector<alglib::spline1dinterpolant> atomicSplines(radFunctionIds.size());
      std::vector<alglib::real_1d_array> atomicRadialNodes(radFunctionIds.size());
      std::vector<alglib::real_1d_array> atomicRadialFunctionNodalValues(radFunctionIds.size());
      std::vector<double> outerMostRadialPointWaveFunction(radFunctionIds.size());

      //pcout << "Number radial Pseudo wavefunctions for atomic number " << atomicNumber << " is: " << radFunctionIds.size() << std::endl;

      //
      // string to store the radial function file name
      //
      std::string tempPsiRadialFunctionFileName;

      for(unsigned int i = 0; i < radFunctionIds.size(); ++i)
	{

	  //
	  // get the radial function file name (name local to the directory)
	  //
	  readPseudoDataFileNames >> tempPsiRadialFunctionFileName;

	  char psiRadialFunctionFileName[256];
	  sprintf(psiRadialFunctionFileName, "%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/%s", DFT_PATH,*it,tempPsiRadialFunctionFileName.c_str());
	  //pcout<<"Radial WaveFunction File Name: " <<psiRadialFunctionFileName<<std::endl;

	  //
	  // 2D vector to store the radial coordinate and its corresponding
	  // function value
	  std::vector< std::vector<double> > radialFunctionData(0);

	  //
	  //read the radial function file
	  //
	  dftUtils::readFile(2,radialFunctionData,psiRadialFunctionFileName);


	  int numRows = radialFunctionData.size();

	  //std::cout << "Number of Rows: " << numRows << std::endl;

	  double xData[numRows];
	  double yData[numRows];

	  for (int iRow = 0; iRow < numRows; ++iRow)
	    {
	      xData[iRow] = radialFunctionData[iRow][0];
	      yData[iRow] = radialFunctionData[iRow][1];
	    }

	  outerMostRadialPointWaveFunction[i] = xData[numRows - 1];

	  alglib::real_1d_array & x = atomicRadialNodes[i];
	  atomicRadialNodes[i].setcontent(numRows, xData);

	  alglib::real_1d_array & y = atomicRadialFunctionNodalValues[i];
	  atomicRadialFunctionNodalValues[i].setcontent(numRows, yData);

	  alglib::ae_int_t natural_bound_type = 1;
	  alglib::spline1dbuildcubic(atomicRadialNodes[i],
				     atomicRadialFunctionNodalValues[i],
				     numRows,
				     natural_bound_type,
				     0.0,
				     natural_bound_type,
				     0.0,
				     atomicSplines[i]);

	}

      //
      // insert into d_splines
      //
      d_pseudoWaveFunctionSplines.insert(d_pseudoWaveFunctionSplines.end(), atomicSplines.begin(), atomicSplines.end());
      d_outerMostPointPseudoWaveFunctionsData.insert(d_outerMostPointPseudoWaveFunctionsData.end(),outerMostRadialPointWaveFunction.begin(),outerMostRadialPointWaveFunction.end());

      //
      // read local PSP filename
      //
      std::string tempLocPseudoPotentialFileName;
      readPseudoDataFileNames >> tempLocPseudoPotentialFileName;

      char localPseudoPotentialFileName[256];
      sprintf(localPseudoPotentialFileName,"%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/%s", DFT_PATH,*it,tempLocPseudoPotentialFileName.c_str());
      //pcout<<"Local Pseudo File Name: " <<localPseudoPotentialFileName<<std::endl;

      //
      //read the local pseudopotential radial data
      //
      std::vector<std::vector<double> > localPseudoPotentialData;

      //
      //read the radial function file
      //
      dftUtils::readFile(2,localPseudoPotentialData,localPseudoPotentialFileName);

      //
      //read the number of angular momentum components
      //
      unsigned int numberAngularMomentumSpecificPotentials;

      //
      // 2D vector to store the function Id details for the current atom type
      // [potential id](global spline id, l quantum number)
      //
      std::vector<std::vector<int> > pseudoPotentialIdDetails;
      //
      // get the file name for the radial function corresponding to single atom electron-density(splines)
      //
      readPseudoDataFileNames >> numberAngularMomentumSpecificPotentials;

      //
      // resize pseudoPotentialIdDetails
      //
      pseudoPotentialIdDetails.resize(numberAngularMomentumSpecificPotentials);

      //
      // Skip the rest in the first line and proceed to next line
      //
      readPseudoDataFileNames.ignore();

      if (dftParameters::verbosity==2)
         pcout << "Number of Angular momentum specific potentials: " << numberAngularMomentumSpecificPotentials<< std::endl;

      std::string readPotLine;

      //
      // set to store the radial(spline) function Ids
      //
      std::set<int> potentialIds;

      for(unsigned int i = 0; i < numberAngularMomentumSpecificPotentials; ++i)
	{

	  std::vector<int>  & radAndAngularFunctionId = pseudoPotentialIdDetails[i];

	  radAndAngularFunctionId.resize(2,0);

	  //
	  // get the next line
	  //
	  std::getline(readPseudoDataFileNames, readPotLine);

	  std::istringstream lineString(readPotLine);

	  //std::cout << "Printing" << readLine << std::endl;

	  unsigned int count = 0;
	  int Id;
	  std::string dummyString;
	  while(lineString >> dummyString)
	    {

	      //std::cout << "DummyString: " << dummyString << std::endl;

	      if(count < 2)
		{

		  Id = atoi(dummyString.c_str());
		  //
		  // insert the radial(spline) Id to the splineIds set
		  //
		  if(count == 0)
		    potentialIds.insert(Id);

		  radAndAngularFunctionId[count] = Id;

		}
	      else
		{

		  std::cerr<< "Invalid argument in the SingleAtomData file" << std::endl;
		  exit(-1);
		}

	      count++;

	    }

	  //
	  // Add the cumulativeSplineId to radialId
	  //
	  radAndAngularFunctionId[0] += cumulativePotSplineId;
	  if (dftParameters::verbosity==2)
	     pcout << "Radial and Angular Function Potential Ids: " << radAndAngularFunctionId[0] << " " << radAndAngularFunctionId[1] << std::endl;

	}

      //
      // map the atomic number to atomicNumberToPotentialIdDetails
      //
      atomicNumberToPotentialIdMap[atomicNumber] = pseudoPotentialIdDetails;

      //
      //update cumulativePotSplineId
      //
      cumulativePotSplineId += potentialIds.size();

      //
      //store the splines for the current atom type
      //
      std::vector<alglib::spline1dinterpolant> deltaVlSplines(potentialIds.size());
      std::vector<alglib::real_1d_array> deltaVlRadialNodes(potentialIds.size());
      std::vector<alglib::real_1d_array> deltaVlNodalValues(potentialIds.size());
      std::vector<double> outerMostRadialPointPseudoPot(potentialIds.size());

      //pcout << "Number radial potential functions for atomic number " << atomicNumber << " is: " << potentialIds.size() << std::endl;

      //
      // string to store the radial function file name
      //
      std::string tempPotentialRadFunctionFileName;

      for(unsigned int i = 0; i < potentialIds.size(); ++i)
	{
	  //
	  // get the radial function file name (name local to the directory)
	  //
	  readPseudoDataFileNames >> tempPotentialRadFunctionFileName;

	  char pseudoPotentialRadFunctionFileName[256];
	  sprintf(pseudoPotentialRadFunctionFileName,"%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/%s", DFT_PATH,*it,tempPotentialRadFunctionFileName.c_str());
	  //pcout<<"Radial Pseudopotential File Name: " <<pseudoPotentialRadFunctionFileName<<std::endl;

	  //
	  // 2D vector to store the radial coordinate and its corresponding
	  // function value
	  std::vector<std::vector<double> > radialFunctionData(0);

	  //
	  //read the radial function file
	  //
	  dftUtils::readFile(2,radialFunctionData,pseudoPotentialRadFunctionFileName);
	  int numRows = radialFunctionData.size();

	  //pcout << "Number of Rows for potentials: " << numRows << std::endl;

	  double xData[numRows];
	  double yData[numRows];

	  for (int iRow = 0; iRow < numRows; ++iRow)
	    {
	      xData[iRow] = radialFunctionData[iRow][0];
	      yData[iRow] = radialFunctionData[iRow][1]-localPseudoPotentialData[iRow][1];
	    }

	  outerMostRadialPointPseudoPot[i] = xData[numRows - 1];

	  deltaVlRadialNodes[i].setcontent(numRows,xData);
	  deltaVlNodalValues[i].setcontent(numRows,yData);

	  alglib::ae_int_t natural_bound_type = 1;
	  alglib::spline1dbuildcubic(deltaVlRadialNodes[i],
				     deltaVlNodalValues[i],
				     numRows,
				     natural_bound_type,
				     0.0,
				     natural_bound_type,
				     0.0,
				     deltaVlSplines[i]);

	}

      //
      // insert into d_splines
      //
      d_deltaVlSplines.insert(d_deltaVlSplines.end(), deltaVlSplines.begin(), deltaVlSplines.end());
      d_outerMostPointPseudoPotData.insert(d_outerMostPointPseudoPotData.end(),outerMostRadialPointPseudoPot.begin(),outerMostRadialPointPseudoPot.end());

    }//atomNumber loop

  //
  // Get the number of charges present in the system
  //
  unsigned int numberGlobalCharges  = atomLocations.size();

  //
  //store information for non-local atoms
  //
  std::vector<int> nonLocalAtomGlobalChargeIds;
  std::vector<std::vector<int> > globalChargeIdToImageIdMap;

  globalChargeIdToImageIdMap.resize(numberGlobalCharges);


  for(unsigned int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
    {

      //
      // Get the atomic number for current nucleus
      //
      unsigned int atomicNumber =  atomLocations[iCharge][0];

      //
      // Get the function id details for the current nucleus
      //
      std::vector<std::vector<int> > & atomicFunctionIdDetails =
	atomicNumberToWaveFunctionIdDetails[atomicNumber];


      std::vector<std::vector<int> > & pseudoPotentialIdDetails =
	atomicNumberToPotentialIdMap[atomicNumber];

      //
      // Get the number of functions associated with the current nucleus
      //
      unsigned int numberAtomicWaveFunctions = atomicFunctionIdDetails.size();
      unsigned int numberAngularMomentumSpecificPotentials = pseudoPotentialIdDetails.size();

      if(numberAtomicWaveFunctions > 0 && numberAngularMomentumSpecificPotentials > 0)
	{
	  nonLocalAtomGlobalChargeIds.push_back(iCharge);
	  d_numberPseudoAtomicWaveFunctions.push_back(numberAtomicWaveFunctions);
	  d_numberPseudoPotentials.push_back(numberAngularMomentumSpecificPotentials);
	}


      //
      // Add the atomic wave function details to the global wave function vectors
      //
      for(unsigned iAtomWave = 0; iAtomWave < numberAtomicWaveFunctions; ++iAtomWave)
	{
	  d_pseudoWaveFunctionIdToFunctionIdDetails.push_back(atomicFunctionIdDetails[iAtomWave]);
	}

      for(unsigned iPot = 0 ; iPot < numberAngularMomentumSpecificPotentials; ++iPot)
	{
	  d_deltaVlIdToFunctionIdDetails.push_back(pseudoPotentialIdDetails[iPot]);
	}

      //
      // insert the master charge Id into the map first
      //
      globalChargeIdToImageIdMap[iCharge].push_back(iCharge);

    }//end of iCharge loop

  d_nonLocalAtomGlobalChargeIds = nonLocalAtomGlobalChargeIds;

  if (dftParameters::verbosity==2)
     pcout<<"Number of Nonlocal Atoms: " <<d_nonLocalAtomGlobalChargeIds.size()<<std::endl;
  //
  //fill up global charge image Id map by inserting the image atoms
  //corresponding to the master chargeId
  const int numberImageCharges = d_imageIds.size();

  for(int iImage = 0; iImage < numberImageCharges; ++iImage)
    {
      //
      //Get the masterChargeId corresponding to the current image atom
      //
      const int masterChargeId = d_imageIds[iImage];

      //
      //insert into the map
      //
      globalChargeIdToImageIdMap[masterChargeId].push_back(iImage+numberGlobalCharges);

    }

  d_globalChargeIdToImageIdMap = globalChargeIdToImageIdMap;

  return;
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::computeSparseStructureNonLocalProjectors()
{

  //
  //get the number of non-local atoms
  //
  int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();
  const double nlpTolerance = 1e-08;


  //
  //pre-allocate data structures that stores the sparsity of deltaVl
  //
  d_sparsityPattern.clear();
  d_elementIteratorsInAtomCompactSupport.clear();
  d_elementIdsInAtomCompactSupport.clear();
  d_elementOneFieldIteratorsInAtomCompactSupport.clear();

  d_sparsityPattern.resize(numberNonLocalAtoms);
  d_elementIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_elementIdsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_elementOneFieldIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_nonLocalAtomIdsInCurrentProcess.clear();

  //
  //loop over nonlocal atoms
  //
  unsigned int sparseFlag = 0;
  int cumulativePotSplineId = 0;
  int pseudoPotentialId;


  //
  //get number of global charges
  //
  unsigned int numberGlobalCharges  = atomLocations.size();

  //
  //get FE data structures
  //
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  //FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  FEValues<3> fe_values(FE, quadrature, update_quadrature_points);
  const unsigned int numberQuadraturePoints = quadrature.size();
  //const parallel::distributed::Triangulation<3> & tria = dofHandlerEigen.get_triangulation();
  //const unsigned int numberElements  = tria.n_locally_owned_active_cells();

  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  int iElemCount = 0;
  for(; cell != endc; ++cell)
    {
      if(cell->is_locally_owned())
	iElemCount += 1;
    }

  const unsigned int numberElements = iElemCount;


  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {

      //
      //temp variables
      //
      int matCount = 0;
      bool isAtomIdInProcessor=false;

      //
      //get the number of angular momentum specific pseudopotentials for the current nonlocal atom
      //
      int numberAngularMomentumSpecificPotentials = d_numberPseudoPotentials[iAtom];

      //
      //get the global charge Id of the current nonlocal atom
      //
      const int globalChargeIdNonLocalAtom =  d_nonLocalAtomGlobalChargeIds[iAtom];

      //
      //get the imageIdmap information corresponding to globalChargeIdNonLocalAtom
      //
      std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMap[globalChargeIdNonLocalAtom];

      //
      //resize the data structure corresponding to sparsity pattern
      //
      d_sparsityPattern[iAtom].resize(numberElements,-1);

      //
      //parallel loop over all elements
      //
      typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
      typename DoFHandler<3>::active_cell_iterator cellEigen = dofHandlerEigen.begin_active();

      int iElem = -1;

      for(; cell != endc; ++cell,++cellEigen)
	{
	  if(cell->is_locally_owned())
	    {

	      //compute the values for the current element
	      fe_values.reinit(cell);

	      iElem += 1;
	      for(int iPsp = 0; iPsp < numberAngularMomentumSpecificPotentials; ++iPsp)
		{
		  sparseFlag = 0;
		  pseudoPotentialId = iPsp + cumulativePotSplineId;
		  const int globalSplineId = d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][0];
		  for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
		    {
		      //MappingQ1<3,3> test;
		      //Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(iQuadPoint)));
		      Point<3> quadPoint=fe_values.quadrature_point(iQuadPoint);

		      for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
			{

			  int chargeId = imageIdsList[iImageAtomCount];

			  //const Point & chargePoint = chargeId < numberGlobalCharges? d_nuclearContainer.getGlobalPoint(chargeId,meshId):
			  //d_nuclearContainer.getImagePoint(chargeId-numberGlobalCharges,meshId);

			  Point<3> chargePoint(0.0,0.0,0.0);

			  if(chargeId < numberGlobalCharges)
			    {
			      chargePoint[0] = atomLocations[chargeId][2];
			      chargePoint[1] = atomLocations[chargeId][3];
			      chargePoint[2] = atomLocations[chargeId][4];
			    }
			  else
			    {
			      chargePoint[0] = d_imagePositions[chargeId-numberGlobalCharges][0];
			      chargePoint[1] = d_imagePositions[chargeId-numberGlobalCharges][1];
			      chargePoint[2] = d_imagePositions[chargeId-numberGlobalCharges][2];
			    }

			  double r = quadPoint.distance(chargePoint);
			  double deltaVl;

			  if(r <= d_pspTail)//d_outerMostPointPseudoPotData[globalSplineId])
			    {
			      pseudoUtils::getRadialFunctionVal(r,
						                deltaVl,
						                &d_deltaVlSplines[globalSplineId]);
			    }
			  else
			    {
			      deltaVl = 0.0;
			    }

			  if(fabs(deltaVl) >= nlpTolerance)
			    {
			      sparseFlag = 1;
			      break;
			    }
			}//imageAtomLoop

		      if(sparseFlag == 1)
			break;

		    }//quadrature loop

		  if(sparseFlag == 1)
		    break;

		}//iPsp loop ("l" loop)

	      if(sparseFlag==1) {
		d_sparsityPattern[iAtom][iElem] = matCount;
		d_elementIteratorsInAtomCompactSupport[iAtom].push_back(cellEigen);
		d_elementIdsInAtomCompactSupport[iAtom].push_back(iElem);
		d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].push_back(cell);
		matCount += 1;
                isAtomIdInProcessor=true;
	      }

	    }
	}//cell loop

      cumulativePotSplineId += numberAngularMomentumSpecificPotentials;

      if (dftParameters::verbosity==2)
         pcout<<"No.of non zero elements in the compact support of atom "<<iAtom<<" is "<<d_elementIteratorsInAtomCompactSupport[iAtom].size()<<std::endl;
      if (isAtomIdInProcessor)
          d_nonLocalAtomIdsInCurrentProcess.push_back(iAtom);
    }//atom loop

  d_nonLocalAtomIdsInElement.clear();
  d_nonLocalAtomIdsInElement.resize(numberElements);


  for(int iElem = 0; iElem < numberElements; ++iElem)
    {
      for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{
	  if(d_sparsityPattern[iAtom][iElem] >= 0)
	    d_nonLocalAtomIdsInElement[iElem].push_back(iAtom);
	}
    }

   //
   //data structures for memory optimization of projectorKetTimesVector
   //
   std::vector<unsigned int> nonLocalAtomIdsAllProcessFlattened;
   pseudoUtils::exchangeLocalList(d_nonLocalAtomIdsInCurrentProcess,
                                  nonLocalAtomIdsAllProcessFlattened,
                                  n_mpi_processes,
                                  mpi_communicator);

   std::vector<unsigned int> nonLocalAtomIdsSizeCurrentProcess(1); nonLocalAtomIdsSizeCurrentProcess[0]=d_nonLocalAtomIdsInCurrentProcess.size();
   std::vector<unsigned int> nonLocalAtomIdsSizesAllProcess;
   pseudoUtils::exchangeLocalList(nonLocalAtomIdsSizeCurrentProcess,
                                  nonLocalAtomIdsSizesAllProcess,
                                  n_mpi_processes,
                                  mpi_communicator);

   std::vector<std::vector<unsigned int> >nonLocalAtomIdsInAllProcess(n_mpi_processes);
   unsigned int count=0;
   for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
   {
     for (unsigned int j=0; j < nonLocalAtomIdsSizesAllProcess[iProc]; j++)
     {
       nonLocalAtomIdsInAllProcess[iProc].push_back(nonLocalAtomIdsAllProcessFlattened[count]);
       count++;
     }
   }
   nonLocalAtomIdsAllProcessFlattened.clear();

   IndexSet nonLocalOwnedAtomIdsInCurrentProcess; nonLocalOwnedAtomIdsInCurrentProcess.set_size(numberNonLocalAtoms);
   nonLocalOwnedAtomIdsInCurrentProcess.add_indices(d_nonLocalAtomIdsInCurrentProcess.begin(),d_nonLocalAtomIdsInCurrentProcess.end());
   IndexSet nonLocalGhostAtomIdsInCurrentProcess(nonLocalOwnedAtomIdsInCurrentProcess);
   for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
   {
      if (iProc < this_mpi_process)
      {
         IndexSet temp; temp.set_size(numberNonLocalAtoms);
         temp.add_indices(nonLocalAtomIdsInAllProcess[iProc].begin(),nonLocalAtomIdsInAllProcess[iProc].end());
         nonLocalOwnedAtomIdsInCurrentProcess.subtract_set(temp);
      }
   }

   nonLocalGhostAtomIdsInCurrentProcess.subtract_set(nonLocalOwnedAtomIdsInCurrentProcess);

   std::vector<unsigned int> ownedNonLocalAtomIdsSizeCurrentProcess(1); ownedNonLocalAtomIdsSizeCurrentProcess[0]=nonLocalOwnedAtomIdsInCurrentProcess.n_elements();
   std::vector<unsigned int> ownedNonLocalAtomIdsSizesAllProcess;
   pseudoUtils::exchangeLocalList(ownedNonLocalAtomIdsSizeCurrentProcess,
                                  ownedNonLocalAtomIdsSizesAllProcess,
                                  n_mpi_processes,
                                  mpi_communicator);
   //renumbering to make contiguous set of nonLocal atomIds
   std::map<int, int> oldToNewNonLocalAtomIds;
   std::map<int, int> newToOldNonLocalAtomIds;
   unsigned int startingCount=0;
   for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
   {
      if (iProc < this_mpi_process)
      {
        startingCount+=ownedNonLocalAtomIdsSizesAllProcess[iProc];
      }
   }

   IndexSet nonLocalOwnedAtomIdsInCurrentProcessRenum, nonLocalGhostAtomIdsInCurrentProcessRenum;
   nonLocalOwnedAtomIdsInCurrentProcessRenum.set_size(numberNonLocalAtoms);
   nonLocalGhostAtomIdsInCurrentProcessRenum.set_size(numberNonLocalAtoms);
   for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcess.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcess.end(); it++)
   {
       oldToNewNonLocalAtomIds[*it]=startingCount;
       newToOldNonLocalAtomIds[startingCount]=*it;
       nonLocalOwnedAtomIdsInCurrentProcessRenum.add_index(startingCount);
       startingCount++;
   }

   pseudoUtils::exchangeNumberingMap(oldToNewNonLocalAtomIds,
                                     n_mpi_processes,
                                     mpi_communicator);
   pseudoUtils::exchangeNumberingMap(newToOldNonLocalAtomIds,
                                     n_mpi_processes,
                                     mpi_communicator);

   for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcess.begin(); it!=nonLocalGhostAtomIdsInCurrentProcess.end(); it++)
   {
       unsigned int newAtomId=oldToNewNonLocalAtomIds[*it];
       nonLocalGhostAtomIdsInCurrentProcessRenum.add_index(newAtomId);
   }

   if(this_mpi_process==0 && false){
     for( std::map<int, int>::const_iterator it=oldToNewNonLocalAtomIds.begin(); it!=oldToNewNonLocalAtomIds.end();it++)
        std::cout<<" old nonlocal atom id: "<<it->first <<" new nonlocal atomid: "<<it->second<<std::endl;

     std::cout<<"number of local owned non local atom ids in all processors"<< '\n';
     for (unsigned int iProc=0; iProc<n_mpi_processes; iProc++)
         std::cout<<ownedNonLocalAtomIdsSizesAllProcess[iProc]<<",";
     std::cout<<std::endl;
   }
   if (false)
   {
     std::stringstream ss1;nonLocalOwnedAtomIdsInCurrentProcess.print(ss1);
     std::stringstream ss2;nonLocalGhostAtomIdsInCurrentProcess.print(ss2);
     std::string s1(ss1.str());s1.pop_back(); std::string s2(ss2.str());s2.pop_back();
     std::cout<<"procId: "<< this_mpi_process<< " old owned: "<< s1<< " old ghost: "<< s2<<std::endl;
     std::stringstream ss3;nonLocalOwnedAtomIdsInCurrentProcessRenum.print(ss3);
     std::stringstream ss4;nonLocalGhostAtomIdsInCurrentProcessRenum.print(ss4);
     std::string s3(ss3.str());s3.pop_back(); std::string s4(ss4.str());s4.pop_back();
     std::cout<<"procId: "<< this_mpi_process<< " new owned: "<< s3<<" new ghost: "<< s4<< std::endl;
   }
   AssertThrow(nonLocalOwnedAtomIdsInCurrentProcessRenum.is_ascending_and_one_to_one(mpi_communicator),ExcMessage("Incorrect renumbering and/or partitioning of non local atom ids"));

   int numberLocallyOwnedProjectors=0;
   int numberGhostProjectors=0;
   std::vector<unsigned int> coarseNodeIdsCurrentProcess;
   for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcessRenum.end(); it++)
   {
       coarseNodeIdsCurrentProcess.push_back(numberLocallyOwnedProjectors);
       numberLocallyOwnedProjectors += d_numberPseudoAtomicWaveFunctions[newToOldNonLocalAtomIds[*it]];

   }

   std::vector<unsigned int> ghostAtomIdNumberPseudoWaveFunctions;
   for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalGhostAtomIdsInCurrentProcessRenum.end(); it++)
   {
       const unsigned temp=d_numberPseudoAtomicWaveFunctions[newToOldNonLocalAtomIds[*it]];
       numberGhostProjectors += temp;
       ghostAtomIdNumberPseudoWaveFunctions.push_back(temp);
   }

   std::vector<unsigned int> numberLocallyOwnedProjectorsCurrentProcess(1); numberLocallyOwnedProjectorsCurrentProcess[0]=numberLocallyOwnedProjectors;
   std::vector<unsigned int> numberLocallyOwnedProjectorsAllProcess;
   pseudoUtils::exchangeLocalList(numberLocallyOwnedProjectorsCurrentProcess,
                                  numberLocallyOwnedProjectorsAllProcess,
                                  n_mpi_processes,
                                  mpi_communicator);

   startingCount=0;
   for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
   {
      if (iProc < this_mpi_process)
      {
	 startingCount+=numberLocallyOwnedProjectorsAllProcess[iProc];
      }
   }

   d_locallyOwnedProjectorIdsCurrentProcess.clear(); d_locallyOwnedProjectorIdsCurrentProcess.set_size(std::accumulate(numberLocallyOwnedProjectorsAllProcess.begin(),numberLocallyOwnedProjectorsAllProcess.end(),0));
   std::vector<unsigned int> v(numberLocallyOwnedProjectors) ;
   std::iota (std::begin(v), std::end(v), startingCount);
   d_locallyOwnedProjectorIdsCurrentProcess.add_indices(v.begin(),v.end());

   std::vector<unsigned int> coarseNodeIdsAllProcess;
   for (unsigned int i=0; i< coarseNodeIdsCurrentProcess.size();++i)
       coarseNodeIdsCurrentProcess[i]+=startingCount;
   pseudoUtils::exchangeLocalList(coarseNodeIdsCurrentProcess,
                                  coarseNodeIdsAllProcess,
                                  n_mpi_processes,
                                  mpi_communicator);

   d_ghostProjectorIdsCurrentProcess.clear(); d_ghostProjectorIdsCurrentProcess.set_size(std::accumulate(numberLocallyOwnedProjectorsAllProcess.begin(),numberLocallyOwnedProjectorsAllProcess.end(),0));
   unsigned int localGhostCount=0;
   for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalGhostAtomIdsInCurrentProcessRenum.end(); it++)
   {
      std::vector<unsigned int> g(ghostAtomIdNumberPseudoWaveFunctions[localGhostCount]);
      std::iota (std::begin(g), std::end(g), coarseNodeIdsAllProcess[*it]);
      d_ghostProjectorIdsCurrentProcess.add_indices(g.begin(),g.end());
      localGhostCount++;
   }
   if (false)
   {
     std::stringstream ss1;d_locallyOwnedProjectorIdsCurrentProcess.print(ss1);
     std::stringstream ss2;d_ghostProjectorIdsCurrentProcess.print(ss2);
     std::string s1(ss1.str());s1.pop_back(); std::string s2(ss2.str());s2.pop_back();
     std::cout<<"procId: "<< this_mpi_process<< " projectors owned: "<< s1<< " projectors ghost: "<< s2<<std::endl;
   }
   AssertThrow(d_locallyOwnedProjectorIdsCurrentProcess.is_ascending_and_one_to_one(mpi_communicator),ExcMessage("Incorrect numbering and/or partitioning of non local projectors"));

   d_projectorIdsNumberingMapCurrentProcess.clear();

   for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcess.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcess.end(); it++)
   {
       const int numberPseudoWaveFunctions=d_numberPseudoAtomicWaveFunctions[*it];

       for (unsigned int i=0; i<numberPseudoWaveFunctions;++i)
       {
	   d_projectorIdsNumberingMapCurrentProcess[std::make_pair(*it,i)]=coarseNodeIdsAllProcess[oldToNewNonLocalAtomIds[*it]]+i;
       }
   }

   for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcess.begin(); it!=nonLocalGhostAtomIdsInCurrentProcess.end(); it++)
   {
       const int numberPseudoWaveFunctions=d_numberPseudoAtomicWaveFunctions[*it];

       for (unsigned int i=0; i<numberPseudoWaveFunctions;++i)
       {
	   d_projectorIdsNumberingMapCurrentProcess[std::make_pair(*it,i)]=coarseNodeIdsAllProcess[oldToNewNonLocalAtomIds[*it]]+i;
       }
   }

   if (false){
     for (std::map<std::pair<unsigned int,unsigned int>, unsigned int>::const_iterator it=d_projectorIdsNumberingMapCurrentProcess.begin(); it!=d_projectorIdsNumberingMapCurrentProcess.end();++it)
     {
        std::cout << "procId: "<< this_mpi_process<<" ["<<it->first.first << "," << it->first.second << "] " << it->second<< std::endl;
     }
   }

#ifdef USE_COMPLEX
  dealii::parallel::distributed::Vector<std::complex<double> > vec(d_locallyOwnedProjectorIdsCurrentProcess,
                                                                   d_ghostProjectorIdsCurrentProcess,
                                                                   mpi_communicator);
#else
  dealii::parallel::distributed::Vector<double > vec(d_locallyOwnedProjectorIdsCurrentProcess,
                                                     d_ghostProjectorIdsCurrentProcess,
                                                     mpi_communicator);
#endif

  d_projectorKetTimesVectorPar.resize(eigenVectors[0].size());
  for (unsigned int i = 0; i < eigenVectors[0].size(); ++i)
    d_projectorKetTimesVectorPar[i].reinit(vec);

}

template<unsigned int FEOrder>
void dftClass<FEOrder>::computeElementalProjectorKets()
{

  //
  //get the number of non-local atoms
  //
  int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();

  //
  //get number of global charges
  //
  unsigned int numberGlobalCharges  = atomLocations.size();


  //
  //get FE data structures
  //
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  //FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  FEValues<3> fe_values(FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int numberNodesPerElement  = FE.dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();


  //
  //get number of kPoints
  //
  const unsigned int maxkPoints = d_kPointWeights.size();


  //
  //preallocate element Matrices
  //
  d_nonLocalProjectorElementMatrices.clear();
  d_nonLocalProjectorElementMatricesConjugate.clear();
  d_nonLocalPseudoPotentialConstants.clear();
  d_nonLocalProjectorElementMatrices.resize(numberNonLocalAtoms);
  d_nonLocalProjectorElementMatricesConjugate.resize(numberNonLocalAtoms);
  d_nonLocalPseudoPotentialConstants.resize(numberNonLocalAtoms);
  int cumulativePotSplineId = 0;
  int cumulativeWaveSplineId = 0;
  int waveFunctionId;
  int pseudoPotentialId;

  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int globalChargeIdNonLocalAtom =  d_nonLocalAtomGlobalChargeIds[iAtom];


      Point<3> nuclearCoordinates(atomLocations[globalChargeIdNonLocalAtom][2],atomLocations[globalChargeIdNonLocalAtom][3],atomLocations[globalChargeIdNonLocalAtom][4]);

      std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMap[globalChargeIdNonLocalAtom];

      //
      //get the number of elements in the compact support of the current nonlocal atom
      //
      int numberElementsInAtomCompactSupport = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();


      //pcout<<"Number of elements in compact support of nonlocal atom "<<iAtom<<" is "<<numberElementsInAtomCompactSupport<<std::endl;
      //pcout<<"Image Ids List: "<<imageIdsList.size()<<std::endl;
      //pcout<<numberElementsInAtomCompactSupport<<std::endl;

      //
      //get the number of pseudowavefunctions for the current nonlocal atoms
      //
      int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
      int numberAngularMomentumSpecificPotentials = d_numberPseudoPotentials[iAtom];

      if (dftParameters::verbosity==2)
      {
        pcout<<"Number of Pseudo wavefunctions: "<<std::endl;
        pcout<<numberPseudoWaveFunctions<<std::endl;
      }

      //
      //allocate element Matrices
      //
      d_nonLocalProjectorElementMatrices[iAtom].resize(numberElementsInAtomCompactSupport);
      d_nonLocalProjectorElementMatricesConjugate[iAtom].resize(numberElementsInAtomCompactSupport);
      d_nonLocalPseudoPotentialConstants[iAtom].resize(numberPseudoWaveFunctions,0.0);

      for(int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport; ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom][iElemComp];

	  //compute values for the current elements
	  fe_values.reinit(cell);

#ifdef USE_COMPLEX
	  d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(maxkPoints,
								      std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
	  d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp].resize(maxkPoints,
									       std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
#else
	  d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(maxkPoints,
								      std::vector<double> (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
#endif

	  int iPsp = -1;
	  int lTemp = 1e5;

	  for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	    {
	      waveFunctionId = iPseudoWave + cumulativeWaveSplineId;
	      const int globalWaveSplineId = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
	      const int lQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
	      const int mQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][2];

	      //
	      //access pseudoPotential Ids
	      //
	      if(lQuantumNumber != lTemp)
		iPsp += 1;
	      pseudoPotentialId = iPsp + cumulativePotSplineId;
	      lTemp = lQuantumNumber;

	      const int globalPotSplineId = d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][0];
	      assert(lQuantumNumber == d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][1]);

	      std::vector<double> nonLocalProjectorBasisReal(maxkPoints*numberQuadraturePoints,0.0);
	      std::vector<double> nonLocalProjectorBasisImag(maxkPoints*numberQuadraturePoints,0.0);
	      std::vector<double> nonLocalPseudoConstant(numberQuadraturePoints,0.0);

	      /*if(iElemComp == 0)
		{
		  pcout<<"lQuantumNumber: "<<lQuantumNumber<<std::endl;
		  pcout<<"lQuantumNumber of Pot: "<<d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][1]<<std::endl;
		  pcout<<"mQuantumNumber: "<<mQuantumNumber<<std::endl;
		  pcout<<"Global Wave Spline Id: "<<globalWaveSplineId<<std::endl;
		  pcout<<"Global Pot Spline Id: "<<globalPotSplineId<<std::endl;
		  pcout<<"Outer Most Point: "<<d_outerMostPointPseudoWaveFunctionsData[globalWaveSplineId]<<std::endl;
		  }*/

	      double nlpValue = 0.0;
	      for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
		{

		  //MappingQ1<3,3> test;
		  //Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(iQuadPoint)));
		  Point<3> quadPoint=fe_values.quadrature_point(iQuadPoint);

		  for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
		    {

		      int chargeId = imageIdsList[iImageAtomCount];

		      //const Point & chargePoint = chargeId < numberGlobalCharges? d_nuclearContainer.getGlobalPoint(chargeId,meshId):
		      //d_nuclearContainer.getImagePoint(chargeId-numberGlobalCharges,meshId);

		      Point<3> chargePoint(0.0,0.0,0.0);

		      if(chargeId < numberGlobalCharges)
			{
			  chargePoint[0] = atomLocations[chargeId][2];
			  chargePoint[1] = atomLocations[chargeId][3];
			  chargePoint[2] = atomLocations[chargeId][4];
			}
		      else
			{
			  chargePoint[0] = d_imagePositions[chargeId-numberGlobalCharges][0];
			  chargePoint[1] = d_imagePositions[chargeId-numberGlobalCharges][1];
			  chargePoint[2] = d_imagePositions[chargeId-numberGlobalCharges][2];
			}


		      double x[3];

		      x[0] = quadPoint[0] - chargePoint[0];
		      x[1] = quadPoint[1] - chargePoint[1];
		      x[2] = quadPoint[2] - chargePoint[2];

		      //
		      // get the spherical coordinates from cartesian
		      //
		      double r,theta,phi;
		      pseudoUtils::convertCartesianToSpherical(x,r,theta,phi);


		      double radialWaveFunVal, sphericalHarmonicVal, radialPotFunVal, pseudoWaveFunctionValue, deltaVlValue;
		      if(r <= d_pspTail)//d_outerMostPointPseudoWaveFunctionsData[globalWaveSplineId])
			{
			  pseudoUtils::getRadialFunctionVal(r,
					       radialWaveFunVal,
					       &d_pseudoWaveFunctionSplines[globalWaveSplineId]);

			  pseudoUtils::getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);

			  pseudoWaveFunctionValue = radialWaveFunVal*sphericalHarmonicVal;

			  pseudoUtils::getRadialFunctionVal(r,
					       radialPotFunVal,
					       &d_deltaVlSplines[globalPotSplineId]);

			  deltaVlValue = radialPotFunVal;
			}
		      else
			{
			  pseudoWaveFunctionValue = 0.0;
			  deltaVlValue = 0.0;
			}


		      /*if(iElemComp == 0 && iQuadPoint == 0 && iPseudoWave == 0)
			{
			  std::cout<<"ChargeId : "<<chargeId<<std::endl;
			  std::cout<<"Coordinates: "<<chargePoint[0]<<" "<<chargePoint[1]<<" "<<chargePoint[2]<<std::endl;
			  std::cout<<"Distance : "<<r<<std::endl;
			  std::cout<<"DeltaVl: "<<deltaVlValue<<std::endl;
			  std::cout<<"JacTimesWeight: "<<fe_values.JxW(iQuadPoint)<<std::endl;
			  }*/

		      //
		      //kpoint loop
		      //
		      double pointMinusLatticeVector[3];
		      pointMinusLatticeVector[0] = x[0] + nuclearCoordinates[0];
		      pointMinusLatticeVector[1] = x[1] + nuclearCoordinates[1];
		      pointMinusLatticeVector[2] = x[2] + nuclearCoordinates[2];
		      for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)
			{
			  double angle = d_kPointCoordinates[3*kPoint+0]*pointMinusLatticeVector[0] + d_kPointCoordinates[3*kPoint+1]*pointMinusLatticeVector[1] + d_kPointCoordinates[3*kPoint+2]*pointMinusLatticeVector[2];
			  nonLocalProjectorBasisReal[maxkPoints*iQuadPoint + kPoint] += cos(angle)*pseudoWaveFunctionValue*deltaVlValue;
#ifdef USE_COMPLEX
			  nonLocalProjectorBasisImag[maxkPoints*iQuadPoint + kPoint] += -sin(angle)*pseudoWaveFunctionValue*deltaVlValue;
#endif
			}

		      nonLocalPseudoConstant[iQuadPoint] += pseudoWaveFunctionValue*deltaVlValue*pseudoWaveFunctionValue;
		    }//image atom loop

		  nlpValue += nonLocalPseudoConstant[iQuadPoint]*fe_values.JxW(iQuadPoint);

		}//end of quad loop

	      d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] += nlpValue;


	      //
	      // access shape functions values at quad points
	      //
	      //ElementQuadShapeFunctions shapeFunctions = dft::ShapeFunctionDataCalculator::v_shapeFunctions[meshId][quadratureRuleId][elementId];

	      for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		{
		  for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)
		    {
		      double tempReal = 0;
		      double tempImag = 0;
		      for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
			{
#ifdef USE_COMPLEX
			  tempReal += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
			  tempImag += nonLocalProjectorBasisImag[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
#else
			  d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode] += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
#endif
			}
#ifdef USE_COMPLEX
		      d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
		      d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(tempImag);
		      d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
		      d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(-tempImag);
#endif
		    }

		}

	    }//end of iPseudoWave loop


	}//element loop

      cumulativePotSplineId += numberAngularMomentumSpecificPotentials;
      cumulativeWaveSplineId += numberPseudoWaveFunctions;

    }//atom loop

  //
  //Add mpi accumulation
  //
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];

      for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	{

	  d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = Utilities::MPI::sum(d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave],mpi_communicator);

	  d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = 1.0/d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave];
	  if (dftParameters::verbosity==2)
	     pcout<<"The value of 1/nlpConst corresponding to atom and lCount "<<iAtom<<' '<<
	    iPseudoWave<<" is "<<d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave]<<std::endl;

	}


    }


}



