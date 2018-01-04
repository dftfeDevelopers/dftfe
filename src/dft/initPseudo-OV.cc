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
// @author Krishnendu Ghosh(2017), adapted from initPseudo.cc
//

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include "stdafx.h"
#include "linalg.h"
#include "../../include/dftParameters.h"


using namespace dftParameters ;

template<unsigned int FEOrder>
void dftClass<FEOrder>::computeElementalOVProjectorKets()
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
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int numberNodesPerElement  = FE.dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();
  

  //
  //get number of kPoints
  //
  int maxkPoints = d_maxkPoints;
  

  //
  //preallocate element Matrices
  //
  d_nonLocalProjectorElementMatrices.clear();
  d_nonLocalProjectorElementMatrices.resize(numberNonLocalAtoms);
  int cumulativeWaveSplineId = 0;
  int waveFunctionId;
  //
  pcout << " pspTail  " << pspTail << std::endl ;
  //
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

      pcout<<"Number of Pseudo wavefunctions: "<<std::endl;
      pcout<<numberPseudoWaveFunctions<<std::endl;

      //
      //allocate element Matrices
      //
      d_nonLocalProjectorElementMatrices[iAtom].resize(numberElementsInAtomCompactSupport);
	
      for(int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport; ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom][iElemComp];

	  //compute values for the current elements
	  fe_values.reinit(cell);

#ifdef ENABLE_PERIODIC_BC
	  d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(maxkPoints,
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
	      lTemp = lQuantumNumber;


	      std::vector<double> nonLocalProjectorBasisReal(maxkPoints*numberQuadraturePoints,0.0);
	      std::vector<double> nonLocalProjectorBasisImag(maxkPoints*numberQuadraturePoints,0.0);

	      double nlpValue = 0.0;
	      for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
		{

		  MappingQ1<3,3> test;
		  Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(iQuadPoint)));

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
		      convertCartesianToSpherical(x,r,theta,phi);
		    

		      double sphericalHarmonicVal, radialProjVal, projectorFunctionValue;
		      if(r <= pspTail)//d_outerMostPointPseudoWaveFunctionsData[globalWaveSplineId])
			{
			  getRadialFunctionVal(r,
					       radialProjVal,
					       &d_pseudoWaveFunctionSplines[globalWaveSplineId]);

			  getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);
			
			  projectorFunctionValue = radialProjVal*sphericalHarmonicVal;

			}
		      else
			{
			  projectorFunctionValue = 0.0;
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
			  nonLocalProjectorBasisReal[maxkPoints*iQuadPoint + kPoint] += cos(angle)*projectorFunctionValue;
#ifdef ENABLE_PERIODIC_BC
			  nonLocalProjectorBasisImag[maxkPoints*iQuadPoint + kPoint] += -sin(angle)*projectorFunctionValue;
#endif
			}

		    }//image atom loop

		}//end of quad loop
	

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
#ifdef ENABLE_PERIODIC_BC
			  tempReal += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
			  tempImag += nonLocalProjectorBasisImag[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
#else
			  d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode] += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint+kPoint]*fe_values.shape_value(iNode,iQuadPoint)*fe_values.JxW(iQuadPoint);
#endif
			}
#ifdef ENABLE_PERIODIC_BC
		      d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
		      d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(tempImag);
#endif		      
		    }

		}

	    }//end of iPseudoWave loop
	

	}//element loop

      cumulativeWaveSplineId += numberPseudoWaveFunctions;

    }//atom loop

  //
  //Add mpi accumulation
  //

}
template<unsigned int FEOrder>
void dftClass<FEOrder>::initNonLocalPseudoPotential_OV()
{

  // Store the Map between the atomic number and the waveFunction details
  // (i.e. map from atomicNumber to a 2D vector storing atom specific wavefunction Id and its corresponding 
  // radial and angular Ids)
  // (atomicNumber->[atomicWaveFunctionId][Global Spline Id, l quantum number, m quantum number]
  //
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToWaveFunctionIdDetails;
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToPotentialIdMap;
  std::map<unsigned int, std::vector< std::vector<double> >> denominatorData;
  
  //
  // Store the number of unique splines encountered so far 
  //
  unsigned int cumulativeSplineId    = 0;
  unsigned int cumulativePotSplineId = 0;
  std::map<unsigned int,std::vector<int>>  projector ;


  for(std::set<unsigned int>::iterator it = atomTypes.begin(); it != atomTypes.end(); ++it)
    {
      char pseudoAtomDataFile[256];
      sprintf(pseudoAtomDataFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/PseudoAtomData", currentPath.c_str(), *it);


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
      unsigned int numberAtomicWaveFunctions ; //numberStates;

      //
      // read number of single-atom wavefunctions
      //
      if(readPseudoDataFileNames.is_open())//{
	readPseudoDataFileNames >> numberAtomicWaveFunctions;
	//readPseudoDataFileNames >> numberStates;
        //}

      //
      // resize atomicFunctionIdDetails
      //
      atomicFunctionIdDetails.resize(numberAtomicWaveFunctions);

      //
      // Skip the rest in the first line and proceed to next line
      //
      readPseudoDataFileNames.ignore();
      pcout << "Number of projectors for atom with Z: " << atomicNumber<<" is " << numberAtomicWaveFunctions << std::endl;

      //
      //string to store each line of the file
      //
      std::string readLine;
  
      //
      // set to store the radial(spline) function Ids
      //
      std::set<int> radFunctionIds, splineFunctionIds;
      std::vector<int> lquantum(numberAtomicWaveFunctions), mquantum(numberAtomicWaveFunctions) ;
      projector[(*it)].resize(numberAtomicWaveFunctions) ;
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
		  if(count == 1) 
		    radFunctionIds.insert(Id); 

                  if(count == 0) 
		    splineFunctionIds.insert(Id); 

		  radAndAngularFunctionId[count] = Id;
	
		}
	      if (count==3) {
		 Id = atoi(dummyString.c_str()); 
		 projector[(*it)][i] = Id ;
		 }
	      if (count>3)
		{

		  std::cerr<<"Invalid argument in the SingleAtomData file"<<std::endl;
		  exit(-1);
		}
 	
	      count++;     
   
	    } 
       
	  //
	  // Add the cumulativeSplineId to radialId
	  //
	  /*lquantum[i] = radAndAngularFunctionId[1] ;
	  mquantum[i] = radAndAngularFunctionId[2] ;
	  if (i==0)
	      projector[(*it)][i] =  0 ;
	  if (i>0 && lquantum[i]==lquantum[i-1] && mquantum[i]==mquantum[i-1])
		projector[(*it)][i] = projector[(*it)][i-1]+1;
	  if(i>0 && lquantum[i]!=lquantum[i-1])
		projector[(*it)][i] = projector[(*it)][i-1]+1;
	  if (i>0 && lquantum[i]==lquantum[i-1] && mquantum[i]!=mquantum[i-1])
		projector[(*it)][i] = projector[(*it)][i-1];*/
		
	  radAndAngularFunctionId[0] += cumulativeSplineId;
	    
	  pcout << "Radial and Angular Functions Ids: " << radAndAngularFunctionId[0] << " " << radAndAngularFunctionId[1] << " " << radAndAngularFunctionId[2] << std::endl;
	  pcout << "Projector Id: " << projector[(*it)][i] << std::endl;

	}

	pcout << " splineFunctionIds.size() " << splineFunctionIds.size() << std::endl;

      //
      // map the atomic number to atomicNumberToFunctionIdDetails
      //
      atomicNumberToWaveFunctionIdDetails[atomicNumber] = atomicFunctionIdDetails;

      //
      // update cumulativeSplineId
      //
      cumulativeSplineId += splineFunctionIds.size();

      //
      // store the splines for the current atom type
      //
      std::vector<alglib::spline1dinterpolant> atomicSplines(splineFunctionIds.size());
      std::vector<alglib::real_1d_array> atomicRadialNodes(splineFunctionIds.size());
      std::vector<alglib::real_1d_array> atomicRadialFunctionNodalValues(splineFunctionIds.size());
      std::vector<double> outerMostRadialPointWaveFunction(splineFunctionIds.size());
      
      //pcout << "Number radial Pseudo wavefunctions for atomic number " << atomicNumber << " is: " << radFunctionIds.size() << std::endl; 

      //
      // string to store the radial function file name
      //
      std::string tempProjRadialFunctionFileName;
  
      unsigned int projId = 0, numProj ;

      for(unsigned int i = 0; i < radFunctionIds.size(); ++i)
	{
          
	  //
	  // get the radial function file name (name local to the directory) 
	  //
	  readPseudoDataFileNames >> tempProjRadialFunctionFileName;
          readPseudoDataFileNames >> numProj;
	  //readPseudoDataFileNames >> numProj ;

	  char projRadialFunctionFileName[512];
	  sprintf(projRadialFunctionFileName, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/%s", currentPath.c_str(),*it,tempProjRadialFunctionFileName.c_str());

	  //
	  // 2D vector to store the radial coordinate and its corresponding
	  // function value
	  std::vector< std::vector<double> > radialFunctionData(0);
         
	  //
	  //read the radial function file
	  //
	  dftUtils::readFile(numProj+1,radialFunctionData,projRadialFunctionFileName);

        
	  int numRows = radialFunctionData.size();
   
	  //std::cout << "Number of Rows: " << numRows << std::endl;
    
          for (int iProj = 0; iProj<numProj; ++iProj) {
	  double xData[numRows];
	  double yData[numRows];
  
	  for (int iRow = 0; iRow < numRows; ++iRow)
	    {
	      xData[iRow] = radialFunctionData[iRow][0];
	      yData[iRow] = radialFunctionData[iRow][iProj+1] ;
	    }

	  outerMostRadialPointWaveFunction[projId] = xData[numRows - 1];
  
	  alglib::real_1d_array & x = atomicRadialNodes[projId];
	  atomicRadialNodes[projId].setcontent(numRows, xData);
  
	  alglib::real_1d_array & y = atomicRadialFunctionNodalValues[projId];
	  atomicRadialFunctionNodalValues[projId].setcontent(numRows, yData);
  
	  alglib::ae_int_t natural_bound_type = 1;
	  alglib::spline1dbuildcubic(atomicRadialNodes[projId],
				     atomicRadialFunctionNodalValues[projId],
				     numRows,
				     natural_bound_type,
				     0.0,
				     natural_bound_type,
				     0.0,
				     atomicSplines[projId]);

  	  projId++ ;
	  }
	}
        d_pseudoWaveFunctionSplines.insert(d_pseudoWaveFunctionSplines.end(), atomicSplines.begin(), atomicSplines.end()); 
        //
     
	  //
	  // 2D vector to store the radial coordinate and its corresponding
	  // function value
	  std::vector< std::vector<double> > denominator(0);
         
	  //
	  //read the radial function file
	  //
          std::string tempDenominatorDataFileName ;
	  char denominatorDataFileName[256];
	  //
	  readPseudoDataFileNames >> tempDenominatorDataFileName ;
	  sprintf(denominatorDataFileName, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/%s", currentPath.c_str(),*it, tempDenominatorDataFileName.c_str());
	  dftUtils::readFile(projId,denominator,denominatorDataFileName);
	  denominatorData[(*it)] = denominator ;

        readPseudoDataFileNames.close() ;
  }
   
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


      //
      // Get the number of functions associated with the current nucleus
      //
      unsigned int numberAtomicWaveFunctions = atomicFunctionIdDetails.size();

      if(numberAtomicWaveFunctions > 0 )
	{
	  nonLocalAtomGlobalChargeIds.push_back(iCharge);
	  d_numberPseudoAtomicWaveFunctions.push_back(numberAtomicWaveFunctions);
	}
	  
  
      //
      // Add the atomic wave function details to the global wave function vectors
      //
      for(unsigned iAtomWave = 0; iAtomWave < numberAtomicWaveFunctions; ++iAtomWave)
	{
	  d_pseudoWaveFunctionIdToFunctionIdDetails.push_back(atomicFunctionIdDetails[iAtomWave]);
	}


      //
      // insert the master charge Id into the map first
      //
      globalChargeIdToImageIdMap[iCharge].push_back(iCharge);
  
    }//end of iCharge loop

  d_nonLocalAtomGlobalChargeIds = nonLocalAtomGlobalChargeIds;
  int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();
  
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
  d_nonLocalPseudoPotentialConstants.resize(numberNonLocalAtoms);

    for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
		
          int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
	   d_nonLocalPseudoPotentialConstants[iAtom].resize(numberPseudoWaveFunctions,0.0);
	  /*
          //
	  char pseudoAtomDataFile[256];
          sprintf(pseudoAtomDataFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/PseudoAtomData", currentPath.c_str(), atomLocations[iAtom][0]);
	  //
	  std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
          if(readPseudoDataFileNames.is_open()){
	  while (!readPseudoDataFileNames.eof()) {
	     std::getline(readPseudoDataFileNames, readLine);       
	     std::istringstream lineString(readLine);  
	     while(lineString >> tempDenominatorDataFileName)               
	            pcout << tempDenominatorDataFileName.c_str() << std::endl ;
	  }
	  }
          //std::cout << c;
          //while (!readPseudoDataFileNames.eof())
	  //        readPseudoDataFileNames >> tempDenominatorDataFileName;
	  pcout << tempDenominatorDataFileName.c_str() << std::endl ;
	  char denominatorDataFileName[256];
	  sprintf(denominatorDataFileName, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/%s", currentPath.c_str(),atomLocations[iAtom][0], tempDenominatorDataFileName.c_str());
     
	  //
	  // 2D vector to store the radial coordinate and its corresponding
	  // function value
	  std::vector< std::vector<double> > denominatorData(0);
         
	  //
	  //read the radial function file
	  //
	  readFile(numberPseudoWaveFunctions,denominatorData,denominatorDataFileName);*/
      
      for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	{
	  d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = denominatorData[atomLocations[iAtom][0]][projector[atomLocations[iAtom][0]][iPseudoWave]][projector[atomLocations[iAtom][0]][iPseudoWave]];
	  //d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = 1.0/d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave];
	  pcout<<"The value of 1/nlpConst corresponding to atom and lCount "<<iAtom<<' '<<
	    iPseudoWave<<" is "<<d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave]<<std::endl;	
	  
	}


    }


  return;


}
template<unsigned int FEOrder>
void dftClass<FEOrder>::computeSparseStructureNonLocalProjectors_OV()
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
  d_elementOneFieldIteratorsInAtomCompactSupport.clear();

  d_sparsityPattern.resize(numberNonLocalAtoms);
  d_elementIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_elementOneFieldIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);

  //
  //loop over nonlocal atoms
  //
  unsigned int sparseFlag = 0;
  int cumulativeSplineId = 0;
  int waveFunctionId;


  //
  //get number of global charges
  //
  unsigned int numberGlobalCharges  = atomLocations.size();

  //
  //get FE data structures
  //
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int numberQuadraturePoints = quadrature.size();
  //const unsigned int numberElements         = triangulation.n_locally_owned_active_cells();
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

      //
      //
      int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];

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
      pcout << " pspTail adjusted to " << pspTail << std::endl ;
      for(; cell != endc; ++cell,++cellEigen)
	{
	  if(cell->is_locally_owned())
	    {

	      //compute the values for the current element
	      fe_values.reinit(cell);

	      iElem += 1;
	      /*int lTemp = 1000 ;
	      
	      for(int iPsp = 0; iPsp < numberPseudoWaveFunctions; ++iPsp)
		{
		  sparseFlag = 0;
		  waveFunctionId = iPsp + cumulativeSplineId;
		  const int globalWaveSplineId = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
		  const int lQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
		  //
		  if(lQuantumNumber != lTemp) {
		     lTemp = lQuantumNumber ;
		  for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
		    {
		      MappingQ1<3,3> test;
		      Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(iQuadPoint)));
		      
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
			  double radialProjVal;

			  if(r <= pspTail)//d_outerMostPointPseudoPotData[globalSplineId])
			      getRadialFunctionVal( r, radialProjVal, &d_pseudoWaveFunctionSplines[globalWaveSplineId] );
			  else
			      radialProjVal = 0.0;
		      
			  if(fabs(radialProjVal) >= nlpTolerance)
			    {
			      sparseFlag = 1;
			      break;
			    }
			}//imageAtomLoop

		      if(sparseFlag == 1)
			break;

		    }//quadrature loop
		
		  }

		  if(sparseFlag == 1)
		    break;

		}//iPsp loop ("l" loop) 
	        */
	      //if(sparseFlag==1) {
		d_sparsityPattern[iAtom][iElem] = matCount;
		d_elementIteratorsInAtomCompactSupport[iAtom].push_back(cellEigen);
		d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].push_back(cell);
		matCount += 1;
	      //}

	    }
	}//cell loop

      cumulativeSplineId += numberPseudoWaveFunctions;

      pcout<<"No.of non zero elements in the compact support of atom "<<iAtom<<" is "<<d_elementIteratorsInAtomCompactSupport[iAtom].size()<<std::endl;

    }//atom loop

  d_nonLocalAtomIdsInElement.resize(numberElements);


  for(int iElem = 0; iElem < numberElements; ++iElem)
    {
      for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{
	  if(d_sparsityPattern[iAtom][iElem] >= 0)
	    d_nonLocalAtomIdsInElement[iElem].push_back(iAtom);
	}
    }

}
