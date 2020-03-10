// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Krishnendu Ghosh
//

#include "stdafx.h"
#include <linalg.h>
#include <dftParameters.h>


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
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  QGauss<3>  quadratureHigh(C_num1DQuadPSP<FEOrder>());

  //FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  FEValues<3> fe_values(FE, dftParameters::useHigherQuadNLP?quadratureHigh:quadrature,
			update_values | update_JxW_values| update_quadrature_points);
  const unsigned int numberNodesPerElement  = FE.dofs_per_cell;
  const unsigned int numberQuadraturePoints = dftParameters::useHigherQuadNLP?quadratureHigh.size()
    :quadrature.size();


  //
  //get number of kPoints
  //
  const unsigned int maxkPoints = d_kPointWeights.size();


  //
  //preallocate element Matrices
  //
  d_nonLocalProjectorElementMatrices.clear();
  d_nonLocalProjectorElementMatricesConjugate.clear();
  d_nonLocalProjectorElementMatricesTranspose.clear();

  d_nonLocalProjectorElementMatrices.resize(numberNonLocalAtoms);
  d_nonLocalProjectorElementMatricesConjugate.resize(numberNonLocalAtoms);
  d_nonLocalProjectorElementMatricesTranspose.resize(numberNonLocalAtoms);

  std::vector<double> nonLocalProjectorBasisReal(maxkPoints*numberQuadraturePoints,0.0);
  std::vector<double> nonLocalProjectorBasisImag(maxkPoints*numberQuadraturePoints,0.0);

  int cumulativeWaveSplineId = 0;
  int waveFunctionId;
  //
  //
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int globalChargeIdNonLocalAtom =  d_nonLocalAtomGlobalChargeIds[iAtom];


      Point<3> nuclearCoordinates(atomLocations[globalChargeIdNonLocalAtom][2],atomLocations[globalChargeIdNonLocalAtom][3],atomLocations[globalChargeIdNonLocalAtom][4]);

      std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMapTrunc[globalChargeIdNonLocalAtom];

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


      //
      //allocate element Matrices
      //
      d_nonLocalProjectorElementMatrices[iAtom].resize(numberElementsInAtomCompactSupport);
      d_nonLocalProjectorElementMatricesConjugate[iAtom].resize(numberElementsInAtomCompactSupport);
      d_nonLocalProjectorElementMatricesTranspose[iAtom].resize(numberElementsInAtomCompactSupport);


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
	  d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp].resize(maxkPoints,
									       std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));

          std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesAtomElem=d_nonLocalProjectorElementMatrices[iAtom][iElemComp];

	  std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesConjugateAtomElem=d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp];

	  std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesTransposeAtomElem=d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp];

#else
	  d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);
	  d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);

          std::vector<double> & nonLocalProjectorElementMatricesAtomElem
	    =d_nonLocalProjectorElementMatrices[iAtom][iElemComp];


          std::vector<double> & nonLocalProjectorElementMatricesTransposeAtomElem
	    =d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp];


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


	      std::fill(nonLocalProjectorBasisReal.begin(),nonLocalProjectorBasisReal.end(),0.0);
	      std::fill(nonLocalProjectorBasisImag.begin(),nonLocalProjectorBasisImag.end(),0.0);

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
			  chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
			  chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
			  chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
			}


		      double x[3];

		      x[0] = quadPoint[0] - chargePoint[0];
		      x[1] = quadPoint[1] - chargePoint[1];
		      x[2] = quadPoint[2] - chargePoint[2];

		      //
		      // get the spherical coordinates from cartesian
		      //
		      //double r,theta,phi;
		      //pseudoUtils::convertCartesianToSpherical(x,r,theta,phi);


		      double sphericalHarmonicVal, radialProjVal, projectorFunctionValue;
		      if(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) <= d_outerMostPointPseudoProjectorData[globalWaveSplineId])
			{
                          double r,theta,phi;
		          pseudoUtils::convertCartesianToSpherical(x,r,theta,phi);


			  pseudoUtils::getRadialFunctionVal(r,
							    radialProjVal,
							    &d_pseudoWaveFunctionSplines[globalWaveSplineId]);

			  pseudoUtils::getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);

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
#ifdef USE_COMPLEX
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
			  const double shapeval=fe_values.shape_value(iNode,iQuadPoint);
			  const double jxw=fe_values.JxW(iQuadPoint);
#ifdef USE_COMPLEX
			  tempReal += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint+kPoint]*shapeval*jxw;
			  tempImag += nonLocalProjectorBasisImag[maxkPoints*iQuadPoint+kPoint]*shapeval*jxw;
#else

			  nonLocalProjectorElementMatricesAtomElem
			    [numberNodesPerElement*iPseudoWave + iNode]
			    += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint]*shapeval*jxw;

			  nonLocalProjectorElementMatricesTransposeAtomElem
			    [numberPseudoWaveFunctions*iNode+iPseudoWave]
			    += nonLocalProjectorBasisReal[maxkPoints*iQuadPoint]*shapeval*jxw;

#endif
			}
#ifdef USE_COMPLEX
		      nonLocalProjectorElementMatricesAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
		      nonLocalProjectorElementMatricesAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(tempImag);

		      nonLocalProjectorElementMatricesConjugateAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
		      nonLocalProjectorElementMatricesConjugateAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(-tempImag);

		      nonLocalProjectorElementMatricesTransposeAtomElem[kPoint]
			[numberPseudoWaveFunctions*iNode+iPseudoWave].real(tempReal);
		      nonLocalProjectorElementMatricesTransposeAtomElem[kPoint]
			[numberPseudoWaveFunctions*iNode+iPseudoWave].imag(tempImag);

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
  d_pseudoWaveFunctionIdToFunctionIdDetails.clear();
  d_numberPseudoAtomicWaveFunctions.clear();
  d_nonLocalAtomGlobalChargeIds.clear();

  //
  //this is the data structure used to store splines corresponding to projector information of various atom types
  //
  d_pseudoWaveFunctionSplines.clear();
  d_nonLocalPseudoPotentialConstants.clear();
  d_outerMostPointPseudoProjectorData.clear();

  // Store the Map between the atomic number and the projector details
  // (i.e. map from atomicNumber to a 2D vector storing atom specific projector Id and its corresponding
  // radial and angular Ids)
  // (atomicNumber->[projectorId][Global Spline Id, l quantum number, m quantum number]
  //
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToWaveFunctionIdDetails;
  std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToPotentialIdMap;
  std::map<unsigned int, std::vector< std::vector<double> >> denominatorData;
  const double truncationTol = 1e-12;


  //
  // Store the number of unique splines encountered so far
  //
  unsigned int cumulativeSplineId    = 0;
  unsigned int cumulativePotSplineId = 0;
  std::map<unsigned int,std::vector<int>>  projector ;


  for(std::set<unsigned int>::iterator it = atomTypes.begin(); it != atomTypes.end(); ++it)
    {
      char pseudoAtomDataFile[256];
      sprintf(pseudoAtomDataFile, "temp/z%u/PseudoAtomDat",*it);


      unsigned int atomicNumber = *it;

      if(dftParameters::verbosity >= 2)
	pcout<<"Reading data from file: "<<pseudoAtomDataFile<<std::endl;

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
      // store the number of single-atom projectors associated with the current atomic number
      //
      unsigned int numberAtomicWaveFunctions ; //numberStates;

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

      if (dftParameters::verbosity>=2)
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
		    {
		      splineFunctionIds.insert(Id);
		      projector[(*it)][i] = Id ;
		    }

		  radAndAngularFunctionId[count] = Id;

		}
	      //if (count==3) {
	      // Id = atoi(dummyString.c_str());
	      // projector[(*it)][i] = Id ;
	      // }
	      if(count>3)
		{
		  std::cerr<<"Invalid argument in the SingleAtomData file"<<std::endl;
		  exit(-1);
		}

	      count++;

	    }

	

	  radAndAngularFunctionId[0] += cumulativeSplineId;

	  if (dftParameters::verbosity>=2)
	    {
	      pcout << "Radial and Angular Functions Ids: " << radAndAngularFunctionId[0] << " " << radAndAngularFunctionId[1] << " " << radAndAngularFunctionId[2] << std::endl;
	      pcout << "Projector Id: " << projector[(*it)][i] << std::endl;
	    }

	}

      if (dftParameters::verbosity>=2)
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
      std::vector<double> outerMostRadialPointProjector(splineFunctionIds.size());

      //pcout << "Number of radial Projector wavefunctions for atomic number " << atomicNumber << " is: " << radFunctionIds.size() << std::endl;

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
	

	  char projRadialFunctionFileName[512];
	  sprintf(projRadialFunctionFileName, "temp/z%u/%s",*it,tempProjRadialFunctionFileName.c_str());

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

          for (int iProj = 0; iProj<numProj; ++iProj) 
	    {
	      double xData[numRows];
	      double yData[numRows];

	      unsigned int maxRowId = 0;
	      for (int iRow = 0; iRow < numRows; ++iRow)
		{
		  xData[iRow] = radialFunctionData[iRow][0];
		  yData[iRow] = radialFunctionData[iRow][iProj+1];

		  if(std::abs(yData[iRow])>truncationTol)
		    maxRowId = iRow;

		}

	      outerMostRadialPointProjector[projId] = xData[maxRowId+10];

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
      d_outerMostPointPseudoProjectorData.insert(d_outerMostPointPseudoProjectorData.end(),outerMostRadialPointProjector.begin(),outerMostRadialPointProjector.end());


      //
      // 2D vector to store the radial coordinate and its corresponding
      // function value
      //
      std::vector<std::vector<double> > denominator(0);

      //
      //read the radial function file
      //
      std::string tempDenominatorDataFileName;
      char denominatorDataFileName[256];

      //
      //read the pseudo data file name
      //
      readPseudoDataFileNames >> tempDenominatorDataFileName ;
      sprintf(denominatorDataFileName, "temp/z%u/%s", *it, tempDenominatorDataFileName.c_str());
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

    }//end of iCharge loop

  d_nonLocalAtomGlobalChargeIds = nonLocalAtomGlobalChargeIds;
  int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();

  if (dftParameters::verbosity>=2)
    pcout<<"Number of Nonlocal Atoms: " <<d_nonLocalAtomGlobalChargeIds.size()<<std::endl;

  d_nonLocalPseudoPotentialConstants.resize(numberNonLocalAtoms);

  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {

      int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
      d_nonLocalPseudoPotentialConstants[iAtom].resize(numberPseudoWaveFunctions,0.0);
      /*
      //
      char pseudoAtomDataFile[256];
      sprintf(pseudoAtomDataFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/PseudoAtomData", DFT_PATH.c_str(), atomLocations[iAtom][0]);
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
      sprintf(denominatorDataFileName, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/%s", DFT_PATH.c_str(),atomLocations[iAtom][0], tempDenominatorDataFileName.c_str());

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
#ifdef DEBUG
	  if (dftParameters::verbosity>=4)
	    pcout<<"The value of 1/nlpConst corresponding to atom and lCount "<<iAtom<<' '<<
	      iPseudoWave<<" is "<<d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave]<<std::endl;
#endif
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
  const double nlpTolerance = 1e-8;


  //
  //pre-allocate data structures that stores the sparsity of deltaVl
  //
  d_sparsityPattern.clear();
  d_elementIteratorsInAtomCompactSupport.clear();
  d_elementIdsInAtomCompactSupport.clear();
  d_elementOneFieldIteratorsInAtomCompactSupport.clear();

  //d_sparsityPattern.resize(numberNonLocalAtoms);
  d_elementIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_elementIdsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_elementOneFieldIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
  d_nonLocalAtomIdsInCurrentProcess.clear();

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
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  //FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  FEValues<3> fe_values(FE, quadrature, update_quadrature_points);
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
  std::vector<int> sparsityPattern(numberElements,-1);
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {

      //
      //temp variables
      //
      int matCount = 0;
      bool isAtomIdInProcessor=false;

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
      std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMapTrunc[globalChargeIdNonLocalAtom];

      //
      //resize the data structure corresponding to sparsity pattern
      //
      //std::vector<int> sparsityPattern;(numberElements,-1);
      //d_sparsityPattern[iAtom].resize(numberElements,-1);

      if (imageIdsList.size()!=0)
      {
              std::fill(sparsityPattern.begin(),sparsityPattern.end(),-1);
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
		      iElem += 1;
                      bool isSkipCell=true;
		      for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
			  {

			    int chargeId = imageIdsList[iImageAtomCount];

			    Point<3> chargePoint(0.0,0.0,0.0);

			    if(chargeId < numberGlobalCharges)
			      {
				chargePoint[0] = atomLocations[chargeId][2];
				chargePoint[1] = atomLocations[chargeId][3];
				chargePoint[2] = atomLocations[chargeId][4];
			      }
			    else
			      {
				chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
				chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
				chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
			      }

			      if (chargePoint.distance(cell->center())<4.0)
                              {
                                 isSkipCell=false;
                                 break;
                              }
			   }

                      if (isSkipCell)
                         continue;

		      //compute the values for the current element
		      fe_values.reinit(cell);

		      int lTemp = 1000 ;

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
				Point<3> quadPoint=fe_values.quadrature_point(iQuadPoint);

				for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
				  {

				    int chargeId = imageIdsList[iImageAtomCount];

				    Point<3> chargePoint(0.0,0.0,0.0);

				    if(chargeId < numberGlobalCharges)
				      {
					chargePoint[0] = atomLocations[chargeId][2];
					chargePoint[1] = atomLocations[chargeId][3];
					chargePoint[2] = atomLocations[chargeId][4];
				      }
				    else
				      {
					chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
					chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
					chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
				      }

				    double r = quadPoint.distance(chargePoint);
				    double radialProjVal;

				    if(r <= d_outerMostPointPseudoProjectorData[globalWaveSplineId])
				      pseudoUtils::getRadialFunctionVal( r, radialProjVal, &d_pseudoWaveFunctionSplines[globalWaveSplineId] );
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

		      if(sparseFlag==1) {
			sparsityPattern[iElem] = matCount;
			d_elementIteratorsInAtomCompactSupport[iAtom].push_back(cellEigen);
			d_elementIdsInAtomCompactSupport[iAtom].push_back(iElem);
			d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].push_back(cell);
			matCount += 1;
			isAtomIdInProcessor=true;
		      }

		    }
		}//cell loop
      }
      cumulativeSplineId += numberPseudoWaveFunctions;
#ifdef DEBUG
      if (dftParameters::verbosity>=4)
	pcout<<"No.of non zero elements in the compact support of atom "<<iAtom<<" is "<<d_elementIteratorsInAtomCompactSupport[iAtom].size()<<std::endl;
#endif

      if (isAtomIdInProcessor)
      {
	d_nonLocalAtomIdsInCurrentProcess.push_back(iAtom);
        d_sparsityPattern[iAtom]=sparsityPattern;
      }

    }//atom loop

  d_nonLocalAtomIdsInElement.clear();
  d_nonLocalAtomIdsInElement.resize(numberElements);


  for(int iElem = 0; iElem < numberElements; ++iElem)
    {
      for(int iAtom = 0; iAtom < d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
	  if(d_sparsityPattern[d_nonLocalAtomIdsInCurrentProcess[iAtom]][iElem] >= 0)
	    d_nonLocalAtomIdsInElement[iElem].push_back(d_nonLocalAtomIdsInCurrentProcess[iAtom]);
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
  dealii::LinearAlgebra::distributed::Vector<std::complex<double> > vec(d_locallyOwnedProjectorIdsCurrentProcess,
                                                                   d_ghostProjectorIdsCurrentProcess,
                                                                   mpi_communicator);
#else
  dealii::LinearAlgebra::distributed::Vector<double > vec(d_locallyOwnedProjectorIdsCurrentProcess,
                                                     d_ghostProjectorIdsCurrentProcess,
                                                     mpi_communicator);
#endif
  vec.update_ghost_values();
  d_projectorKetTimesVectorPar.resize(1);
  d_projectorKetTimesVectorPar[0].reinit(vec);
}
