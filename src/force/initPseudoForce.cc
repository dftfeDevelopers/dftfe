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
// @author Sambit Das (2017)
//
#include "pseudoForceUtils.cc"

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initLocalPseudoPotentialForce()
{
  d_gradPseudoVLoc.clear();
  d_gradPseudoVLocAtoms.clear();

  //
  //Reading single atom rho initial guess
  //
  std::map<unsigned int, alglib::spline1dinterpolant> pseudoSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > pseudoPotentialData;
  std::map<unsigned int, double> outerMostPointPseudo;


  //
  //loop over atom types
  //
  for(std::set<unsigned int>::iterator it=dftPtr->atomTypes.begin(); it!=dftPtr->atomTypes.end(); it++)
    {
      char pseudoFile[256];
      if (dftParameters::pseudoProjector==2)
	  sprintf(pseudoFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/locPot.dat", DFT_PATH,*it);
      else
          sprintf(pseudoFile, "%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/locPot.dat", DFT_PATH,*it);
      //pcout<<"Reading Local Pseudo-potential data from: " <<pseudoFile<<std::endl;
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
    }

  //
  //Initialize pseudopotential
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (dftPtr->FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points = quadrature_formula.size();


  const int numberGlobalCharges=dftPtr->atomLocations.size();
  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = dftPtr->d_imageIds.size();
  //MappingQ1<3,3> test;
  //
  //loop over elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  //compute values for the current elements
	  fe_values.reinit(cell);
	  d_gradPseudoVLoc[cell->id()]=std::vector<double>(n_q_points*3);
	  std::vector<Tensor<1,3,double> > gradPseudoValContribution(n_q_points);
	  //loop over atoms
	  for (unsigned int n=0; n<dftPtr->atomLocations.size(); n++)
	  {
              Point<3> atom(dftPtr->atomLocations[n][2],dftPtr->atomLocations[n][3],dftPtr->atomLocations[n][4]);
	      bool isPseudoDataInCell=false;
	      //loop over quad points
	      for (unsigned int q = 0; q < n_q_points; ++q)
	      {

		  Point<3> quadPoint=fe_values.quadrature_point(q);
		  double distanceToAtom = quadPoint.distance(atom);
		  double value,firstDer,secondDer;
		  if(distanceToAtom <= dftPtr->d_pspTail)//outerMostPointPseudo[atomLocations[n][0]])
		    {
		      alglib::spline1ddiff(pseudoSpline[dftPtr->atomLocations[n][0]], distanceToAtom,value,firstDer,secondDer);
		      isPseudoDataInCell=true;
		    }
		  else
		    {
		      firstDer= (dftPtr->atomLocations[n][1])/distanceToAtom/distanceToAtom;
		    }
		    gradPseudoValContribution[q]=firstDer*(quadPoint-atom)/distanceToAtom;
		    d_gradPseudoVLoc[cell->id()][q*3+0]+=gradPseudoValContribution[q][0];
		    d_gradPseudoVLoc[cell->id()][q*3+1]+=gradPseudoValContribution[q][1];
		    d_gradPseudoVLoc[cell->id()][q*3+2]+=gradPseudoValContribution[q][2];
	      }//loop over quad points
	      if (isPseudoDataInCell){
	          d_gradPseudoVLocAtoms[n][cell->id()]=std::vector<double>(n_q_points*3);
	          for (unsigned int q = 0; q < n_q_points; ++q)
	          {
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+0]=gradPseudoValContribution[q][0];
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+1]=gradPseudoValContribution[q][1];
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+2]=gradPseudoValContribution[q][2];
	          }
	      }
	  }//loop pver atoms

	  //loop over image charges
	  for(unsigned int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
	  {
	      Point<3> imageAtom(dftPtr->d_imagePositions[iImageCharge][0],dftPtr->d_imagePositions[iImageCharge][1],dftPtr->d_imagePositions[iImageCharge][2]);
	      bool isPseudoDataInCell=false;
	      //loop over quad points
	      for (unsigned int q = 0; q < n_q_points; ++q)
	      {

		  Point<3> quadPoint=fe_values.quadrature_point(q);
		  double distanceToAtom = quadPoint.distance(imageAtom);
		  int masterAtomId = dftPtr->d_imageIds[iImageCharge];
		  double value,firstDer,secondDer;
		  if(distanceToAtom <= dftPtr->d_pspTail)//outerMostPointPseudo[atomLocations[masterAtomId][0]])
		    {
		      alglib::spline1ddiff(pseudoSpline[dftPtr->atomLocations[masterAtomId][0]], distanceToAtom,value,firstDer,secondDer);
		      isPseudoDataInCell=true;
		    }
		  else
		    {
		      firstDer= (dftPtr->atomLocations[masterAtomId][1])/distanceToAtom/distanceToAtom;
		    }
		    gradPseudoValContribution[q]=firstDer*(quadPoint-imageAtom)/distanceToAtom;
		    d_gradPseudoVLoc[cell->id()][q*3+0]+=gradPseudoValContribution[q][0];
		    d_gradPseudoVLoc[cell->id()][q*3+1]+=gradPseudoValContribution[q][1];
		    d_gradPseudoVLoc[cell->id()][q*3+2]+=gradPseudoValContribution[q][2];
	      }//loop over quad points
	      if (isPseudoDataInCell){
	          d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()]=std::vector<double>(n_q_points*3);
	          for (unsigned int q = 0; q < n_q_points; ++q)
	          {
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+0]=gradPseudoValContribution[q][0];
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+1]=gradPseudoValContribution[q][1];
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+2]=gradPseudoValContribution[q][2];
	          }
	      }
	   }//loop over image charges
	}//cell locally owned check
    }//cell loop

}

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeElementalNonLocalPseudoDataForce()
{
  //
  //get the number of non-local atoms
  //
  const unsigned int numberNonLocalAtoms = dftPtr->d_nonLocalAtomGlobalChargeIds.size();

  //
  //get number of global charges
  //
  const unsigned int numberGlobalCharges  = dftPtr->atomLocations.size();


  //
  //get FE data structures
  //
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values(dftPtr->FE, quadrature, update_quadrature_points);
  const unsigned int numberNodesPerElement  = dftPtr->FE.dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();

  //
  //get number of kPoints
  //
  const unsigned int numkPoints = dftPtr->d_kPointWeights.size();

  //
  //clear existing data
  //
  d_nonLocalPSP_ZetalmDeltaVl.clear();
#ifdef ENABLE_PERIODIC_BC
  d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint.clear();
  d_nonLocalPSP_gradZetalmDeltaVl_KPoint.clear();
  d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.clear();
#else
  d_nonLocalPSP_gradZetalmDeltaVl.clear();
#endif
  //
  //
  int cumulativePotSplineId = 0;
  int cumulativeWaveSplineId = 0;
  int waveFunctionId;
  int pseudoPotentialId;
  unsigned int count=0;
  const unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
  d_nonLocalPSP_ZetalmDeltaVl.resize(numNonLocalAtomsCurrentProcess);
#ifdef ENABLE_PERIODIC_BC
  d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint.resize(numNonLocalAtomsCurrentProcess);
  d_nonLocalPSP_gradZetalmDeltaVl_KPoint.resize(numNonLocalAtomsCurrentProcess);
  d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.resize(numNonLocalAtomsCurrentProcess);
#else
  d_nonLocalPSP_gradZetalmDeltaVl.resize(numNonLocalAtomsCurrentProcess);
#endif

  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[iAtom];


      Point<3> nuclearCoordinates(dftPtr->atomLocations[globalChargeIdNonLocalAtom][2],dftPtr->atomLocations[globalChargeIdNonLocalAtom][3],dftPtr->atomLocations[globalChargeIdNonLocalAtom][4]);

      std::vector<int> & imageIdsList = dftPtr->d_globalChargeIdToImageIdMap[globalChargeIdNonLocalAtom];

      //
      //get the number of elements in the compact support of the current nonlocal atom
      //
      const unsigned int numberElementsInAtomCompactSupport = dftPtr->d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();

      //
      //get the number of pseudowavefunctions for the current nonlocal atoms
      //
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      const unsigned int numberAngularMomentumSpecificPotentials = dftPtr->d_numberPseudoPotentials[iAtom];

      //
      //allocate
      //
      if (numberElementsInAtomCompactSupport !=0)
      {
            d_nonLocalPSP_ZetalmDeltaVl[count].resize(numberPseudoWaveFunctions);
#ifdef ENABLE_PERIODIC_BC
            d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count].resize(numberPseudoWaveFunctions);
            d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count].resize(numberPseudoWaveFunctions);
	    d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count].resize(numberPseudoWaveFunctions);
#else
            d_nonLocalPSP_gradZetalmDeltaVl[count].resize(numberPseudoWaveFunctions);
#endif
      }

      for(unsigned int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport; ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = dftPtr->d_elementOneFieldIteratorsInAtomCompactSupport[iAtom][iElemComp];

	  //compute values for the current elements
	  fe_values.reinit(cell);

	  int iPsp = -1;
	  int lTemp = 1e5;

	  for(unsigned int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	    {
#ifdef ENABLE_PERIODIC_BC
	      d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*2);
	      d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*2);
	      d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*2);
              d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*C_DIM*2);
#else
	      d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numberQuadraturePoints);
	      d_nonLocalPSP_gradZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numberQuadraturePoints*C_DIM);
#endif

	      waveFunctionId = iPseudoWave + cumulativeWaveSplineId;
	      const int globalWaveSplineId = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
	      const int lQuantumNumber = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
	      const int mQuantumNumber = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][2];

	      //
	      //access pseudoPotential Ids
	      //
	      if(lQuantumNumber != lTemp)
		iPsp += 1;
	      pseudoPotentialId = iPsp + cumulativePotSplineId;
	      lTemp = lQuantumNumber;

	      const int globalPotSplineId = dftPtr->d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][0];
	      assert(lQuantumNumber == dftPtr->d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][1]);
#ifdef ENABLE_PERIODIC_BC
	      std::vector<double>  ZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*2,0.0);
	      std::vector<double> gradZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*C_DIM*2,0.0);
	      std::vector<double> gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*C_DIM*2,0.0);
	      std::vector<double> gradZetalmDeltaVlDyadicDistImageAtoms_KPoint(numkPoints*numberQuadraturePoints*C_DIM*C_DIM*2,0.0);
#else
	      std::vector<double> ZetalmDeltaVl(numberQuadraturePoints,0.0);
	      std::vector<double> gradZetalmDeltaVl(numberQuadraturePoints*C_DIM,0.0);
#endif

	      double nlpValue = 0.0;
	      for(unsigned int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
		{

		  Point<3> quadPoint=fe_values.quadrature_point(iQuadPoint);

		  for(unsigned int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
		    {

		      int chargeId = imageIdsList[iImageAtomCount];

		      Point<3> chargePoint(0.0,0.0,0.0);

		      if(chargeId < numberGlobalCharges)
			{
			  chargePoint[0] = dftPtr->atomLocations[chargeId][2];
			  chargePoint[1] = dftPtr->atomLocations[chargeId][3];
			  chargePoint[2] = dftPtr->atomLocations[chargeId][4];
			}
		      else
			{
			  chargePoint[0] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][0];
			  chargePoint[1] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][1];
			  chargePoint[2] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][2];
			}

		      double x[3],qMinusLr[3];

		      x[0] = quadPoint[0] - chargePoint[0];
		      x[1] = quadPoint[1] - chargePoint[1];
		      x[2] = quadPoint[2] - chargePoint[2];
		      qMinusLr[0] =x[0] +nuclearCoordinates[0];
		      qMinusLr[1] =x[1] +nuclearCoordinates[1];
		      qMinusLr[2] =x[2] +nuclearCoordinates[2];

		      //
		      // get the spherical coordinates from cartesian
		      //
		      double r,theta,phi;
		      pseudoForceUtils::convertCartesianToSpherical(x,r,theta,phi);

		      double radialWaveFunVal, sphericalHarmonicVal, radialPotFunVal, pseudoWaveFunctionValue, deltaVlValue;
	              std::vector<double> pseudoWaveFunctionDerivatives(3,0.0);
		      std::vector<double> deltaVlDerivatives(3,0.0);
		      if(r <= dftPtr->d_pspTail)//d_outerMostPointPseudoWaveFunctionsData[globalWaveSplineId])
			{
			  pseudoForceUtils::getRadialFunctionVal(r,
					                         radialWaveFunVal,
					                         &dftPtr->d_pseudoWaveFunctionSplines[globalWaveSplineId]);

			  pseudoForceUtils::getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);

			  pseudoWaveFunctionValue = radialWaveFunVal*sphericalHarmonicVal;

			  pseudoForceUtils::getRadialFunctionVal(r,
					                         radialPotFunVal,
					                         &dftPtr->d_deltaVlSplines[globalPotSplineId]);

			  deltaVlValue = radialPotFunVal;

			  pseudoForceUtils::getPseudoWaveFunctionDerivatives(r,
							                     theta,
							                     phi,
							                     lQuantumNumber,
							                     mQuantumNumber,
							                     pseudoWaveFunctionDerivatives,
							                     dftPtr->d_pseudoWaveFunctionSplines[globalWaveSplineId]);

			  pseudoForceUtils::getDeltaVlDerivatives(r,
						                  x,
						                  deltaVlDerivatives,
						                  dftPtr->d_deltaVlSplines[globalPotSplineId]);
			  std::vector<double> tempDer(3);
			  for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
			  {
				tempDer[iDim]=pseudoWaveFunctionDerivatives[iDim]*radialPotFunVal + pseudoWaveFunctionValue*deltaVlDerivatives[iDim];
			  }
#ifdef ENABLE_PERIODIC_BC
                          for (unsigned int ik=0; ik < numkPoints; ++ik)
			  {

			     const double kDotqMinusLr= dftPtr->d_kPointCoordinates[ik*C_DIM+0]*qMinusLr[0]+ dftPtr->d_kPointCoordinates[ik*C_DIM+1]*qMinusLr[1]+dftPtr->d_kPointCoordinates[ik*C_DIM+2]*qMinusLr[2];
			     const double tempReal=std::cos(-kDotqMinusLr);
			     const double tempImag=std::sin(-kDotqMinusLr);
		             ZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*2+2*iQuadPoint+0] += tempReal*deltaVlValue*pseudoWaveFunctionValue;
		             ZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*2+2*iQuadPoint+1] += tempImag*deltaVlValue*pseudoWaveFunctionValue;
			     for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
			     {
			         gradZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempReal*tempDer[iDim];
				 gradZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]+= tempImag*tempDer[iDim];
			         gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempReal*tempDer[iDim];
			         gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]+= tempImag*tempDer[iDim];
			         gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempImag*deltaVlValue*pseudoWaveFunctionValue*dftPtr->d_kPointCoordinates[ik*C_DIM+iDim];
			         gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]-= tempReal*deltaVlValue*pseudoWaveFunctionValue*dftPtr->d_kPointCoordinates[ik*C_DIM+iDim];
				 for(unsigned int jDim=0; jDim < C_DIM; ++jDim)
				 {
				     gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[ik*numberQuadraturePoints*C_DIM*C_DIM*2+iQuadPoint*C_DIM*C_DIM*2+iDim*C_DIM*2+jDim*2+0]+= tempReal*tempDer[iDim]*x[jDim];
				     gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[ik*numberQuadraturePoints*C_DIM*C_DIM*2+iQuadPoint*C_DIM*C_DIM*2+iDim*C_DIM*2+jDim*2+1]+= tempImag*tempDer[iDim]*x[jDim];
				 }
			     }
			  }
#else

		          ZetalmDeltaVl[iQuadPoint] += deltaVlValue*pseudoWaveFunctionValue;

			  for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
			      gradZetalmDeltaVl[iQuadPoint*C_DIM+iDim]+= tempDer[iDim];

#endif
			}// within psp tail check

		    }//image atom loop (contribution added)

		}//end of quad loop
#ifdef ENABLE_PERIODIC_BC
	        d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl_KPoint;
		d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl_KPoint;
		d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint;
		d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVlDyadicDistImageAtoms_KPoint;
#else
	        d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl;
		d_nonLocalPSP_gradZetalmDeltaVl[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl;
#endif

	    }//end of iPseudoWave loop


	}//element loop

      cumulativePotSplineId += numberAngularMomentumSpecificPotentials;
      cumulativeWaveSplineId += numberPseudoWaveFunctions;
      if (numberElementsInAtomCompactSupport !=0)
         count++;

    }//atom loop

}


template<unsigned int FEOrder>
void forceClass<FEOrder>::computeNonLocalProjectorKetTimesPsiTimesV(const std::vector<vectorType> &src,
							            std::vector<std::vector<double> > & projectorKetTimesPsiTimesVReal,
                                                                    std::vector<std::vector<std::complex<double> > > & projectorKetTimesPsiTimesVComplex,
								    const unsigned int kPointIndex)
{
  //
  //get FE data
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());

  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

  int numberNodesPerElement;
#ifdef ENABLE_PERIODIC_BC
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  //compute nonlocal projector ket times x i.e C^{T}*X
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::complex<double> > > & projectorKetTimesVector=projectorKetTimesPsiTimesVComplex;
#else
  std::vector<std::vector<double> > & projectorKetTimesVector=projectorKetTimesPsiTimesVReal;
#endif

  int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();
  projectorKetTimesVector.resize(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());

  //
  //allocate memory for matrix-vector product
  //
  std::map<unsigned int, unsigned int> globalToLocalMap;
  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      globalToLocalMap[atomId]=iAtom;
      int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  //
  //some useful vectors
  //
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif


  //
  //parallel loop over all elements to compute nonlocal projector ket times x i.e C^{T}*X
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen.begin_active(), endc = dftPtr->dofHandlerEigen.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem += 1;
	  cell->get_dof_indices(local_dof_indices);

	  unsigned int index=0;
#ifdef ENABLE_PERIODIC_BC
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for (std::vector<vectorType>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //
		  //This is the component index 0(real) or 1(imag).
		  //
		  const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
		  const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;
		  if(ck == 0)
		    inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
		  else
		    inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
		}
	      index++;
	    }


#else
	  for (std::vector<vectorType>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), inputVectors.begin()+numberNodesPerElement*index);
	      index++;
	    }
#endif

	  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef ENABLE_PERIODIC_BC
	      char transA = 'C';
	      char transB = 'N';
	      std::complex<double> alpha = 1.0;
	      std::complex<double> beta = 1.0;
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#else
	      char transA = 'T';
	      char transB = 'N';
	      double alpha = 1.0;
	      double beta = 1.0;
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#endif
	    }

	}

    }//element loop

  //std::cout<<"Finished Element Loop"<<std::endl;
#ifdef ENABLE_PERIODIC_BC
  std::vector<dealii::parallel::distributed::Vector<std::complex<double> > > projectorKetTimesVectorPar(numberWaveFunctions);
#else
  std::vector<dealii::parallel::distributed::Vector<double> > projectorKetTimesVectorPar(numberWaveFunctions);
#endif
#ifdef ENABLE_PERIODIC_BC
  dealii::parallel::distributed::Vector<std::complex<double> > vec(dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
                                                                   dftPtr->d_ghostProjectorIdsCurrentProcess,
                                                                   mpi_communicator);
#else
  dealii::parallel::distributed::Vector<double > vec(dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
                                                     dftPtr->d_ghostProjectorIdsCurrentProcess,
                                                     mpi_communicator);
#endif
  vec.update_ghost_values();
  for (unsigned int i=0; i<numberWaveFunctions;++i)
  {
#ifdef ENABLE_PERIODIC_BC
      projectorKetTimesVectorPar[i].reinit(vec);
#else
      projectorKetTimesVectorPar[i].reinit(vec);
#endif
  }

  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	       projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]]
		  =projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave];
	    }
	}
    }

   for (unsigned int i=0; i<numberWaveFunctions;++i)
   {
      projectorKetTimesVectorPar[i].compress(VectorOperation::add);
      projectorKetTimesVectorPar[i].update_ghost_values();
   }

  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	      projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]
	           =projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]];

	    }
	}
    }


  //
  //compute V*C^{T}*X
  //
  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
	}
    }


}
