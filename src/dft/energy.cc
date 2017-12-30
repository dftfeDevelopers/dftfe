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


//source file for all energy computations 
double FermiDiracFunctionValue(double x,
			       std::vector<std::vector<double> > & eigenValues,
			       std::vector<double> & kPointWeights,
			       double &TVal)
{

  int numberkPoints = eigenValues.size();
  int numberEigenValues = eigenValues[0].size();
  double functionValue = 0.0;
  double temp1,temp2;

  
  for(unsigned int kPoint = 0; kPoint < numberkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < numberEigenValues; i++)
	{
	  temp1 = (eigenValues[kPoint][i]-x)/(C_kb*TVal);
	  if(temp1 <= 0.0)
	    {
	      temp2  =  1.0/(1.0+exp(temp1));
	      functionValue += 2.0*kPointWeights[kPoint]*temp2;
	    }
	  else
	    {
	      temp2 =  1.0/(1.0+exp(-temp1));
	      functionValue += 2.0*kPointWeights[kPoint]*exp(-temp1)*temp2;
	    }
	}
    }

  return functionValue;

}

double FermiDiracFunctionDerivativeValue(double x,
					 std::vector<std::vector<double> > & eigenValues,
					 std::vector<double> & kPointWeights,
					 double &TVal)
{

  int numberkPoints = eigenValues.size();
  int numberEigenValues = eigenValues[0].size();
  double functionDerivative = 0.0;
  double temp1,temp2;

  for(unsigned int kPoint = 0; kPoint < numberkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < numberEigenValues; i++)
	{
	  temp1 = (eigenValues[kPoint][i]-x)/(C_kb*TVal);
	  if(temp1 <= 0.0)
	    {
	      temp2  =  1.0/(1.0 + exp(temp1));
	      functionDerivative += 2.0*kPointWeights[kPoint]*(exp(temp1)/(C_kb*TVal))*temp2*temp2;
	    }
	  else
	    {
	      temp2 =  1.0/(1.0 + exp(-temp1));
	      functionDerivative += 2.0*kPointWeights[kPoint]*(exp(-temp1)/(C_kb*TVal))*temp2*temp2; 
	    }
	}
    }

  return functionDerivative;

}


//compute energies
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_energy()
{
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   num_quad_points    = quadrature.size();
  std::vector<double> cellPhiTotRhoIn(num_quad_points);  
  std::vector<double> cellPhiTotRhoOut(num_quad_points);  
  std::vector<double> cellPhiExt(num_quad_points);
  double TVal = dftParameters::TVal;



  //
  // Loop through all cells.
  //
  double bandEnergy=0.0;
  double partialOccupancy, factor;
  char buffer[100];
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      pcout << "kPoint: "<< kPoint <<std::endl;
      for (unsigned int i=0; i<numEigenValues; i++)
	{
	  factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*TVal);
	  //partialOccupancy=1.0/(1.0+exp(temp));
	  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
	  bandEnergy+= 2*partialOccupancy*d_kPointWeights[kPoint]*eigenValues[kPoint][i];
	  sprintf(buffer, "%s %u: %0.14f\n", "fractional occupancy", i, partialOccupancy); pcout << buffer;
	}
      pcout << std::endl; 
      sprintf(buffer, "number of electrons: %18.16e \n", integralRhoValue); pcout << buffer;
    }
  pcout <<std::endl;
  double potentialTimesRho = 0.0, exchangeEnergy = 0.0, correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0; 

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  if(dftParameters::xc_id == 4)
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get exc
	      std::vector<double> densityValueIn(num_quad_points), densityValueOut(num_quad_points);
	      std::vector<double> exchangeEnergyDensity(num_quad_points), corrEnergyDensity(num_quad_points);
	      std::vector<double> derExchEnergyWithInputDensity(num_quad_points), derCorrEnergyWithInputDensity(num_quad_points);
	      std::vector<double> derExchEnergyWithSigmaGradDenInput(num_quad_points),derCorrEnergyWithSigmaGradDenInput(num_quad_points);
	      std::vector<double> sigmaWithOutputGradDensity(num_quad_points), sigmaWithInputGradDensity(num_quad_points);
	      std::vector<double> gradRhoInDotgradRhoOut(num_quad_points);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  densityValueIn[q_point] = (*rhoInValues)[cell->id()][q_point];
		  densityValueOut[q_point] = (*rhoOutValues)[cell->id()][q_point];
		  double gradRhoInX = ((*gradRhoInValues)[cell->id()][3*q_point + 0]);
		  double gradRhoInY = ((*gradRhoInValues)[cell->id()][3*q_point + 1]);
		  double gradRhoInZ = ((*gradRhoInValues)[cell->id()][3*q_point + 2]);
		  double gradRhoOutX = ((*gradRhoOutValues)[cell->id()][3*q_point + 0]);
		  double gradRhoOutY = ((*gradRhoOutValues)[cell->id()][3*q_point + 1]);
		  double gradRhoOutZ = ((*gradRhoOutValues)[cell->id()][3*q_point + 2]);
		  sigmaWithInputGradDensity[q_point] = gradRhoInX*gradRhoInX + gradRhoInY*gradRhoInY + gradRhoInZ*gradRhoInZ;
		  sigmaWithOutputGradDensity[q_point] = gradRhoOutX*gradRhoOutX + gradRhoOutY*gradRhoOutY + gradRhoOutZ*gradRhoOutZ;
		  gradRhoInDotgradRhoOut[q_point] = gradRhoInX*gradRhoOutX + gradRhoInY*gradRhoOutY + gradRhoInZ*gradRhoOutZ;
		}
	      xc_gga_exc(&funcX,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&exchangeEnergyDensity[0]);
	      xc_gga_exc(&funcC,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&corrEnergyDensity[0]);

	      xc_gga_vxc(&funcX,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derExchEnergyWithInputDensity[0],&derExchEnergyWithSigmaGradDenInput[0]);
	      xc_gga_vxc(&funcC,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derCorrEnergyWithInputDensity[0],&derCorrEnergyWithSigmaGradDenInput[0]);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+derExchEnergyWithInputDensity[q_point]+derCorrEnergyWithInputDensity[q_point];
		  double VxcGrad = 2.0*(derExchEnergyWithSigmaGradDenInput[q_point]+derCorrEnergyWithSigmaGradDenInput[q_point])*gradRhoInDotgradRhoOut[q_point];

		  // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];

		  // quad rule
		  potentialTimesRho+=(Veff*((*rhoOutValues)[cell->id()][q_point])+VxcGrad)*fe_values.JxW (q_point);
		  exchangeEnergy+=(exchangeEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  correlationEnergy+=(corrEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  electrostaticEnergyTotPot+=0.5*(Vtot+Vext*0)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }
  else
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get Exc
	      std::vector<double> densityValueIn(num_quad_points), densityValueOut(num_quad_points);
	      std::vector<double> exchangeEnergyVal(num_quad_points), corrEnergyVal(num_quad_points);
	      std::vector<double> exchangePotentialVal(num_quad_points), corrPotentialVal(num_quad_points);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  densityValueIn[q_point] = (*rhoInValues)[cell->id()][q_point];
		  densityValueOut[q_point] = (*rhoOutValues)[cell->id()][q_point];
		}
	      xc_lda_exc(&funcX,num_quad_points,&densityValueOut[0],&exchangeEnergyVal[0]);
	      xc_lda_exc(&funcC,num_quad_points,&densityValueOut[0],&corrEnergyVal[0]);
	      xc_lda_vxc(&funcX,num_quad_points,&densityValueIn[0],&exchangePotentialVal[0]);
	      xc_lda_vxc(&funcC,num_quad_points,&densityValueIn[0],&corrPotentialVal[0]);
	      for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		{
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point];
		  // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];
		  potentialTimesRho+=Veff*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW (q_point);
		  exchangeEnergy+=(exchangeEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  correlationEnergy+=(corrEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  //electrostaticEnergyTotPot+=0.5*(Vtot+Vext)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }


  
  double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergyTotPot;

  //
  //get nuclear electrostatic energy 0.5*sum_I*(phi_tot(RI) - VselfI(RI))
  //
  
  //
  //First evaluate sum_I*(Z_I*phi_tot(RI)) on atoms belonging to current processor
  //
  double phiContribution = 0.0,vSelfContribution=0.0;
  for (std::map<unsigned int, double>::iterator it=atoms.begin(); it!=atoms.end(); ++it)
    {
      phiContribution += (-it->second)*poissonPtr->phiTotRhoOut(it->first);//-charge*potential
    }

  //
  //Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to current processor
  //
  for(int i = 0; i < d_localVselfs.size(); ++i)
    {
      vSelfContribution += (-d_localVselfs[i][0])*(d_localVselfs[i][1]);//-charge*potential
    }

  double nuclearElectrostaticEnergy = 0.5*(phiContribution - vSelfContribution);



  //sum over all processors
  double totalEnergy= Utilities::MPI::sum(energy, mpi_communicator);
  double totalpotentialTimesRho= Utilities::MPI::sum(potentialTimesRho, mpi_communicator); 
  double totalexchangeEnergy= Utilities::MPI::sum(exchangeEnergy, mpi_communicator); 
  double totalcorrelationEnergy= Utilities::MPI::sum(correlationEnergy, mpi_communicator);
  double totalelectrostaticEnergyPot= Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
  double totalNuclearElectrostaticEnergy = Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);


  //
  //total energy
  //
  totalEnergy+=bandEnergy;
 
 
#ifdef ENABLE_PERIODIC_BC
  totalEnergy+=totalNuclearElectrostaticEnergy;
#else
  totalEnergy+=totalNuclearElectrostaticEnergy;//repulsiveEnergy();
#endif


  double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;

  //output
  char bufferEnergy[200];
  pcout << "Energy computations\n";
  pcout << "-------------------\n";
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Band energy", bandEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Kinetic energy", totalkineticEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Exchange energy", totalexchangeEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Correlation energy", totalcorrelationEnergy); pcout << bufferEnergy; 
#ifdef ENABLE_PERIODIC_BC  
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy); pcout << bufferEnergy; 
#else
  //double repulsive_energy= repulsiveEnergy();
  //sprintf(bufferEnergy, "%-24s:%25.16e\n", "Repulsive energy", repulsive_energy); pcout << bufferEnergy; 
  //sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+repulsive_energy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy); pcout << bufferEnergy;   
#endif
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy", totalEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy per atom", totalEnergy/((double) atomLocations.size())); pcout << bufferEnergy; 
  /* 
  if (this_mpi_process == 0) {
    std::printf("Total energy:%30.20e \nTotal energy per atom:%30.20e \n", totalEnergy, totalEnergy/((double) atomLocations.size()));
    std::printf("Band energy:%30.20e \nKinetic energy:%30.20e \nExchange energy:%30.20e \nCorrelation energy:%30.20e \nElectrostatic energy Total Potential:%30.20e \nRepulsive energy:%30.20e \nNuclear Electrostatic Energy:%30.20e \n\n", bandEnergy, totalkineticEnergy, totalexchangeEnergy, totalcorrelationEnergy, totalelectrostaticEnergyPot, repulsiveEnergy(),totalNuclearElectrostaticEnergy);
  }
  */
}
 
//compute fermi energy
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_fermienergy()
{
  char bufferFermi[100];
  int count =  std::ceil(static_cast<double>(numElectrons)/2.0);
  double TVal = dftParameters::TVal;





  std::vector<double> eigenValuesAllkPoints;
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for(int statesIter = 0; statesIter < eigenValues[0].size(); ++statesIter)
	{
	  eigenValuesAllkPoints.push_back(eigenValues[kPoint][statesIter]);
	}
    }

  std::sort(eigenValuesAllkPoints.begin(),eigenValuesAllkPoints.end());

  unsigned int maxNumberFermiEnergySolveIterations = 100;
  double fe;
  double R = 1.0;

#ifdef ENABLE_PERIODIC_BC
  //
  //compute Fermi-energy first by bisection method
  //  
  double initialGuessLeft = eigenValuesAllkPoints[0];
  double initialGuessRight = eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1];

  double xLeft,xRight;

  xRight = initialGuessRight;
  xLeft = initialGuessLeft;
  

  for(int iter = 0; iter < maxNumberFermiEnergySolveIterations; ++iter)
    {
      double yRight = FermiDiracFunctionValue(xRight,
					      eigenValues,
					      d_kPointWeights,
					      TVal) - numElectrons;

      double yLeft =  FermiDiracFunctionValue(xLeft,
					      eigenValues,
					      d_kPointWeights,
					      TVal) - numElectrons;

      if((yLeft*yRight) > 0.0)
	{
	  pcout << " Bisection Method Failed " <<std::endl;
	  exit(-1);
	}
      
      double xBisected = (xLeft + xRight)/2.0;

      double yBisected = FermiDiracFunctionValue(xBisected,
						 eigenValues,
						 d_kPointWeights,
						 TVal) - numElectrons;

      if((yBisected*yLeft) > 0.0)
	xLeft = xBisected;
      else
	xRight = xBisected;
					      
      if (std::abs(yBisected) <= 1.0e-09 || iter == maxNumberFermiEnergySolveIterations-1)
	{
	  fe = xBisected;
	  R  = std::abs(yBisected);
	  break;
	}

    }
  sprintf(bufferFermi, "%-50s: %25.16e\n", "Fermi energy constraint residual (bisection)", R); pcout << bufferFermi; 
#else
   fe = eigenValuesAllkPoints[d_maxkPoints*count - 1];
#endif  
  //
  //compute residual and find FermiEnergy using Newton-Raphson solve
  //
  //double R = 1.0;
  unsigned int iter = 0;
  double  functionValue, functionDerivativeValue;

  while((std::abs(R) > 1.0e-12) && (iter < maxNumberFermiEnergySolveIterations))
    {

      functionValue = FermiDiracFunctionValue(fe,
					      eigenValues,
					      d_kPointWeights,
					      TVal);

      functionDerivativeValue = FermiDiracFunctionDerivativeValue(fe,
								  eigenValues,
								  d_kPointWeights,
								  TVal);

      R   =  functionValue - numElectrons;
      fe += -R/functionDerivativeValue; 
      iter++;
    }

  if(std::abs(R) > 1.0e-12)
    {
      pcout << "Fermi Energy computation: Newton iterations failed to converge\n";
      //exit(-1);
    }

  sprintf(bufferFermi, "%-50s: %25.16e\n", "Fermi energy constraint residual (Newton-Raphson)", std::abs(R)); pcout << bufferFermi; 

  //set Fermi energy
  fermiEnergy = fe;
  sprintf(bufferFermi, "%-50s: %25.16e\n\n", "Fermi energy", fermiEnergy); pcout << bufferFermi; 
}

template<unsigned int FEOrder>
double dftClass<FEOrder>::repulsiveEnergy()
{
  double energy=0.0;
  for (unsigned int n1=0; n1<atomLocations.size(); n1++){
    for (unsigned int n2=n1+1; n2<atomLocations.size(); n2++){
      double Z1,Z2;
      if(dftParameters::isPseudopotential)
	{
	  Z1=atomLocations[n1][1];
	  Z2=atomLocations[n2][1];
	}
      else
	{
	  Z1=atomLocations[n1][0];
	  Z2=atomLocations[n2][0];
	}
      Point<3> atom1(atomLocations[n1][2],atomLocations[n1][3],atomLocations[n1][4]);
      Point<3> atom2(atomLocations[n2][2],atomLocations[n2][3],atomLocations[n2][4]);
      energy+=(Z1*Z2)/atom1.distance(atom2);
    }
  }
  return energy;
}
