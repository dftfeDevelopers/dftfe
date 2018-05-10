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
// @author Shiva Rudraraju, Phani Motamarri
//


namespace internal {

    double FermiDiracFunctionValue(const double x,
				   const std::vector<std::vector<double> > & eigenValues,
				   const std::vector<double> & kPointWeights,
				   const double &TVal)
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
		  functionValue += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*temp2;
		}
	      else
		{
		  temp2 =  1.0/(1.0+exp(-temp1));
		  functionValue += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*exp(-temp1)*temp2;
		}
	    }
	}

      return functionValue;

    }

    double FermiDiracFunctionDerivativeValue(const double x,
					     const std::vector<std::vector<double> > & eigenValues,
					     const std::vector<double> & kPointWeights,
					     const double &TVal)
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
		  functionDerivative += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*(exp(temp1)/(C_kb*TVal))*temp2*temp2;
		}
	      else
		{
		  temp2 =  1.0/(1.0 + exp(-temp1));
		  functionDerivative += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*(exp(-temp1)/(C_kb*TVal))*temp2*temp2;
		}
	    }
	}

      return functionDerivative;

    }

}

//compute fermi energy
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_fermienergy()
{

  int count =  std::ceil(static_cast<double>(numElectrons)/(2.0-dftParameters::spinPolarized));
  double TVal = dftParameters::TVal;


  std::vector<double> eigenValuesAllkPoints;
  for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
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

#ifdef USE_COMPLEX
  //
  //compute Fermi-energy first by bisection method
  //
  //double initialGuessLeft = Utilities::MPI::min(eigenValuesAllkPoints[0],interpoolcomm);
  //double initialGuessRight = Utilities::MPI::max(eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1],interpoolcomm);

  double initialGuessLeft = eigenValuesAllkPoints[0];
  double initialGuessRight = eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1];


  double xLeft,xRight;

  xRight = Utilities::MPI::max(initialGuessRight, interpoolcomm);
  xLeft =  Utilities::MPI::min(initialGuessLeft, interpoolcomm);


  for(int iter = 0; iter < maxNumberFermiEnergySolveIterations; ++iter)
    {
      double yRightLocal = internal::FermiDiracFunctionValue(xRight,
					      eigenValues,
					      d_kPointWeights,
					      TVal);

      double yRight = Utilities::MPI::sum(yRightLocal, interpoolcomm);

      yRight -=  (double)numElectrons;

      double yLeftLocal =  internal::FermiDiracFunctionValue(xLeft,
					      eigenValues,
					      d_kPointWeights,
					      TVal);

      double yLeft = Utilities::MPI::sum(yLeftLocal, interpoolcomm);

      yLeft -=  (double)numElectrons;

      if((yLeft*yRight) > 0.0)
	{
	  pcout << " Bisection Method Failed " <<std::endl;
	  exit(-1);
	}

      double xBisected = (xLeft + xRight)/2.0;

      double yBisectedLocal = internal::FermiDiracFunctionValue(xBisected,
						 eigenValues,
						 d_kPointWeights,
						 TVal) ;
      double yBisected = Utilities::MPI::sum(yBisectedLocal, interpoolcomm);
      yBisected -=  (double)numElectrons;

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
    if (dftParameters::verbosity==2)
      pcout<< "Fermi energy constraint residual (bisection): "<< R << std::endl;
#else
   fe = eigenValuesAllkPoints[d_kPointWeights.size()*count - 1];
#endif
  //
  //compute residual and find FermiEnergy using Newton-Raphson solve
  //
  //double R = 1.0;
  unsigned int iter = 0;
  double  functionValue, functionDerivativeValue;

  while((std::abs(R) > 1.0e-12) && (iter < maxNumberFermiEnergySolveIterations))
    {

      double functionValueLocal = internal::FermiDiracFunctionValue(fe,
					      eigenValues,
					      d_kPointWeights,
					      TVal);
      functionValue = Utilities::MPI::sum(functionValueLocal, interpoolcomm);

      double functionDerivativeValueLocal  = internal::FermiDiracFunctionDerivativeValue(fe,
								  eigenValues,
								  d_kPointWeights,
								  TVal);

      functionDerivativeValue = Utilities::MPI::sum(functionDerivativeValueLocal, interpoolcomm);


      R   =  functionValue - numElectrons;
      fe += -R/functionDerivativeValue;
      iter++;
    }

  if(std::abs(R) > 1.0e-12)
    {
      pcout << "Fermi Energy computation: Newton iterations failed to converge\n";
      //exit(-1);
    }

  //set Fermi energy
  fermiEnergy = fe;

  if (dftParameters::verbosity==2)
     pcout<< "Fermi energy constraint residual (Newton-Raphson): "<< std::abs(R)<<std::endl;

  if (dftParameters::verbosity==2)
     pcout<< "Fermi energy                                     : "<< fermiEnergy<<std::endl;
}
