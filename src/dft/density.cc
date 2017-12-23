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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Krishnendu Ghosh(2017)
//

//source file for electron density related computations

//calculate electron density
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_rhoOut()
{
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FEEigen, quadrature, update_values | update_gradients| update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
   
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for(int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < numEigenValues; ++i)
	{
	  *eigenVectorsOrig[kPoint][i]=*eigenVectors[kPoint][i];
	  (*eigenVectorsOrig[kPoint][i]).scale(eigenPtr->massVector);
	  constraintsNoneEigen.distribute(*eigenVectorsOrig[kPoint][i]);
	  eigenVectorsOrig[kPoint][i]->update_ghost_values();
	}
    }

  //pcout<<"check 6: "<<std::endl;
  
  //create new rhoValue tables
  rhoOutValues = new std::map<dealii::CellId,std::vector<double> >;
  rhoOutVals.push_back(rhoOutValues);
  if (spinPolarized==1)
    {
    rhoOutValuesSpinPolarized = new std::map<dealii::CellId,std::vector<double> >;
    rhoOutValsSpinPolarized.push_back(rhoOutValuesSpinPolarized);
    }

  //pcout<<"check 6.1: "<<std::endl;
  if(xc_id == 4)
    {
      gradRhoOutValues = new std::map<dealii::CellId, std::vector<double> >;
      gradRhoOutVals.push_back(gradRhoOutValues);
      if (spinPolarized==1)
       {
         gradRhoOutValuesSpinPolarized = new std::map<dealii::CellId, std::vector<double> >;
         gradRhoOutValsSpinPolarized.push_back(gradRhoOutValuesSpinPolarized);
       }
    }
  
  //temp arrays
  std::vector<double> rhoOut(num_quad_points), rhoOutSpinPolarized(2*num_quad_points);
  std::vector<double> gradRhoOut(3*num_quad_points), gradRhoOutSpinPolarized(6*num_quad_points);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandlerEigen.begin_active(), endc = dofHandlerEigen.end();
  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{

	  fe_values.reinit (cell); 

	  (*rhoOutValues)[cell->id()] = std::vector<double>(num_quad_points);
	  std::fill(rhoOut.begin(),rhoOut.end(),0.0);
	  if (spinPolarized==1)
    	     {
	       	(*rhoOutValuesSpinPolarized)[cell->id()] = std::vector<double>(2*num_quad_points);
		std::fill(rhoOutSpinPolarized.begin(),rhoOutSpinPolarized.end(),0.0);
	     }
	
#ifdef ENABLE_PERIODIC_BC
	  std::vector<Vector<double> > tempPsi(num_quad_points), tempPsi2(num_quad_points);
 	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    {
	      tempPsi[q_point].reinit(2);
	      tempPsi2[q_point].reinit(2);
	    }
#else
	  std::vector<double> tempPsi(num_quad_points), tempPsi2(num_quad_points);
#endif



	  if(xc_id == 4)//GGA
	    {
	      (*gradRhoOutValues)[cell->id()] = std::vector<double>(3*num_quad_points);
	      std::fill(gradRhoOut.begin(),gradRhoOut.end(),0.0);
	      if (spinPolarized==1)
    	        {
	       	   (*gradRhoOutValuesSpinPolarized)[cell->id()] = std::vector<double>(6*num_quad_points);
	            std::fill(gradRhoOutSpinPolarized.begin(),gradRhoOutSpinPolarized.end(),0.0);
	        } 
#ifdef ENABLE_PERIODIC_BC
	      std::vector<std::vector<Tensor<1,3,double> > > tempGradPsi(num_quad_points), tempGradPsi2(num_quad_points);
	      for(unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
	         {
		   tempGradPsi[q_point].resize(2);
		   tempGradPsi2[q_point].resize(2);
		 }
#else
	      std::vector<Tensor<1,3,double> > tempGradPsi(num_quad_points), tempGradPsi2(num_quad_points);
#endif


	      for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
		{
		  for(unsigned int i=0; i<numEigenValues; ++i)
		    {
		      fe_values.get_function_values(*eigenVectorsOrig[(1+spinPolarized)*kPoint][i], tempPsi);
		      if(spinPolarized==1)
		         fe_values.get_function_values(*eigenVectorsOrig[(1+spinPolarized)*kPoint+1][i], tempPsi2);
		      //
		      fe_values.get_function_gradients(*eigenVectorsOrig[(1+spinPolarized)*kPoint][i],tempGradPsi);
		      if(spinPolarized==1)
		         fe_values.get_function_gradients(*eigenVectorsOrig[(1+spinPolarized)*kPoint+1][i], tempGradPsi2);

		      for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  double factor = (eigenValues[kPoint][i]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			  //
			   factor=(eigenValues[kPoint][i+spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			  if(spinPolarized==1)
			    {
			      rhoOutSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      rhoOutSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			      //
			      gradRhoOutSpinPolarized[6*q_point + 0] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			      gradRhoOutSpinPolarized[6*q_point + 1] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			      gradRhoOutSpinPolarized[6*q_point + 2] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			      gradRhoOutSpinPolarized[6*q_point + 3] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][0] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][0]);
			      gradRhoOutSpinPolarized[6*q_point + 4] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][1] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][1]);
			      gradRhoOutSpinPolarized[6*q_point + 5] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][2] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][2]);
			    }
			  else
			    {
			      rhoOut[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      gradRhoOut[3*q_point + 0] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			      gradRhoOut[3*q_point + 1] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			      gradRhoOut[3*q_point + 2] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			    }
#else
			  if(spinPolarized==1)
			    {
			      rhoOutSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			      rhoOutSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			      //rhoOut[q_point] += rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
			      gradRhoOutSpinPolarized[6*q_point + 0] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][0]) ;
			      gradRhoOutSpinPolarized[6*q_point + 1] +=  2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][1]) ;
			      gradRhoOutSpinPolarized[6*q_point + 2] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][2]) ;			      
			      gradRhoOutSpinPolarized[6*q_point + 3] +=  2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][0]);
			      gradRhoOutSpinPolarized[6*q_point + 4] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][1]) ;
			      gradRhoOutSpinPolarized[6*q_point + 5] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][2]) ;
			    }
			  else
			    {
			      rhoOut[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0); 
			      gradRhoOut[3*q_point + 0] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][0];
			      gradRhoOut[3*q_point + 1] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][1];
			      gradRhoOut[3*q_point + 2] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][2];
			    }

#endif
			}
		    }
		}

	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  if(spinPolarized==1)
		      {
			(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point]=rhoOutSpinPolarized[2*q_point] ;
			(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1]=rhoOutSpinPolarized[2*q_point+1] ;
			(*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 0] = gradRhoOutSpinPolarized[6*q_point + 0];
		        (*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 1] = gradRhoOutSpinPolarized[6*q_point + 1];
		        (*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 2] = gradRhoOutSpinPolarized[6*q_point + 2];
			(*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 3] = gradRhoOutSpinPolarized[6*q_point + 3];
		        (*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 4] = gradRhoOutSpinPolarized[6*q_point + 4];
		        (*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 5] = gradRhoOutSpinPolarized[6*q_point + 5];
			//
			(*rhoOutValues)[cell->id()][q_point]= rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
			(*gradRhoOutValues)[cell->id()][3*q_point + 0] = gradRhoOutSpinPolarized[6*q_point + 0] + gradRhoOutSpinPolarized[6*q_point + 3];
		        (*gradRhoOutValues)[cell->id()][3*q_point + 1] = gradRhoOutSpinPolarized[6*q_point + 1] + gradRhoOutSpinPolarized[6*q_point + 4];
		        (*gradRhoOutValues)[cell->id()][3*q_point + 2] = gradRhoOutSpinPolarized[6*q_point + 2] + gradRhoOutSpinPolarized[6*q_point + 5];
                      }
		  else
		      {
			(*rhoOutValues)[cell->id()][q_point]  = rhoOut[q_point];
		        (*gradRhoOutValues)[cell->id()][3*q_point + 0] = gradRhoOut[3*q_point + 0];
		        (*gradRhoOutValues)[cell->id()][3*q_point + 1] = gradRhoOut[3*q_point + 1];
		        (*gradRhoOutValues)[cell->id()][3*q_point + 2] = gradRhoOut[3*q_point + 2];
		      }
		}

	    }
	  else
	    {
	      for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
		{
		  for(unsigned int i=0; i<numEigenValues; ++i)
		    {
		      fe_values.get_function_values(*eigenVectorsOrig[(1+spinPolarized)*kPoint][i], tempPsi);
		      if(spinPolarized==1)
		         fe_values.get_function_values(*eigenVectorsOrig[(1+spinPolarized)*kPoint+1][i], tempPsi2);

		      for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  double factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			  //
			  factor=(eigenValues[kPoint][i+spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			   if(spinPolarized==1)
			    {
			      rhoOutSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      rhoOutSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			      //rhoOut[q_point] += rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
			    }
			  else
			      rhoOut[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
#else
			   if(spinPolarized==1)
			    {
			      rhoOutSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			      rhoOutSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			      //rhoOut[q_point] += rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
			    }
			  else
			      rhoOut[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0); 
			  //
#endif
			}
		    }
		}
              //pcout<<"check 6.2: "<<std::endl;
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  if(spinPolarized==1)
		      {
			(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point]=rhoOutSpinPolarized[2*q_point] ;
			(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1]=rhoOutSpinPolarized[2*q_point+1] ;
			(*rhoOutValues)[cell->id()][q_point]= rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
                      }
		  else
			(*rhoOutValues)[cell->id()][q_point]  = rhoOut[q_point];
		}

	    }

	}

    }
 
  //pcout<<"check 7: "<<std::endl;

  //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size
  if(rhoInVals.size() == mixingHistory)
    {
      rhoInVals.pop_front();
      rhoOutVals.pop_front();
      rhoInValsSpinPolarized.pop_front();
      rhoOutValsSpinPolarized.pop_front();
      gradRhoInVals.pop_front();
      gradRhoOutVals.pop_front();
      gradRhoInValsSpinPolarized.pop_front();
      gradRhoOutValsSpinPolarized.pop_front();
    }

}
