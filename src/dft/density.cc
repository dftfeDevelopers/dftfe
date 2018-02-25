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
  unsigned int xc_id = dftParameters::xc_id;
   
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for(int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < numEigenValues; ++i)
	{
	  *eigenVectorsOrig[kPoint][i]=*eigenVectors[kPoint][i];
	  (*eigenVectorsOrig[kPoint][i]).scale(eigenPtr->massVector);
	  eigenVectorsOrig[kPoint][i]->update_ghost_values();
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
  std::vector<double> rhoTemp(num_quad_points), rhoTempSpinPolarized(2*num_quad_points), rhoOut(num_quad_points), rhoOutSpinPolarized(2*num_quad_points);
  std::vector<double> gradRhoTemp(3*num_quad_points), gradRhoTempSpinPolarized(6*num_quad_points),gradRhoOut(3*num_quad_points), gradRhoOutSpinPolarized(6*num_quad_points);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandlerEigen.begin_active(), endc = dofHandlerEigen.end();
  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{

	  fe_values.reinit (cell); 

	  (*rhoOutValues)[cell->id()] = std::vector<double>(num_quad_points);
	  std::fill(rhoTemp.begin(),rhoTemp.end(),0.0); std::fill(rhoOut.begin(),rhoOut.end(),0.0);
	  if (spinPolarized==1)
    	     {
	       	(*rhoOutValuesSpinPolarized)[cell->id()] = std::vector<double>(2*num_quad_points);
		std::fill(rhoTempSpinPolarized.begin(),rhoTempSpinPolarized.end(),0.0);
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
	      std::fill(gradRhoTemp.begin(),gradRhoTemp.end(),0.0);
	      if (spinPolarized==1)
    	        {
	       	   (*gradRhoOutValuesSpinPolarized)[cell->id()] = std::vector<double>(6*num_quad_points);
	            std::fill(gradRhoTempSpinPolarized.begin(),gradRhoTempSpinPolarized.end(),0.0);
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
			  double factor = (eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			  //
			   factor=(eigenValues[kPoint][i+spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			  if(spinPolarized==1)
			    {
			      rhoTempSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			      //
			      gradRhoTempSpinPolarized[6*q_point + 0] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			      gradRhoTempSpinPolarized[6*q_point + 1] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			      gradRhoTempSpinPolarized[6*q_point + 2] += 
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			      gradRhoTempSpinPolarized[6*q_point + 3] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][0] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][0]);
			      gradRhoTempSpinPolarized[6*q_point + 4] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][1] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][1]);
			      gradRhoTempSpinPolarized[6*q_point + 5] += 
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][2] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][2]);
			    }
			  else
			    {
			      rhoTemp[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      gradRhoTemp[3*q_point + 0] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			      gradRhoTemp[3*q_point + 1] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			      gradRhoTemp[3*q_point + 2] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			    }
#else
			  if(spinPolarized==1)
			    {
			      rhoTempSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			      rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			      gradRhoTempSpinPolarized[6*q_point + 0] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][0]) ;
			      gradRhoTempSpinPolarized[6*q_point + 1] +=  2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][1]) ;
			      gradRhoTempSpinPolarized[6*q_point + 2] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][2]) ;			      
			      gradRhoTempSpinPolarized[6*q_point + 3] +=  2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][0]);
			      gradRhoTempSpinPolarized[6*q_point + 4] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][1]) ;
			      gradRhoTempSpinPolarized[6*q_point + 5] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][2]) ;
			    }
			  else
			    {
			      rhoTemp[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0); 
			      gradRhoTemp[3*q_point + 0] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][0];
			      gradRhoTemp[3*q_point + 1] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][1];
			      gradRhoTemp[3*q_point + 2] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][2];
			    }

#endif
			}
		    }
		}

              //  gather density from all pools
	      int numPoint = num_quad_points ;
              MPI_Allreduce(&rhoTemp[0], &rhoOut[0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	      MPI_Allreduce(&gradRhoTemp[0], &gradRhoOut[0], 3*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
              if (spinPolarized==1) {
                 MPI_Allreduce(&rhoTempSpinPolarized[0], &rhoOutSpinPolarized[0], 2*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	         MPI_Allreduce(&gradRhoTempSpinPolarized[0], &gradRhoOutSpinPolarized[0], 6*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ; 
              }

       //


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
			  double factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			  //
			  factor=(eigenValues[kPoint][i+spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*TVal);
			  double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			   if(spinPolarized==1)
			    {
			      rhoTempSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			      rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			      //rhoOut[q_point] += rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
			    }
			  else
			      rhoTemp[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
#else
			   if(spinPolarized==1)
			    {
			      rhoTempSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			      rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			    }
			  else
			      rhoTemp[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0); 
			  //
#endif
			}
		    }
		}
              //  gather density from all pools
	      int numPoint = num_quad_points ;
              MPI_Allreduce(&rhoTemp[0], &rhoOut[0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
              if (spinPolarized==1) 
                 MPI_Allreduce(&rhoTempSpinPolarized[0], &rhoOutSpinPolarized[0], 2*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	      //   
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
  if(rhoInVals.size() == dftParameters::mixingHistory)
    {
      (**(rhoInVals.begin())).clear();
      delete *(rhoInVals.begin());	
      rhoInVals.pop_front();

      (**(rhoOutVals.begin())).clear();
      delete *(rhoOutVals.begin());	      
      rhoOutVals.pop_front();

      if(spinPolarized==1)
      {
	  (**(rhoInValsSpinPolarized.begin())).clear();
	  delete *(rhoInValsSpinPolarized.begin());
	  rhoInValsSpinPolarized.pop_front();

	  (**(rhoOutValsSpinPolarized.begin())).clear();
	  delete *(rhoOutValsSpinPolarized.begin());
	  rhoOutValsSpinPolarized.pop_front();
      }
	  
      if(xc_id == 4)//GGA
      {
	  (**(gradRhoInVals.begin())).clear();
	  delete *(gradRhoInVals.begin());	      
	  gradRhoInVals.pop_front();

	  (**(gradRhoOutVals.begin())).clear();
	  delete *(gradRhoOutVals.begin());	      
	  gradRhoOutVals.pop_front();
      }

      if(spinPolarized==1 && xc_id==4)
      {      
	  (**(gradRhoInValsSpinPolarized.begin())).clear();
	  delete *(gradRhoInValsSpinPolarized.begin());	 
	  gradRhoInValsSpinPolarized.pop_front();

	  (**(gradRhoOutValsSpinPolarized.begin())).clear();
	  delete *(gradRhoOutValsSpinPolarized.begin());	   
	  gradRhoOutValsSpinPolarized.pop_front();
      }
    }

}

//rho data reinitilization without remeshing. The rho out of last ground state solve is made the rho in of the new solve
template<unsigned int FEOrder>
void dftClass<FEOrder>::noRemeshRhoDataInit()
{

  std::map<dealii::CellId, std::vector<double> > *rhoOutValuesCopy=new std::map<dealii::CellId, std::vector<double> >;
  *rhoOutValuesCopy=*(rhoOutValues);
  std::map<dealii::CellId, std::vector<double> > *gradRhoOutValuesCopy;
  if (dftParameters::xc_id==4)
  {
     gradRhoOutValuesCopy = new std::map<dealii::CellId, std::vector<double> >;
    *gradRhoOutValuesCopy=*(gradRhoOutValues);
  }

  std::map<dealii::CellId, std::vector<double> > *rhoOutValuesSpinPolarizedCopy;  
  if(spinPolarized==1)
  {
     rhoOutValuesSpinPolarizedCopy = new std::map<dealii::CellId, std::vector<double> >;
    *rhoOutValuesSpinPolarizedCopy=*(rhoOutValuesSpinPolarized);

  } 

  std::map<dealii::CellId, std::vector<double> > *gradRhoOutValuesSpinPolarizedCopy;  
  if(spinPolarized==1 && dftParameters::xc_id==4)
  {
     gradRhoOutValuesSpinPolarizedCopy = new std::map<dealii::CellId, std::vector<double> >;
    *gradRhoOutValuesSpinPolarizedCopy=*(gradRhoOutValuesSpinPolarized);

  }    
  //cleanup of existing data
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = rhoInVals.begin(); it!=rhoInVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  rhoInVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = rhoOutVals.begin(); it!=rhoOutVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  rhoOutVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = gradRhoInVals.begin(); it!=gradRhoInVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  gradRhoInVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = gradRhoOutVals.begin(); it!=gradRhoOutVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  gradRhoOutVals.clear();

  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = rhoInValsSpinPolarized.begin(); it!=rhoInValsSpinPolarized.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  rhoInValsSpinPolarized.clear();

  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = rhoOutValsSpinPolarized.begin(); it!=rhoOutValsSpinPolarized.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  rhoOutValsSpinPolarized.clear();

  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = gradRhoInValsSpinPolarized.begin(); it!=gradRhoInValsSpinPolarized.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  gradRhoInValsSpinPolarized.clear();

  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = gradRhoOutValsSpinPolarized.begin(); it!=gradRhoOutValsSpinPolarized.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  gradRhoOutValsSpinPolarized.clear();  
  ///

  rhoInValues=new std::map<dealii::CellId, std::vector<double> >;
  *(rhoInValues)=*rhoOutValuesCopy;
  rhoOutValuesCopy->clear();  delete rhoOutValuesCopy;
  rhoInVals.push_back(rhoInValues);

  if (dftParameters::xc_id==4)
  {
    gradRhoInValues = new std::map<dealii::CellId, std::vector<double> >;
    *(gradRhoInValues)=*gradRhoOutValuesCopy;
    gradRhoOutValuesCopy->clear();  delete gradRhoOutValuesCopy;
    gradRhoInVals.push_back(gradRhoInValues);
  }

  if(spinPolarized==1)
  {
    rhoInValuesSpinPolarized=new std::map<dealii::CellId, std::vector<double> >;      
    *(rhoInValuesSpinPolarized) =  *(rhoOutValuesSpinPolarizedCopy);
    rhoOutValuesSpinPolarizedCopy->clear();  delete rhoOutValuesSpinPolarizedCopy;
    rhoInValsSpinPolarized.push_back(rhoInValuesSpinPolarized);
  } 

  if (dftParameters::xc_id==4 && spinPolarized==1)
  {
    gradRhoInValuesSpinPolarized = new std::map<dealii::CellId, std::vector<double> >;
    *(gradRhoInValuesSpinPolarized)=*gradRhoOutValuesSpinPolarizedCopy;
    gradRhoOutValuesSpinPolarizedCopy->clear();  delete gradRhoOutValuesSpinPolarizedCopy;
    gradRhoInValsSpinPolarized.push_back(gradRhoInValuesSpinPolarized);
  }

}

