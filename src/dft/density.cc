//source file for electron density related computations

//calculate electron density
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_rhoOut()
{
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FEEigen, quadrature, update_values | update_gradients| update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
   
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < numEigenValues; ++i)
	{
	  *eigenVectorsOrig[kPoint][i]=*eigenVectors[kPoint][i];
	  (*eigenVectorsOrig[kPoint][i]).scale(eigen.massVector);
	  constraintsNoneEigen.distribute(*eigenVectorsOrig[kPoint][i]);
	  eigenVectorsOrig[kPoint][i]->update_ghost_values();
	}
    }
  
  //create new rhoValue tables
  rhoOutValues = new std::map<dealii::CellId,std::vector<double> >;
  rhoOutVals.push_back(rhoOutValues);


  if(xc_id == 4)
    {
      gradRhoOutValues = new std::map<dealii::CellId, std::vector<double> >;
      gradRhoOutVals.push_back(gradRhoOutValues);
    }
  
  //temp arrays
  std::vector<double> rhoOut(num_quad_points);
  std::vector<double> gradRhoOut(3*num_quad_points);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandlerEigen.begin_active(), endc = dofHandlerEigen.end();
  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{

	  fe_values.reinit (cell); 

	  (*rhoOutValues)[cell->id()] = std::vector<double>(num_quad_points);

	  std::fill(rhoOut.begin(),rhoOut.end(),0.0);

	
#ifdef ENABLE_PERIODIC_BC
	  std::vector<Vector<double> > tempPsi(num_quad_points);
 	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    tempPsi[q_point].reinit(2);
#else
	  std::vector<double> tempPsi(num_quad_points);
#endif



	  if(xc_id == 4)//GGA
	    {
	      (*gradRhoOutValues)[cell->id()] = std::vector<double>(3*num_quad_points);
	      std::fill(gradRhoOut.begin(),gradRhoOut.end(),0.0);
#ifdef ENABLE_PERIODIC_BC
	      std::vector<std::vector<Tensor<1,3,double> > > tempGradPsi(num_quad_points);
	      for(unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		tempGradPsi[q_point].resize(2);
#else
	      std::vector<Tensor<1,3,double> > tempGradPsi(num_quad_points);
#endif


	      for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
		{
		  for(unsigned int i=0; i<numEigenValues; ++i)
		    {
		      fe_values.get_function_values(*eigenVectorsOrig[kPoint][i], tempPsi);
		      fe_values.get_function_gradients(*eigenVectorsOrig[kPoint][i],tempGradPsi);

		      for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  double factor = (eigenValues[kPoint][i]-fermiEnergy)/(kb*TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			  rhoOut[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			  gradRhoOut[3*q_point + 0] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			  gradRhoOut[3*q_point + 1] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			  gradRhoOut[3*q_point + 2] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
#else
			  rhoOut[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0); 
			  gradRhoOut[3*q_point + 0] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][0];
			  gradRhoOut[3*q_point + 1] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][1];
			  gradRhoOut[3*q_point + 2] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][2];
#endif
			}
		    }
		}

	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  (*rhoOutValues)[cell->id()][q_point]           = rhoOut[q_point];
		  (*gradRhoOutValues)[cell->id()][3*q_point + 0] = gradRhoOut[3*q_point + 0];
		  (*gradRhoOutValues)[cell->id()][3*q_point + 1] = gradRhoOut[3*q_point + 1];
		  (*gradRhoOutValues)[cell->id()][3*q_point + 2] = gradRhoOut[3*q_point + 2];
		}

	    }
	  else
	    {
	      for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
		{
		  for(unsigned int i=0; i<numEigenValues; ++i)
		    {
		      fe_values.get_function_values(*eigenVectorsOrig[kPoint][i], tempPsi);

		      for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  double factor=(eigenValues[kPoint][i]-fermiEnergy)/(kb*TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef ENABLE_PERIODIC_BC
			  rhoOut[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
#else
			  rhoOut[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point]; 
#endif
			}
		    }
		}

	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  (*rhoOutValues)[cell->id()][q_point]=rhoOut[q_point];
		}

	    }

	}

    }

}
