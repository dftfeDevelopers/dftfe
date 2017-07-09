//source file for all the mixing schemes

//implement simple mixing scheme 
double dftClass::mixing_simple()
{
  double normValue=0.0;
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
  double alpha=0.5;
  
  //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> >* rhoInValuesOld = rhoInValues;
  rhoInValues=new std::map<dealii::CellId,std::vector<double> >;
  rhoInVals.push_back(rhoInValues); 

#ifdef xc_id
#if xc_id == 4
  std::map<dealii::CellId,std::vector<double> >* gradRhoInValuesOld = gradRhoInValues;
  gradRhoInValues = new std::map<dealii::CellId,std::vector<double> >;
  gradRhoInVals.push_back(gradRhoInValues);
#endif
#endif

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  fe_values.reinit (cell); 
	  (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);

#ifdef xc_id
#if xc_id == 4
	  (*gradRhoInValues)[cell->id()]=std::vector<double>(3*num_quad_points);
#endif
#endif
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    {
	      //Compute (rhoIn-rhoOut)^2
	      normValue+=std::pow(((*rhoInValuesOld)[cell->id()][q_point])- ((*rhoOutValues)[cell->id()][q_point]),2.0)*fe_values.JxW(q_point);
	      
	      //Simple mixing scheme
	      ((*rhoInValues)[cell->id()][q_point])=std::abs((1-alpha)*(*rhoInValuesOld)[cell->id()][q_point]+ alpha*(*rhoOutValues)[cell->id()][q_point]);

#ifdef xc_id
#if xc_id == 4
	      ((*gradRhoInValues)[cell->id()][3*q_point + 0])=std::abs((1-alpha)*(*gradRhoInValuesOld)[cell->id()][3*q_point + 0]+ alpha*(*gradRhoOutValues)[cell->id()][3*q_point + 0]);
	      ((*gradRhoInValues)[cell->id()][3*q_point + 1])=std::abs((1-alpha)*(*gradRhoInValuesOld)[cell->id()][3*q_point + 1]+ alpha*(*gradRhoOutValues)[cell->id()][3*q_point + 1]);
	      ((*gradRhoInValues)[cell->id()][3*q_point + 2])=std::abs((1-alpha)*(*gradRhoInValuesOld)[cell->id()][3*q_point + 2]+ alpha*(*gradRhoOutValues)[cell->id()][3*q_point + 2]);
#endif
#endif
	    }
	  
	}

    }
  
  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//implement anderson mixing scheme 
double dftClass::mixing_anderson(){
  double normValue=0.0;
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();
  double alpha=0.5;
  
   //initialize data structures
  int N=rhoOutVals.size()-1;
  pcout << "\nN:" << N << "\n";
  int NRHS=1, lda=N, ldb=N, info;
  std::vector<int> ipiv(N);
  std::vector<double> A(lda*N), c(ldb*NRHS); 
  for (int i=0; i<lda*N; i++) A[i]=0.0;
  for (int i=0; i<ldb*NRHS; i++) c[i]=0.0;
  
  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell); 
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	//fill coefficient matrix, rhs
	double Fn=((*rhoOutVals[N])[cell->id()][q_point])- ((*rhoInVals[N])[cell->id()][q_point]);
	for (int m=0; m<N; m++){
	  double Fnm=((*rhoOutVals[N-1-m])[cell->id()][q_point])- ((*rhoInVals[N-1-m])[cell->id()][q_point]);
	  for (int k=0; k<N; k++){
	    double Fnk=((*rhoOutVals[N-1-k])[cell->id()][q_point])- ((*rhoInVals[N-1-k])[cell->id()][q_point]); 
	    A[k*N+m] += (Fn-Fnm)*(Fn-Fnk)*fe_values.JxW(q_point); // (m,k)^th entry
	  }	  
	  c[m] += (Fn-Fnm)*(Fn)*fe_values.JxW(q_point); // (m)^th entry
	}
      }
    }
  }
  //accumulate over all processors
  std::vector<double> ATotal(lda*N), cTotal(ldb*NRHS); 
  MPI_Allreduce(&A[0], &ATotal[0], lda*N, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  MPI_Allreduce(&c[0], &cTotal[0], ldb*NRHS, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  //
  pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
  //solve for coefficients
  dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
  if((info > 0) && (this_mpi_process==0)) {
    printf( "Anderson Mixing: The diagonal element of the triangular factor of A,\n" );
    printf( "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n", info, info );
    exit(1);
  }
  double cn=1.0;
  for (int i=0; i<N; i++) cn-=cTotal[i];
  if(this_mpi_process==0) {
    printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
    for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
    printf("\n");
  }
  
  //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> >* rhoInValuesOld=rhoInValues;
  rhoInValues=new std::map<dealii::CellId,std::vector<double> >;
  rhoInVals.push_back(rhoInValues);

  //implement anderson mixing
  cell = dofHandler.begin_active();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);
      fe_values.reinit (cell); 
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	//Compute (rhoIn-rhoOut)^2
        normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
	//Anderson mixing scheme
	double rhoOutBar=cn*(*rhoOutVals[N])[cell->id()][q_point];
	double rhoInBar=cn*(*rhoInVals[N])[cell->id()][q_point];
	for (int i = 0; i < N; i++){
	  rhoOutBar+=cTotal[i]*(*rhoOutVals[N-1-i])[cell->id()][q_point];
	  rhoInBar+=cTotal[i]*(*rhoInVals[N-1-i])[cell->id()][q_point];
	}
	(*rhoInValues)[cell->id()][q_point]=std::abs((1-alpha)*rhoInBar+alpha*rhoOutBar);
      }
    }
  }

  //compute gradRho for GGA using mixing constants from rho mixing

#ifdef xc_id
#if xc_id == 4
  std::map<dealii::CellId,std::vector<double> >* gradRhoInValuesOld = gradRhoInValues;
  gradRhoInValues = new std::map<dealii::CellId,std::vector<double> >;
  gradRhoInVals.push_back(gradRhoInValues);
  cell = dofHandler.begin_active();
  for (; cell!=endc; ++cell) 
    {
    if (cell->is_locally_owned())
      {
	(*gradRhoInValues)[cell->id()]=std::vector<double>(3*num_quad_points);
	fe_values.reinit (cell); 
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	  {
	    //
	    //Anderson mixing scheme
	    //
	    double gradRhoXOutBar = cn*(*gradRhoOutVals[N])[cell->id()][3*q_point + 0];
	    double gradRhoYOutBar = cn*(*gradRhoOutVals[N])[cell->id()][3*q_point + 1];
	    double gradRhoZOutBar = cn*(*gradRhoOutVals[N])[cell->id()][3*q_point + 2];
	    
	    double gradRhoXInBar = cn*(*gradRhoInVals[N])[cell->id()][3*q_point + 0];
	    double gradRhoYInBar = cn*(*gradRhoInVals[N])[cell->id()][3*q_point + 1];
	    double gradRhoZInBar = cn*(*gradRhoInVals[N])[cell->id()][3*q_point + 2];
	    
	    for (int i = 0; i < N; i++)
	      {
		gradRhoXOutBar += cTotal[i]*(*gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 0];
		gradRhoYOutBar += cTotal[i]*(*gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 1];
		gradRhoZOutBar += cTotal[i]*(*gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 2];

		gradRhoXInBar += cTotal[i]*(*gradRhoInVals[N-1-i])[cell->id()][3*q_point + 0];
		gradRhoYInBar += cTotal[i]*(*gradRhoInVals[N-1-i])[cell->id()][3*q_point + 1];
		gradRhoZInBar += cTotal[i]*(*gradRhoInVals[N-1-i])[cell->id()][3*q_point + 2];
	      }

	    (*gradRhoInValues)[cell->id()][3*q_point + 0] = std::abs((1-alpha)*gradRhoXInBar+alpha*gradRhoXOutBar);
	    (*gradRhoInValues)[cell->id()][3*q_point + 1] = std::abs((1-alpha)*gradRhoYInBar+alpha*gradRhoYOutBar);
	    (*gradRhoInValues)[cell->id()][3*q_point + 2] = std::abs((1-alpha)*gradRhoZInBar+alpha*gradRhoZOutBar);
	  }
      }

    }
#endif
#endif


  return Utilities::MPI::sum(normValue, mpi_communicator);
}
