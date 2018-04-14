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

//source file for all the mixing schemes

#include "../../include/dftParameters.h"


//implement simple mixing scheme
template<unsigned int FEOrder>
double dftClass<FEOrder>::mixing_simple()
{
  double normValue=0.0;
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();


  //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> > rhoInValuesOld= *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValues=&(rhoInVals.back());


  //create new gradRhoValue tables
  std::map<dealii::CellId,std::vector<double> > gradRhoInValuesOld;

  if(dftParameters::xc_id == 4)
    {
      gradRhoInValuesOld=*gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValues=&(gradRhoInVals.back());
    }

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  fe_values.reinit (cell);
	  (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);


	  if(dftParameters::xc_id == 4)
	    (*gradRhoInValues)[cell->id()]=std::vector<double>(3*num_quad_points);


	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    {
	      //Compute (rhoIn-rhoOut)^2
	      normValue+=std::pow(((rhoInValuesOld)[cell->id()][q_point])- ((*rhoOutValues)[cell->id()][q_point]),2.0)*fe_values.JxW(q_point);

	      //Simple mixing scheme
	      ((*rhoInValues)[cell->id()][q_point])=std::abs((1-dftParameters::mixingParameter)*(rhoInValuesOld)[cell->id()][q_point]+ dftParameters::mixingParameter*(*rhoOutValues)[cell->id()][q_point]);


	      if(dftParameters::xc_id == 4)
		{
		  ((*gradRhoInValues)[cell->id()][3*q_point + 0])= ((1-dftParameters::mixingParameter)*(gradRhoInValuesOld)[cell->id()][3*q_point + 0]+ dftParameters::mixingParameter*(*gradRhoOutValues)[cell->id()][3*q_point + 0]);
		  ((*gradRhoInValues)[cell->id()][3*q_point + 1])= ((1-dftParameters::mixingParameter)*(gradRhoInValuesOld)[cell->id()][3*q_point + 1]+ dftParameters::mixingParameter*(*gradRhoOutValues)[cell->id()][3*q_point + 1]);
		  ((*gradRhoInValues)[cell->id()][3*q_point + 2])= ((1-dftParameters::mixingParameter)*(gradRhoInValuesOld)[cell->id()][3*q_point + 2]+ dftParameters::mixingParameter*(*gradRhoOutValues)[cell->id()][3*q_point + 2]);
		}

	    }

	}

    }

  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//implement anderson mixing scheme
template<unsigned int FEOrder>
double dftClass<FEOrder>::mixing_anderson(){
  double normValue=0.0;
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();


  //initialize data structures
  int N = rhoOutVals.size()- 1;
  //pcout << "\nN:" << N << "\n";
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
	double Fn=((rhoOutVals[N])[cell->id()][q_point])- ((rhoInVals[N])[cell->id()][q_point]);
	for (int m=0; m<N; m++){
	  double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])- ((rhoInVals[N-1-m])[cell->id()][q_point]);
	  for (int k=0; k<N; k++){
	    double Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])- ((rhoInVals[N-1-k])[cell->id()][q_point]);
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
  //pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
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
    //printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
    //for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
    //printf("\n");
  }

  //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> > rhoInValuesOld= *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValues=&(rhoInVals.back());


  //implement anderson mixing
  cell = dofHandler.begin_active();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	//Compute (rhoIn-rhoOut)^2
        normValue+=std::pow((rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
	//Anderson mixing scheme
	double rhoOutBar=cn*(rhoOutVals[N])[cell->id()][q_point];
	double rhoInBar=cn*(rhoInVals[N])[cell->id()][q_point];
	for (int i = 0; i < N; i++){
	  rhoOutBar+=cTotal[i]*(rhoOutVals[N-1-i])[cell->id()][q_point];
	  rhoInBar+=cTotal[i]*(rhoInVals[N-1-i])[cell->id()][q_point];
	}
	(*rhoInValues)[cell->id()][q_point]=std::abs((1-dftParameters::mixingParameter)*rhoInBar+dftParameters::mixingParameter*rhoOutBar);
      }
    }
  }

  //compute gradRho for GGA using mixing constants from rho mixing


  if(dftParameters::xc_id == 4)
    {
      std::map<dealii::CellId,std::vector<double> > gradRhoInValuesOld=*gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValues=&(gradRhoInVals.back());
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
		  double gradRhoXOutBar = cn*(gradRhoOutVals[N])[cell->id()][3*q_point + 0];
		  double gradRhoYOutBar = cn*(gradRhoOutVals[N])[cell->id()][3*q_point + 1];
		  double gradRhoZOutBar = cn*(gradRhoOutVals[N])[cell->id()][3*q_point + 2];

		  double gradRhoXInBar = cn*(gradRhoInVals[N])[cell->id()][3*q_point + 0];
		  double gradRhoYInBar = cn*(gradRhoInVals[N])[cell->id()][3*q_point + 1];
		  double gradRhoZInBar = cn*(gradRhoInVals[N])[cell->id()][3*q_point + 2];

		  for (int i = 0; i < N; i++)
		    {
		      gradRhoXOutBar += cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 0];
		      gradRhoYOutBar += cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 1];
		      gradRhoZOutBar += cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point + 2];

		      gradRhoXInBar += cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point + 0];
		      gradRhoYInBar += cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point + 1];
		      gradRhoZInBar += cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point + 2];
		    }

		  (*gradRhoInValues)[cell->id()][3*q_point + 0] = ((1-dftParameters::mixingParameter)*gradRhoXInBar+dftParameters::mixingParameter*gradRhoXOutBar);
		  (*gradRhoInValues)[cell->id()][3*q_point + 1] = ((1-dftParameters::mixingParameter)*gradRhoYInBar+dftParameters::mixingParameter*gradRhoYOutBar);
		  (*gradRhoInValues)[cell->id()][3*q_point + 2] = ((1-dftParameters::mixingParameter)*gradRhoZInBar+dftParameters::mixingParameter*gradRhoZOutBar);
		}
	    }

	}
    }



  return Utilities::MPI::sum(normValue, mpi_communicator);
}

template<unsigned int FEOrder>
double dftClass<FEOrder>::mixing_simple_spinPolarized()
{
  double normValue=0.0;
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();

   //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> > rhoInValuesOld= *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValues=&(rhoInVals.back());

  std::map<dealii::CellId,std::vector<double> > rhoInValuesOldSpinPolarized= *rhoInValuesSpinPolarized;
  rhoInValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
  //

  //create new gradRhoValue tables
  std::map<dealii::CellId,std::vector<double> > gradRhoInValuesOld;
  std::map<dealii::CellId,std::vector<double> > gradRhoInValuesOldSpinPolarized;

  if(dftParameters::xc_id == 4)
    {
      gradRhoInValuesOld=*gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValues=&(gradRhoInVals.back());
      //
      gradRhoInValuesOldSpinPolarized=*gradRhoInValuesSpinPolarized;
      gradRhoInValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());

    }

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  fe_values.reinit (cell);
	  // if (s==0) {
	  (*rhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(2*num_quad_points);
	  (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);
	  // }

	  if(dftParameters::xc_id == 4)
	    {
	      (*gradRhoInValues)[cell->id()]=std::vector<double>(3*num_quad_points);
	      (*gradRhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(6*num_quad_points);
            }


	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    {
	      //Compute (rhoIn-rhoOut)^2
	      //normValue+=std::pow(((*rhoInValuesOld)[cell->id()][2*q_point+s])- ((*rhoOutValues)[cell->id()][2*q_point+s]),2.0)*fe_values.JxW(q_point);

	      //Simple mixing scheme
	      (*rhoInValuesSpinPolarized)[cell->id()][2*q_point]= std::abs((1-dftParameters::mixingParameter)*(rhoInValuesOldSpinPolarized)[cell->id()][2*q_point]+
									   dftParameters::mixingParameter*(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point]);
	      (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1]= std::abs((1-dftParameters::mixingParameter)*(rhoInValuesOldSpinPolarized)[cell->id()][2*q_point+1]+
									     dftParameters::mixingParameter*(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1]);

	      (*rhoInValues)[cell->id()][q_point]=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point] + (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1] ;
	      //
	      normValue+=std::pow((rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);

	      if(dftParameters::xc_id == 4)
		{
		  for (unsigned int i=0; i<6; ++i)
		    {
		      ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + i])=
			((1-dftParameters::mixingParameter)*(gradRhoInValuesOldSpinPolarized)[cell->id()][6*q_point + i]+ dftParameters::mixingParameter*(*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + i]);
		    }

		  //
		  ((*gradRhoInValues)[cell->id()][3*q_point + 0])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 0]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 3]) ;
		  ((*gradRhoInValues)[cell->id()][3*q_point + 1])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 1]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 4]) ;
		  ((*gradRhoInValues)[cell->id()][3*q_point + 2])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 2]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 5]) ;
		}

	    }

	}

    }

  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//implement anderson mixing scheme
template<unsigned int FEOrder>
double dftClass<FEOrder>::mixing_anderson_spinPolarized(){
  double normValue=0.0;
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int num_quad_points = quadrature.size();



  //initialize data structures
  int N = rhoOutVals.size()- 1;
  //pcout << "\nN:" << N << "\n";
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
	double Fn=((rhoOutVals[N])[cell->id()][q_point])- ((rhoInVals[N])[cell->id()][q_point]);
	for (int m=0; m<N; m++){
	  double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])- ((rhoInVals[N-1-m])[cell->id()][q_point]);
	  for (int k=0; k<N; k++){
	    double Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])- ((rhoInVals[N-1-k])[cell->id()][q_point]);
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
  //pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
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
    //printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
    //for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
    //printf("\n");
  }

  //create new rhoValue tables
  std::map<dealii::CellId,std::vector<double> > rhoInValuesOld= *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValues=&(rhoInVals.back());

  //
  std::map<dealii::CellId,std::vector<double> > rhoInValuesOldSpinPolarized= *rhoInValuesSpinPolarized;
  rhoInValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> >());
  rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());

  //
  //implement anderson mixing
  cell = dofHandler.begin_active();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //if (s==0) {
      (*rhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(2*num_quad_points);
      (*rhoInValues)[cell->id()]=std::vector<double>(num_quad_points);
      //}
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	//Compute (rhoIn-rhoOut)^2
        //normValue+=std::pow((*rhoInValuesOld)[cell->id()][2*q_point+s]-(*rhoOutValues)[cell->id()][2*q_point+s],2.0)*fe_values.JxW(q_point);
	//Anderson mixing scheme
	//normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point],2.0)*fe_values.JxW(q_point);
	//normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point+1]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1],2.0)*fe_values.JxW(q_point);
	normValue+=std::pow((rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
	double rhoOutBar1=cn*(rhoOutValsSpinPolarized[N])[cell->id()][2*q_point];
	double rhoInBar1=cn*(rhoInValsSpinPolarized[N])[cell->id()][2*q_point];
	for (int i = 0; i < N; i++){
	  rhoOutBar1+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
	  rhoInBar1+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
	}
	(*rhoInValuesSpinPolarized)[cell->id()][2*q_point]=std::abs((1-dftParameters::mixingParameter)*rhoInBar1+dftParameters::mixingParameter*rhoOutBar1);
	//
        double rhoOutBar2=cn*(rhoOutValsSpinPolarized[N])[cell->id()][2*q_point+1];
	double rhoInBar2=cn*(rhoInValsSpinPolarized[N])[cell->id()][2*q_point+1];
	for (int i = 0; i < N; i++){
	  rhoOutBar2+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
	  rhoInBar2+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
	}
	(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1]=std::abs((1-dftParameters::mixingParameter)*rhoInBar2+dftParameters::mixingParameter*rhoOutBar2);
	//
	//if (s==1)
        //   {
	//    (*rhoInValues)[cell->id()][q_point]+=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s] ;
        //     normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
	//   }
	//else
	//    (*rhoInValues)[cell->id()][q_point]=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s] ;
	(*rhoInValues)[cell->id()][q_point]=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point] + (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1] ;
	//normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
      }
    }
  }

  //compute gradRho for GGA using mixing constants from rho mixing


  if(dftParameters::xc_id == 4)
    {
      std::map<dealii::CellId,std::vector<double> > gradRhoInValuesOld=*gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValues=&(gradRhoInVals.back());

      //
      gradRhoInValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> >());
      gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
      //
      cell = dofHandler.begin_active();
      for (; cell!=endc; ++cell)
	{
	  if (cell->is_locally_owned())
	    {
	      (*gradRhoInValues)[cell->id()]=std::vector<double>(3*num_quad_points);
	      (*gradRhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(6*num_quad_points);
	      //
	      fe_values.reinit (cell);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  //
		  //Anderson mixing scheme spin up
		  //
		  double gradRhoXOutBar1 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 0];
		  double gradRhoYOutBar1 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 1];
		  double gradRhoZOutBar1 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 2];

		  double gradRhoXInBar1 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 0];
		  double gradRhoYInBar1 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 1];
		  double gradRhoZInBar1 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 2];

		  for (int i = 0; i < N; i++)
		    {
		      gradRhoXOutBar1 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 0];
		      gradRhoYOutBar1 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 1];
		      gradRhoZOutBar1 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 2];

		      gradRhoXInBar1 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 0];
		      gradRhoYInBar1 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 1];
		      gradRhoZInBar1 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 2];
		    }
		  //
		  //Anderson mixing scheme spin down
		  //
		  double gradRhoXOutBar2 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 3];
		  double gradRhoYOutBar2 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 4];
		  double gradRhoZOutBar2 = cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point + 5];

		  double gradRhoXInBar2 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 3];
		  double gradRhoYInBar2 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 4];
		  double gradRhoZInBar2 = cn*(gradRhoInValsSpinPolarized[N])[cell->id()][6*q_point + 5];

		  for (int i = 0; i < N; i++)
		    {
		      gradRhoXOutBar2 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 3];
		      gradRhoYOutBar2 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 4];
		      gradRhoZOutBar2 += cTotal[i]*(gradRhoOutValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 5];

		      gradRhoXInBar2 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 3];
		      gradRhoYInBar2 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 4];
		      gradRhoZInBar2 += cTotal[i]*(gradRhoInValsSpinPolarized[N-1-i])[cell->id()][6*q_point + 5];
		    }
		  //
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 0] = ((1-dftParameters::mixingParameter)*gradRhoXInBar1+dftParameters::mixingParameter*gradRhoXOutBar1);
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 1] = ((1-dftParameters::mixingParameter)*gradRhoYInBar1+dftParameters::mixingParameter*gradRhoYOutBar1);
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 2] = ((1-dftParameters::mixingParameter)*gradRhoZInBar1+dftParameters::mixingParameter*gradRhoZOutBar1);
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 3] = ((1-dftParameters::mixingParameter)*gradRhoXInBar2+dftParameters::mixingParameter*gradRhoXOutBar2);
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 4] = ((1-dftParameters::mixingParameter)*gradRhoYInBar2+dftParameters::mixingParameter*gradRhoYOutBar2);
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 5] = ((1-dftParameters::mixingParameter)*gradRhoZInBar2+dftParameters::mixingParameter*gradRhoZOutBar2);

		  ((*gradRhoInValues)[cell->id()][3*q_point + 0])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 0]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 3]) ;
		  ((*gradRhoInValues)[cell->id()][3*q_point + 1])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 1]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 4]) ;
		  ((*gradRhoInValues)[cell->id()][3*q_point + 2])= ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 2]) + ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 5]) ;
		}
	    }

	}
    }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}
