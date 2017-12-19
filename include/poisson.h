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

#ifndef poisson_H_
#define poisson_H_
#include "headers.h"
#include "constants.h"
//#include "dft.h"

typedef double dataType;
typedef dealii::parallel::distributed::Vector<double> vectorType;

//forward declaration
template <unsigned int T>
class dftClass;
template <unsigned int T>
class forceClass;
//
//Define poisson class
//
template <unsigned int FEOrder>
class poissonClass
{
  template <unsigned int T>
  friend class dftClass; 

  template <unsigned int T>
  friend class forceClass; 
  
public:
  poissonClass(dftClass<FEOrder>* _dftPtr);
  void vmult(vectorType &dst, vectorType &src) const;
  void precondition_Jacobi(vectorType& dst, const vectorType& src, const double omega) const;
  void subscribe (const char *identifier=0) const{};   //function needed to mimic SparseMatrix for Jacobi preconditioning
  void unsubscribe (const char *identifier=0) const{}; //function needed to mimic SparseMatrix for Jacobi preconditioning
  bool operator!= (double val) const {return true;};   //function needed to mimic SparseMatrix
  typedef unsigned int size_type;                      //add this line


private: 
  void init ();

  void computeRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues);

  void computeRHS2();

  void solve(vectorType& phi, 
	     int constraintMatrixId, 
	     std::map<dealii::CellId,std::vector<double> >* rhoValues=0);

  void AX(const dealii::MatrixFree<3,double>  &data,
	  vectorType &dst, 
	  const vectorType &src,
	  const std::pair<unsigned int,unsigned int> &cell_range) const;

  //pointer to dft class
  dftClass<FEOrder> * dftPtr;

  //FE data structres
  dealii::FE_Q<3> FE;

  int d_constraintMatrixId;

  //data structures
  vectorType rhs, rhs2, jacobianDiagonal;
  vectorType phiTotRhoIn, phiTotRhoOut, phiExt, vselfBinScratch;
  
  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;
};



#endif
