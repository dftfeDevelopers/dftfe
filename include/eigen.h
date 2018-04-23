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

#ifndef eigen_H_
#define eigen_H_
#include "headers.h"
#include "constants.h"

namespace dftfe {

    using namespace dealii;
    template <unsigned int T> class dftClass;

    //
    //Define eigenClass class
    //
    template <unsigned int FEOrder>
    class eigenClass
    {
      template <unsigned int T>
      friend class dftClass;

      template <unsigned int T>
      friend class symmetryClass;

    public:
      eigenClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica);
      void HX(std::vector<vectorType> &src,
	      std::vector<vectorType> &dst);

      void XHX(std::vector<vectorType> &src);
     private:
      void implementHX(const dealii::MatrixFree<3,double>  &data,
		       std::vector<vectorType>  &dst,
		       const std::vector<vectorType>  &src,
		       const std::pair<unsigned int,unsigned int> &cell_range) const;

      void computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
					    std::vector<vectorType>       &dst);

      void computeNonLocalHamiltonianTimesXMemoryOpt(const std::vector<vectorType> &src,
						     std::vector<vectorType>       &dst);

      void init ();
      void computeMassVector();
      void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);
      void computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       unsigned int j,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      void computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       unsigned int j,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);



      //pointer to dft class
      dftClass<FEOrder>* dftPtr;


      //FE data structres
      dealii::FE_Q<3>   FE;

      //data structures
      vectorType invSqrtMassVector,sqrtMassVector;
#ifdef ENABLE_PERIODIC_BC
      std::vector<std::complex<double> > XHXValue;
#else
      std::vector<double> XHXValue;
#endif

      dealii::Table<2, dealii::VectorizedArray<double> > vEff;
      dealii::Table<3, dealii::VectorizedArray<double> > derExcWithSigmaTimesGradRho;

      //parallel objects
      const MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;
      dealii::ConditionalOStream   pcout;

      //compute-time logger
      dealii::TimerOutput computing_timer;
      //mutex thread for managing multi-thread writing to XHXvalue
      mutable dealii::Threads::Mutex  assembler_lock;
    };

}
#endif
