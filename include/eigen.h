#ifndef eigen_H_
#define eigen_H_
#include "headers.h"

//
//Define eigenClass class
//
template <unsigned int FEOrder>
class eigenClass
{
  template <unsigned int FEOrder>
  friend class dftClass;

public:
  eigenClass(dftClass<FEOrder>* _dftPtr);
  void HX(const std::vector<vectorType*> &src, 
	  std::vector<vectorType*> &dst);

  void XHX(const std::vector<vectorType*> &src); 
 private:
  void implementHX(const dealii::MatrixFree<3,double>  &data,
		   std::vector<vectorType*>  &dst, 
		   const std::vector<vectorType*>  &src,
		   const std::pair<unsigned int,unsigned int> &cell_range) const;

  void computeNonLocalHamiltonianTimesX(const std::vector<vectorType*> &src,
					std::vector<vectorType*>       &dst);

  void init ();
  void computeMassVector();
  void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
		   const vectorType & phi,
		   const vectorType & phiExt,
		   std::map<dealii::CellId,std::vector<double> >* pseudoValues=0);

  void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
		   std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
		   const vectorType & phi,
		   const vectorType & phiExt,
		   std::map<dealii::CellId,std::vector<double> >* pseudoValues=0);

  
  //pointer to dft class
  dftClass<FEOrder>* dftPtr;


  //FE data structres
  dealii::FE_Q<3>   FE;
 
  //data structures
  vectorType massVector;
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > XHXValue;
#else
  std::vector<double> XHXValue;
#endif
  std::vector<vectorType*> HXvalue;
  dealii::Table<2, dealii::VectorizedArray<double> > vEff;

  dealii::Table<3, dealii::VectorizedArray<double> > derExcWithSigmaTimesGradRho;

  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;
  //mutex thread for managing multi-thread writing to XHXvalue
  mutable dealii::Threads::Mutex  assembler_lock;
};

#endif
