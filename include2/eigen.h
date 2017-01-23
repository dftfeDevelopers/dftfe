#ifndef eigen_H_
#define eigen_H_
#include "headers.h"

//Define eigenClass class
class eigenClass
{
  friend class dftClass;
public:
  eigenClass(dftClass* _dftPtr);
  void HX(const std::vector<vectorType*> &src, std::vector<vectorType*> &dst);
  void XHX(const std::vector<vectorType*> &src); 
 private:
  void implementHX(const dealii::MatrixFree<3,double>  &data,
		   std::vector<vectorType*>  &dst, 
		   const std::vector<vectorType*>  &src,
		   const std::pair<unsigned int,unsigned int> &cell_range) const;
  void init ();
  void computeMassVector();
  void computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
		   const vectorType & phi,
		   const vectorType & phiExt,
		   std::map<dealii::CellId,std::vector<double> >* pseudoValues=0);
  
  //pointer to dft class
  dftClass* dftPtr;


  //FE data structres
  dealii::FE_Q<3>   FE;
 
  //data structures
  vectorType massVector;
  std::vector<double> XHXValue;
  std::vector<vectorType*> HXvalue;
  dealii::Table<2, dealii::VectorizedArray<double> > vEff;

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
