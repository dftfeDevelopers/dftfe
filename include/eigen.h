#ifndef eigen_H_
#define eigen_H_
#include "headers.h"

//Define eigenClass class
class eigenClass
{
  friend class dftClass;
public:
  eigenClass(dftClass* _dftPtr);
  void computeLocalHamiltonians(std::map<dealii::CellId,std::vector<double> >* rhoValues, const vectorType& phi);
  void HX(std::vector<vectorType*> &dst, const std::vector<vectorType*> &src);
  void XHX(std::vector<vectorType*> &src); 
 private:
  void implementHX(const dealii::MatrixFree<3,double>  &data,
		   std::vector<vectorType*>  &dst, 
		   const std::vector<vectorType*>  &src,
		   const std::pair<unsigned int,unsigned int> &cell_range) const;
  void implementXHX(const dealii::MatrixFree<3,double>  &data,
		   std::vector<vectorType*>  &dst, 
		   const std::vector<vectorType*>  &src,
		   const std::pair<unsigned int,unsigned int> &cell_range) const;
  void init ();
  void computeMassVector();

  //pointer to dft class
  dftClass* dftPtr;

  //FE data structres
  dealii::FE_Q<3>   FE;
  std::map<dealii::CellId,std::vector<double> >   localHamiltonians;
  std::map<dealii::CellId,std::vector<double> >*   localHamiltoniansPtr; //this ptr created to circumvent problem with const definition of HX
 
  //constraints
  dealii::ConstraintMatrix  constraintsNone;

  //data structures
  vectorType massVector;
  std::vector<double> XHXValue;
  std::vector<double>* XHXValuePtr;
  std::vector<vectorType*> HXvalue;

  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;
  //mutex thread for managing multi-thread writing to XHXvalue
  mutable dealii::Threads::ThreadMutex     assembler_lock;
};

#endif
