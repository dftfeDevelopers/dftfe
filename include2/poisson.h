#ifndef poisson_H_
#define poisson_H_
#include "headers.h"

typedef double dataType;
typedef dealii::parallel::distributed::Vector<double> vectorType;

class dftClass;

//Define poisson class
class poissonClass
{
  friend class dftClass; 
public:
  poissonClass(dftClass* _dftPtr);
  void vmult(vectorType &dst, const vectorType &src) const;
  void precondition_Jacobi(vectorType& dst, const vectorType& src, const double omega) const;
  void subscribe (const char *identifier=0) const{}; //function needed to mimic SparseMatrix for Jacobi Preconditioning
  void unsubscribe (const char *identifier=0) const{}; //function needed to mimic SparseMatrix for Jacobi Preconditioning
  bool operator!= (double val) const {return true;}; //function needed to mimic SparseMatrix
  typedef unsigned int size_type; //add this line


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
  dftClass* dftPtr;

  //FE data structres
  dealii::FE_Q<3> FE;
  //constraints
  //dealii::ConstraintMatrix  constraintsNone, constraints1byR;
  //std::map<dealii::types::global_dof_index, double> values1byR;
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
