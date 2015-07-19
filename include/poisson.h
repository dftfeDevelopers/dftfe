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
  void computeLocalJacobians();
  void vmult(vectorType &dst, const vectorType &src) const;
private:
  void init ();
  void computeRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues);
  void solve(std::map<dealii::CellId,std::vector<double> >* rhoValues=0);
  void AX(const dealii::MatrixFree<3,double>  &data,
	  vectorType &dst, 
	  const vectorType &src,
	  const std::pair<unsigned int,unsigned int> &cell_range) const;
  //pointer to dft class
  dftClass* dftPtr;

  //FE data structres
  dealii::FE_Q<3>   FE;
  std::map<dealii::CellId,std::vector<double> >   localJacobians;
  std::map<dealii::CellId,std::vector<double> >*   localJacobiansPtr; //this ptr created to circumvent problem with const definition of vmult and Ax
  //constraints
  dealii::ConstraintMatrix  constraintsNone, constraintsZero, constraints1byR;
  std::map<dealii::types::global_dof_index, double> valuesZero, values1byR;

  //data structures
  vectorType rhs, Ax;
  vectorType jacobianDiagonal;
  vectorType phiTotRhoIn, phiTotRhoOut, phiExt;
 
  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;
};

#endif
