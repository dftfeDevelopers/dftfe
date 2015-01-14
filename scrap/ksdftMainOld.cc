//Include all deal.II header files
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h> 
#include <deal.II/numerics/matrix_tools.h> 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/slepc_solver.h>
//Include generic C++ headers
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>
//Include alglib
#include "/nfs/mcfs_home/rudraa/Public/alglib/cpp/src/interpolation.h"
#include "/nfs/mcfs_home/rudraa/Public/libxc/libxc-2.2.0/installDir/include/xc.h"

//lapack routine
extern "C"{
void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info );
}

//Initialize Namespace
using namespace dealii;

#define spectral
#define standardEigenProblem

//Define parameters
const unsigned int dim=3;
const double radius=20.0;
unsigned int degree=4;
unsigned int n_refinement_steps=0;
unsigned int numElectrons=6;
unsigned int numEigenValues=numElectrons/2+2;
unsigned int quadratureRule=5;
unsigned int numAtomTypes=1;
double atomCharge=6.0;

//Define constants
double TVal=100.0;
double kb = 3.166811429e-06;

void readFile(std::vector<std::vector<double> > &data){
  unsigned int numColumns=2; //change this for being generic
  std::vector<double> rowData(numColumns, 0.0);
  std::ifstream readFile;
  readFile.open("rhoInitialGuess/rho_C");
  if (readFile.is_open()) {
    while (!readFile.eof()) {
      for(unsigned int i=0; i <numColumns; i++)
	readFile>>rowData[i];
      data.push_back(rowData);
    }
  }
  readFile.close();
  return;
}

// Define poissonProblem class
class poissonProblem
{
public:
  poissonProblem ();
  void run ();
  
private:
  void generateMesh();
  double totalCharge();
  void setup_system ();
  void assemble_system (bool rhoIn);
  void assemble_eigenSystem();
  void solve_cg (bool rhoIn);
  void solve_eigenSystem();  
  void output_results () const;
  void locate_origin();
  void computeRhoOut();
  void compute_energy();
  void compute_fermienergy();
  double mixing_simple();
  double mixing_anderson();
  Triangulation<dim>     triangulation;
  FE_Q<dim>              fe;
  DoFHandler<dim>        dof_handler;
  ConstraintMatrix meshConstraints, meshConstraintsKS;

  // Sparsity pattern and values of the system matrix resulting from
  // the discretization of the Laplace equation.
  bool setPoissonMatrixPhiTot, setPoissonMatrixPhiExt;
  PETScWrappers::MPI::SparseMatrix poissonMatrixPhiTot, poissonMatrixPhiExt;
  PETScWrappers::MPI::SparseMatrix ksHamiltonian_matrix;
  PETScWrappers::MPI::SparseMatrix ksMass_matrix;
  PETScWrappers::MPI::Vector massVector;
 
  // Right hand side and solution vectors.
  PETScWrappers::MPI::Vector       phiTotRhoIn;
  PETScWrappers::MPI::Vector       phiTotRhoOut;
  PETScWrappers::MPI::Vector       phiExtRhoOut;
  PETScWrappers::MPI::Vector       rhsPhiTot, rhsPhiExt;
  
  std::vector<PETScWrappers::MPI::Vector> kohnShamEigenVectors;
  std::vector<PETScWrappers::MPI::Vector> kohnShamEigenVectorsProjected;
  std::vector<double> kohnShamEigenValues;
  
  MPI_Comm mpi_communicator;
  
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  
  ConditionalOStream   pcout;
  TimerOutput computing_timer;
  unsigned int originID;
  
  std::vector<Table<2,double>*> rhoInVals, rhoOutVals;
  Table<2,double> *rhoInValues, *rhoOutValues;
  Table<2,double> rhoInitialGuess;
  std::vector<alglib::spline1dinterpolant> denSpline;
  xc_func_type funcX, funcC;
  double fermiEnergy;
};

// Specify that bi-linear elements (denoted by the
// parameter to the finite element object, which indicates the polynomial degree) is used.
// Associate the dof_handler variable to the triangulation
poissonProblem::poissonProblem ()
    :
      pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
#ifdef spectral
      fe (QGaussLobatto<1>(degree+1)),
#else
      fe (degree),
#endif
      dof_handler (triangulation),
      computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
      mpi_communicator (MPI_COMM_WORLD),
      n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
      denSpline(numAtomTypes)
{}

// Generate triangulation.
void poissonProblem::generateMesh()
{
  //GridGenerator::hyper_cube (triangulation, -1, 1);
  //triangulation.refine_global (6);
  //Read mesh written out in UCD format
  static const Point<dim> center = Point<dim>();
  static const HyperBallBoundary<dim, dim> boundary(center,radius);
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  //Read mesh in UCD format generated from Cubit
  std::ifstream f("ucd.inp");
  gridin.read_ucd(f);
  triangulation.set_boundary(0, boundary);
  triangulation.refine_global (n_refinement_steps);
  pcout << "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl;
  // Output the total number of cells.
  pcout << "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl;
  GridTools::partition_triangulation (n_mpi_processes, triangulation);
}

//Boundary condition function
template <int dim>
class BoundaryValuesFunction : public Function<dim>
{
public:
  BoundaryValuesFunction (bool _zeroBC) : Function<dim>(), zeroBC(_zeroBC) {}
  virtual double value (const Point<dim> &p,
			const unsigned int component = 0) const;
  bool zeroBC;
};
template <int dim>
double BoundaryValuesFunction<dim>::value (const Point<dim> &p,
					   const unsigned int /*component*/) const
{
  double value=0.0;
  if (!zeroBC) value=-atomCharge/sqrt(p.square());
  return value;
}

//Compute total charge                                                                                                                                                                          
double poissonProblem::totalCharge()
{
  double normValue=0.0;
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
        normValue+=(*rhoInValues)(cellID,q_point)*fe_values.JxW(q_point);
      }
    }
    cellID++;
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//Setup data structures
void poissonProblem::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::subdomain_wise (dof_handler);

    const unsigned int n_local_dofs
      = DoFTools::count_dofs_with_subdomain_association (dof_handler, this_mpi_process);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
        pcout << (p==0 ? ' ' : '+') << (DoFTools::count_dofs_with_subdomain_association (dof_handler, p));
    pcout << ")" << std::endl;
    //constraints
    IndexSet system_index_set, system_relevant_set;
    system_index_set = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler, system_relevant_set);
    meshConstraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,meshConstraints);
    meshConstraints.close ();
    meshConstraintsKS.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,meshConstraintsKS);
    DoFTools::make_zero_boundary_constraints (dof_handler, meshConstraintsKS);
    meshConstraintsKS.close ();
    
    // Initialize poissonMatrixPhiTot using the sparsity pattern.
    setPoissonMatrixPhiTot=false;
    setPoissonMatrixPhiExt=false;
    poissonMatrixPhiTot.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());
    poissonMatrixPhiExt.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());
    ksHamiltonian_matrix.reinit (mpi_communicator,
				 dof_handler.n_dofs(),
				 dof_handler.n_dofs(),
				 n_local_dofs,
				 n_local_dofs,
				 dof_handler.max_couplings_between_dofs());
    ksMass_matrix.reinit (mpi_communicator,
			  dof_handler.n_dofs(),
			  dof_handler.n_dofs(),
			  n_local_dofs,
			  n_local_dofs,
			  dof_handler.max_couplings_between_dofs());
    // Set the sizes of the right hand side vector and the solution vector.;
    phiTotRhoIn.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    phiTotRhoOut.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    phiExtRhoOut.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    rhsPhiTot.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    rhsPhiExt.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    massVector.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    kohnShamEigenValues.resize(numEigenValues);
    kohnShamEigenVectors.resize(numEigenValues);
    kohnShamEigenVectorsProjected.resize(numEigenValues);
    for (unsigned int i=0; i<numEigenValues; ++i){
      kohnShamEigenVectors[i].reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
      kohnShamEigenVectorsProjected[i].reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    }
    //Initialize electron density table storage
    rhoInValues=new Table<2,double>(triangulation.n_active_cells(),std::pow(quadratureRule,dim));
    rhoInVals.push_back(rhoInValues);

    //Readin single atom rho initial guess
    pcout << "reading initial guess for rho\n";
    std::vector<std::vector<std::vector<double> > > singleAtomElectronDensity(numAtomTypes);
    readFile(singleAtomElectronDensity[0]);
    unsigned int numRows = singleAtomElectronDensity[0].size()-1;
    double xData[numRows], yData[numRows];
    for(unsigned int irow = 0; irow < numRows; ++irow){
      xData[irow] = singleAtomElectronDensity[0][irow][0];
      yData[irow] = singleAtomElectronDensity[0][irow][1];
    }
    alglib::real_1d_array x;
    x.setcontent(numRows,xData);
    alglib::real_1d_array y;
    y.setcontent(numRows,yData);
    alglib::ae_int_t natural_bound_type = 1;
    spline1dbuildcubic(x, y, numRows,
		       natural_bound_type, 0.0,
		       natural_bound_type, 0.0,
		       denSpline[0]);
    std::vector<double> outerMostPointDen(numAtomTypes);
    outerMostPointDen[0]= xData[numRows-1];
    //Initialize rho
    pcout << "setting rho initial values\n";
    QGauss<dim>  quadrature_formula(quadratureRule);
    FEValues<dim> fe_values (fe, quadrature_formula, update_values);
    const unsigned int n_q_points    = quadrature_formula.size();
    DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    unsigned int cellID=0;
    for (; cell!=endc; ++cell) {
      for (unsigned int q=0; q<n_q_points; ++q){
	MappingQ<dim> test(1); 
	Point<dim> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	double distance=std::sqrt(quadPoint.square());
	if(distance <= outerMostPointDen[0]){
	  (*rhoInValues)(cellID,q)=std::abs(alglib::spline1dcalc(denSpline[0], distance));
	}
	else{
	  (*rhoInValues)(cellID,q)=0.0;
	} 
      }
      cellID++;
    }
    double charge=totalCharge();
    printf("inital charge: %18.12e\n", charge);
    //Normalize rho
    cellID=0;
    for (; cell!=endc; ++cell) {
      for (unsigned int q=0; q<n_q_points; ++q){
          (*rhoInValues)(cellID,q)*=1.0/charge;
      }
      cellID++;
    }

    //Initialize libxc
    int exceptParamX, exceptParamC;
    exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
    exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
    if(exceptParamX != 0 || exceptParamC != 0){
      std::cout<<"-------------------------------------"<<std::endl;
      std::cout<<"Exchange or Correlation Functional not found"<<std::endl;
      std::cout<<"-------------------------------------"<<std::endl;
    }
    else{
      std::cout<<"-------------------------------------"<<std::endl;
      cout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	  <<endl<<funcX.info->refs<<endl;
      cout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	  <<endl<<funcX.info->refs<<endl;
      std::cout<<"-------------------------------------"<<std::endl;	  
    }
}

//Assemble poisson problem
void poissonProblem::assemble_system (bool rhoIn=true)
{
  computing_timer.enter_section("A"); 
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhsPhiTot (dofs_per_cell), cell_rhsPhiExt(dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  //Intiailize data structures to zero
  MatZeroEntries(poissonMatrixPhiTot);
  if (rhoIn) {
    VecSet(phiTotRhoIn, 0.0); VecSet(rhsPhiTot,0.0);
  }
  else {
    MatZeroEntries(poissonMatrixPhiExt);
    VecSet(phiTotRhoOut, 0.0); VecSet(rhsPhiTot,0.0);
    VecSet(phiExtRhoOut, 0.0); VecSet(rhsPhiExt,0.0);
  }
  computing_timer.exit_section("A"); 
  
    // Loop through all cells.
    DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    unsigned int cellID=0;
    for (; cell!=endc; ++cell) {
      if (cell->subdomain_id() == this_mpi_process){
	// Compute values for current cell.
	fe_values.reinit (cell);
	// Reset the local matrix and right hand side.
	cell_matrix = 0;
	cell_rhsPhiTot = 0;
	cell_rhsPhiExt = 0;
	computing_timer.enter_section("B"); 
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  for (unsigned int j=0; j<dofs_per_cell; ++j){
	    for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	      cell_matrix(i,j) += (1.0/(4.0*M_PI))*(fe_values.shape_grad (i, q_point) *
						    fe_values.shape_grad (j, q_point) *
						    fe_values.JxW (q_point));
	    }
	  }
	}
	computing_timer.exit_section("B");
	computing_timer.enter_section("C");  
	// Loop through degree of freedoms on each cell and compute cell_rhs using quadrature.
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	    if (rhoIn) {
	      cell_rhsPhiTot(i) += fe_values.shape_value(i, q_point)*(*rhoInValues)(cellID, q_point)*fe_values.JxW (q_point);
	    }
	    else {
	      cell_rhsPhiTot(i) += fe_values.shape_value(i, q_point)*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	    }
	  }
	}
	computing_timer.exit_section("C");
	computing_timer.enter_section("D");  
	// Copy dof indices of the cell to "local_dof_indices"
	cell->get_dof_indices (local_dof_indices);
	//assemble to global matrices
        if (!rhoIn){                 
	  computing_timer.enter_section("I");        
	  meshConstraints.distribute_local_to_global(cell_matrix,local_dof_indices,poissonMatrixPhiExt);
	  computing_timer.exit_section("I");        
	  computing_timer.enter_section("G");                                                                                                      	  meshConstraints.distribute_local_to_global(cell_matrix,local_dof_indices,poissonMatrixPhiTot);                                           	  computing_timer.exit_section("G");
	}
	else{
	  computing_timer.enter_section("J");  
	  meshConstraints.distribute_local_to_global(cell_matrix,local_dof_indices,poissonMatrixPhiTot);
	  computing_timer.exit_section("J"); 
	}
	computing_timer.enter_section("H");   
	meshConstraints.distribute_local_to_global(cell_rhsPhiTot,local_dof_indices,rhsPhiTot);
	computing_timer.exit_section("H");
	computing_timer.exit_section("D"); 
      }
      cellID++;
    }
    computing_timer.enter_section("E"); 
    //Add nodal force to the node at the origin
    if (rhsPhiTot.in_local_range(originID)){
      std::vector<unsigned int> local_dof_indices_origin; local_dof_indices_origin.push_back(originID);
      Vector<double> cell_rhs_origin (1); cell_rhs_origin(0)=-atomCharge;
      meshConstraints.distribute_local_to_global(cell_rhs_origin,local_dof_indices_origin,rhsPhiTot);
      if(!rhoIn) meshConstraints.distribute_local_to_global(cell_rhs_origin,local_dof_indices_origin,rhsPhiExt);
    }
    poissonMatrixPhiTot.compress(VectorOperation::add);
    rhsPhiTot.compress(VectorOperation::add);
    if (!rhoIn){
      poissonMatrixPhiExt.compress(VectorOperation::add);      
      rhsPhiExt.compress(VectorOperation::add);
    }
    // Map the degree of freedom of the boudaries to the values they should have.
    std::map<unsigned int,double> boundary_valuesTot, boundary_valuesExt;
    VectorTools::interpolate_boundary_values (dof_handler, 0, BoundaryValuesFunction<dim>(true), boundary_valuesTot);
    VectorTools::interpolate_boundary_values (dof_handler, 0, BoundaryValuesFunction<dim>(false), boundary_valuesExt);
    // Apply boundary values to the related matrix and vectors.
    if (rhoIn){
      MatrixTools::apply_boundary_values (boundary_valuesTot, poissonMatrixPhiTot, phiTotRhoIn, rhsPhiTot, false);
    }
    else{
      MatrixTools::apply_boundary_values (boundary_valuesTot, poissonMatrixPhiTot, phiTotRhoOut, rhsPhiTot, false);
      MatrixTools::apply_boundary_values (boundary_valuesExt, poissonMatrixPhiExt, phiExtRhoOut, rhsPhiExt, false); //check if poissonMatrixPhiTot can be used twice in apply_boundary_condition
    }
    computing_timer.exit_section("E"); 
}

//Assemble eigen system
void poissonProblem::assemble_eigenSystem()
{
#ifdef spectral
   QGaussLobatto<dim> quadrature_formulaM(quadratureRule);
   QGauss<dim>  quadrature_formulaH(quadratureRule);
#else
   QGauss<dim>  quadrature_formulaM(quadratureRule-1);
   QGauss<dim>  quadrature_formulaH(quadratureRule-1);
#endif
   MatZeroEntries(ksHamiltonian_matrix);
   MatZeroEntries(ksMass_matrix);
   VecSet(massVector,0.0);
   FEValues<dim> fe_valuesM (fe, quadrature_formulaM, update_values | update_gradients | update_JxW_values);
   FEValues<dim> fe_valuesH (fe, quadrature_formulaH, update_values | update_gradients | update_JxW_values);
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formulaH.size();
   FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double> cell_mass_matrix (dofs_per_cell, dofs_per_cell);
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   std::vector<double> cellPhiTotal(n_q_points);  
   Vector<double>       cell_massVector (dofs_per_cell);

   DoFHandler<dim>::active_cell_iterator
     cell =dof_handler.begin_active(), 
     endc = dof_handler.end();
   Vector<double>  localsolution(phiTotRhoIn);
   unsigned int cellID=0;
   for (; cell!=endc; ++cell){
     if (cell->subdomain_id() == this_mpi_process){ 
       fe_valuesM.reinit (cell); fe_valuesH.reinit (cell);
       cell_matrix = 0;
       cell_mass_matrix=0;
       cell_massVector=0;
       fe_valuesH.get_function_values(localsolution, cellPhiTotal);
       //Get Exc
       std::vector<double> densityValue(n_q_points);
       std::vector<double> exchangePotentialVal(n_q_points);
       std::vector<double> corrPotentialVal(n_q_points);
       for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	 densityValue[q_point] = (*rhoInValues)(cellID, q_point);
       xc_lda_vxc(&funcX,n_q_points,&densityValue[0],&exchangePotentialVal[0]);
       xc_lda_vxc(&funcC,n_q_points,&densityValue[0],&corrPotentialVal[0]);
       
       for (unsigned int i=0; i<dofs_per_cell; ++i)
	 {
	   for (unsigned int j=0; j<dofs_per_cell; ++j)
	     {
	       for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		 {
		   cell_matrix(i,j) += (0.5*fe_valuesH.shape_grad (i, q_point)*fe_valuesH.shape_grad (j, q_point)+
					fe_valuesH.shape_value(i, q_point)*
					fe_valuesH.shape_value(j, q_point)*
					(cellPhiTotal[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point]))*fe_valuesH.JxW (q_point);
 		  cell_mass_matrix(i,j)+= (fe_valuesM.shape_value(i, q_point)*
 					   fe_valuesM.shape_value(j, q_point))*fe_valuesM.JxW (q_point);
		 }
	     }
	 }
       for (unsigned int i=0; i<dofs_per_cell; ++i)
	 {
	   for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	     cell_massVector(i)+=(fe_valuesM.shape_value(i, q_point)*fe_valuesM.shape_value(i, q_point))*fe_valuesM.JxW (q_point);
	   }
	 }
       cell->get_dof_indices (local_dof_indices);
       //assemble to global matrices
       meshConstraintsKS.distribute_local_to_global(cell_matrix,local_dof_indices, ksHamiltonian_matrix);
       meshConstraintsKS.distribute_local_to_global(cell_mass_matrix,local_dof_indices,ksMass_matrix);
       meshConstraintsKS.distribute_local_to_global(cell_massVector,local_dof_indices,massVector);
     }
     cellID++;
   }
   ksHamiltonian_matrix.compress (VectorOperation::add);
   ksMass_matrix.compress (VectorOperation::add);
   massVector.compress(VectorOperation::add);
   //multiple by M^(-1/2)
#ifdef standardEigenProblem
   VecSqrtAbs(massVector);
   VecReciprocal(massVector);
   MatDiagonalScale(ksHamiltonian_matrix,massVector,massVector);
#endif  
}


// Conjugate gradiant solver.
void poissonProblem::solve_cg (bool rhoIn=true)
{
  // Stopping criterion
  SolverControl solver_control1 (rhsPhiTot.size(), 1e-12*rhsPhiTot.l2_norm());
  PETScWrappers::SolverCG cg1 (solver_control1, mpi_communicator);
  //printf("solver rhs: %12.6e\n", rhsPhiTot.l2_norm());

  // Solver
  PETScWrappers::PreconditionJacobi precond1(poissonMatrixPhiTot);
  // Solve system with out preconditioning.
  if (rhoIn){
    PETScWrappers::MPI::Vector distributed_solution1(phiTotRhoIn);
    cg1.solve (poissonMatrixPhiTot, distributed_solution1, rhsPhiTot, precond1);
    meshConstraints.distribute (distributed_solution1);
    phiTotRhoIn = distributed_solution1;
  }
  else{
    PETScWrappers::MPI::Vector distributed_solution1(phiTotRhoOut);
    cg1.solve (poissonMatrixPhiTot, distributed_solution1, rhsPhiTot, precond1);
    meshConstraints.distribute (distributed_solution1);
    phiTotRhoOut = distributed_solution1;
    //Solve for Phi Ext
    SolverControl solver_control2 (rhsPhiExt.size(), 1e-12*rhsPhiExt.l2_norm());
    PETScWrappers::SolverCG cg2 (solver_control2, mpi_communicator);
    PETScWrappers::PreconditionJacobi precond2(poissonMatrixPhiExt);
    PETScWrappers::MPI::Vector distributed_solution2(phiExtRhoOut);
    cg2.solve (poissonMatrixPhiExt, distributed_solution2, rhsPhiExt, precond2);
    meshConstraints.distribute (distributed_solution2);
    phiExtRhoOut = distributed_solution2;
  }
  // Number of iterations needed to reach solution.
  //pcout<< "CG iterations with Jacobi preconditioner:" << solver_control.last_step() << std::endl;
  pcout<< "CG solve complete" << std::endl;
}

//Eigen solver
void poissonProblem::solve_eigenSystem()
{
  SolverControl solver_control (dof_handler.n_dofs(), 1.0e-7); 
  SLEPcWrappers::SolverJacobiDavidson eigensolver (solver_control,mpi_communicator);
  //SLEPcWrappers::SolverKrylovSchur eigensolver (solver_control,mpi_communicator);
  //SLEPcWrappers::SolverArnoldi  eigensolver (solver_control,mpi_communicator);
  eigensolver.set_which_eigenpairs (EPS_SMALLEST_REAL);
#ifdef standardEigenProblem
  eigensolver.solve(ksHamiltonian_matrix, kohnShamEigenValues, kohnShamEigenVectors, numEigenValues);
#else
  eigensolver.solve(ksHamiltonian_matrix, ksMass_matrix, kohnShamEigenValues, kohnShamEigenVectors, numEigenValues);
#endif
  //pcout << "Eigen Value is "<< kohnShamEigenValues[0] <<std::endl;
  for (unsigned int i=0; i<numEigenValues; i++)
    if (this_mpi_process == 0) std::printf("Eigen value %u : %30.20e \n", i, kohnShamEigenValues[i]);
}

// Output results to file.
void poissonProblem::output_results () const 
{
    const PETScWrappers::Vector localized_solution (kohnShamEigenVectors[2]);
    if (this_mpi_process == 0)
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (localized_solution, "solution");
        std::vector<unsigned int> partition_int (triangulation.n_active_cells());
        GridTools::get_subdomain_association (triangulation, partition_int);
        const Vector<double> partitioning(partition_int.begin(), partition_int.end());
        data_out.add_data_vector (partitioning, "partitioning");
        data_out.build_patches ();
        std::ofstream output ("solution.vtk");
        data_out.write_vtk (output);
    }
}

void poissonProblem::locate_origin()
{ 
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values);
  DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  //
  unsigned int vertices_per_cell=GeometryInfo<dim>::vertices_per_cell;
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      for (unsigned int i=0; i<vertices_per_cell; ++i)
	{
	  Point<dim> feNodeGlobalCoord = cell->vertex(i);
	  if (sqrt(feNodeGlobalCoord.square())<1.0e-10)
	    {  
	      originID=cell->vertex_dof_index(i,0);
	      pcout <<"Origin Global Node id is "<<cell->vertex(i) << " " << cell->vertex_dof_index(i,0) << std::endl;
	      return;
	    }
	}
    }
}

//Calculate rho out
void poissonProblem::computeRhoOut()
{
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
   
  std::vector<Vector<double> > localEigenVectors(numEigenValues);
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for (unsigned int i=0; i<numEigenValues; ++i){
    VecPointwiseMult(kohnShamEigenVectorsProjected[i],kohnShamEigenVectors[i], massVector);    
    localEigenVectors[i]=kohnShamEigenVectorsProjected[i]; //Check if this correct way to obtain local vector when running in parallel
  }
  
  //create new rhoValue tables
  rhoOutValues=new Table<2,double>(triangulation.n_active_cells(),std::pow(quadratureRule,dim));
  rhoOutVals.push_back(rhoOutValues);

  //loop over elements
  std::vector<double> rhoOut(n_q_points);
   DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      fe_values.reinit (cell); 
      std::vector<double> tempPsi(n_q_points);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	rhoOut[q_point]=0.0;
      }
      for (unsigned int i=0; i<numEigenValues; ++i){
	fe_values.get_function_values(localEigenVectors[i], tempPsi);
	for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	  double temp=(kohnShamEigenValues[i]-fermiEnergy)/(kb*TVal);
	  double partialOccupancy=1.0/(1.0+exp(temp)); 
	  rhoOut[q_point]+=2.0*partialOccupancy*std::pow(tempPsi[q_point],2.0); 
	}
      }
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	(*rhoOutValues)(cellID,q_point)=rhoOut[q_point];
      }
    }
    cellID++;
  }
}

//Implement simple mixing scheme 
double poissonProblem::mixing_simple()
{
  double normValue=0.0;
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   n_q_points    = quadrature_formula.size();
  double alpha=0.8;
  
  //create new rhoValue tables
  Table<2,double> *rhoInValuesOld=rhoInValues;
  rhoInValues=new Table<2,double>(triangulation.n_active_cells(),std::pow(quadratureRule,dim));
  rhoInVals.push_back(rhoInValues); 

  //loop over elements
  DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      fe_values.reinit (cell); 
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	//Compute (rhoIn-rhoOut)^2
        normValue+=std::pow((*rhoInValuesOld)(cellID,q_point)-(*rhoOutValues)(cellID,q_point),2.0)*fe_values.JxW(q_point);
	//Simple mixing scheme
	(*rhoInValues)(cellID,q_point)=std::abs((1-alpha)*(*rhoInValuesOld)(cellID,q_point)+ alpha*(*rhoOutValues)(cellID,q_point));
      }
    }
    cellID++;
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//Implement anderson mixing scheme 
double poissonProblem::mixing_anderson()
{
  double normValue=0.0;
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   n_q_points    = quadrature_formula.size();
  double alpha=0.5; 

  //initialize data structures
  int N=rhoOutVals.size()-1;
  pcout << "\nN:" << N << "\n";
  int NRHS=1, lda=N, ldb=N, info;
  int ipiv[N];
  double A[lda*N], c[ldb*NRHS]; 
  for (int i=0; i<lda*N; i++) A[i]=0.0;
  for (int i=0; i<ldb*NRHS; i++) c[i]=0.0;

  //loop over elements
  DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      fe_values.reinit (cell); 
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	//fill coefficient matrix, rhs
	double Fn=(*rhoOutVals[N])(cellID,q_point)-(*rhoInVals[N])(cellID,q_point);
	for (int m=0; m<N; m++){
	  double Fnm=(*rhoOutVals[N-1-m])(cellID,q_point)-(*rhoInVals[N-1-m])(cellID,q_point);
	  for (int k=0; k<N; k++){
	    double Fnk=(*rhoOutVals[N-1-k])(cellID,q_point)-(*rhoInVals[N-1-k])(cellID,q_point);
	    A[k*N+m] += (Fn-Fnm)*(Fn-Fnk)*fe_values.JxW(q_point); // (m,k)^th entry
	  }	  
	  c[m] += (Fn-Fnm)*(Fn)*fe_values.JxW(q_point); // (m)^th entry
	}
      }
    }
    cellID++;
  }
  std::cout << "A,c:" << A[0] << " " << c[0] << "\n";
  //solve for coefficients
  dgesv_(&N, &NRHS, A, &lda, ipiv, c, &ldb, &info);
  if((info > 0) && (this_mpi_process==0)) {
    printf( "Anderson Mixing: The diagonal element of the triangular factor of A,\n" );
    printf( "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n", info, info );
    exit(1);
  }
  double cn=1.0;
  for (int i=0; i<N; i++) cn-=c[i];
  if(this_mpi_process==0) {
    printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
    for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, c[N-1-i]);
    printf("\n");
  }

  //create new rhoValue tables
  Table<2,double> *rhoInValuesOld=rhoInValues;
  rhoInValues=new Table<2,double>(triangulation.n_active_cells(),std::pow(quadratureRule,dim));
  rhoInVals.push_back(rhoInValues);

  //implement anderson mixing
  cell = dof_handler.begin_active();
  cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      fe_values.reinit (cell); 
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	//Compute (rhoIn-rhoOut)^2
        normValue+=std::pow((*rhoInValuesOld)(cellID,q_point)-(*rhoOutValues)(cellID,q_point),2.0)*fe_values.JxW(q_point);
	//Anderson mixing scheme
	double rhoOutBar=cn*(*rhoOutVals[N])(cellID,q_point);
	double rhoInBar=cn*(*rhoInVals[N])(cellID,q_point);
	for (int i=0; i<N; i++){
	  rhoOutBar+=c[i]*(*rhoOutVals[N-1-i])(cellID,q_point);
	  rhoInBar+=c[i]*(*rhoInVals[N-1-i])(cellID,q_point);
	}
	(*rhoInValues)(cellID,q_point)=(1-alpha)*rhoInBar+alpha*rhoOutBar;
      }
    }
    cellID++;
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//Compute energies
void poissonProblem::compute_energy(){
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<double> cellPhiTotRhoIn(n_q_points);  
  std::vector<double> cellPhiTotRhoOut(n_q_points);  
  std::vector<double> cellPhiExtRhoOut(n_q_points);
  
  // Loop through all cells.
  DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  Vector<double>  localPhiTotRhoIn(phiTotRhoIn);
  Vector<double>  localPhiTotRhoOut(phiTotRhoOut);
  Vector<double>  localPhiExtRhoOut(phiExtRhoOut);
  //
  double bandEnergy=0.0;
  double partialOccupancy, temp;
  for (unsigned int i=0; i<numEigenValues; i++){
    temp=(kohnShamEigenValues[i]-fermiEnergy)/(kb*TVal);
    partialOccupancy=1.0/(1.0+exp(temp));
    bandEnergy+= 2*partialOccupancy*kohnShamEigenValues[i];
    if (this_mpi_process == 0) std::printf("partialOccupancy %u: %30.20e \n", i, partialOccupancy);
  }
  double kineticEnergy=0.0, exchangeEnergy=0.0, correlationEnergy=0.0, electrostaticEnergy=0.0; 
  unsigned int cellID=0;
  //
  for (; cell!=endc; ++cell) {
    if (cell->subdomain_id() == this_mpi_process){
      // Compute values for current cell.
      fe_values.reinit (cell);
      fe_values.get_function_values(localPhiTotRhoIn, cellPhiTotRhoIn);
      fe_values.get_function_values(localPhiTotRhoOut, cellPhiTotRhoOut);
      fe_values.get_function_values(localPhiExtRhoOut, cellPhiExtRhoOut);
      //Get Exc
      std::vector<double> densityValueIn(n_q_points), densityValueOut(n_q_points);
      std::vector<double> exchangeEnergyVal(n_q_points), corrEnergyVal(n_q_points);
      std::vector<double> exchangePotentialVal(n_q_points), corrPotentialVal(n_q_points);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	densityValueIn[q_point] = (*rhoInValues)(cellID, q_point);
	densityValueOut[q_point] = (*rhoOutValues)(cellID, q_point);
      }
      xc_lda_exc(&funcX,n_q_points,&densityValueOut[0],&exchangeEnergyVal[0]);
      xc_lda_exc(&funcC,n_q_points,&densityValueOut[0],&corrEnergyVal[0]);
      xc_lda_vxc(&funcX,n_q_points,&densityValueIn[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,n_q_points,&densityValueIn[0],&corrPotentialVal[0]);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	//Veff computed with rhoIn
	double Veff=cellPhiTotRhoIn[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point];
	//Vtot, Vext computet with rhoIn
	double Vtot=cellPhiTotRhoOut[q_point];
	double Vext=cellPhiExtRhoOut[q_point];
	kineticEnergy+=Veff*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	exchangeEnergy+=(exchangeEnergyVal[q_point])*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	correlationEnergy+=(corrEnergyVal[q_point])*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	electrostaticEnergy+=0.5*(Vtot+Vext)*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	//energy+= 0.0; //This should be the repulsion energy for multi-atom
      }
    }
    cellID++;
  } 
  kineticEnergy=bandEnergy-kineticEnergy;
  double energy=kineticEnergy+exchangeEnergy+correlationEnergy+electrostaticEnergy;
  double totalEnergy= Utilities::MPI::sum(energy, mpi_communicator);
  if (this_mpi_process == 0) std::printf("Total energy:%30.20e \n", totalEnergy);
  if (this_mpi_process == 0) std::printf("Band energy:%30.20e \nKinetic energy:%30.20e \nExchange energy:%30.20e \nCorrelation energy:%30.20e \nElectrostatic energy:%30.20e \n", bandEnergy, kineticEnergy, exchangeEnergy, correlationEnergy, electrostaticEnergy);
}

//compute fermi energy
void poissonProblem::compute_fermienergy(){
  //initial guess for fe
  double fe;
  if (numElectrons%2==0)
    fe = kohnShamEigenValues[numElectrons/2-1];
  else
    fe = kohnShamEigenValues[numElectrons/2];
  
  //compute residual
  double R=1.0;
  unsigned int iter=0;
  double temp1, temp2, temp3, temp4;
  while((std::abs(R)>1.0e-12) && (iter<100)){
    temp3=0.0; temp4=0.0;
    for (unsigned int i=0; i<numEigenValues; i++){
      temp1=(kohnShamEigenValues[i]-fe)/(kb*TVal);
      if (temp1<=0.0){
	temp2=1.0/(1.0+exp(temp1));
	temp3+=2.0*temp2;
	temp4+=2.0*(exp(temp1)/(kb*TVal))*temp2*temp2;
      }
      else{
	temp2=1.0/(1.0+exp(-temp1));
	temp3+=2.0*exp(-temp1)*temp2;
	temp4+=2.0*(exp(-temp1)/(kb*TVal))*temp2*temp2;       
      }
    }
    R=temp3-numElectrons;
    fe+=-R/temp4;
    iter++;
    //if (this_mpi_process == 0) std::printf("Fermi energy Residual %u:%30.20e \n", iter, std::abs(R));
  }
  if(std::abs(R)>1.0e-12){
    pcout << "Fermi Energy computation: Newton iterations failed to converge\n";
    exit(-1);
  }
  if (this_mpi_process == 0) std::printf("Fermi energy Residual:%30.20e \n", std::abs(R));

  //set Fermi energy
  fermiEnergy=fe;
  if (this_mpi_process == 0) std::printf("Fermi energy:%30.20e \n", fermiEnergy);
}

// Run the application with number of refinements passed to this method.
void poissonProblem::run ()
{
    generateMesh();
    setup_system();
    locate_origin();
    //compute phiExt
    assemble_system(); solve_cg();
    //Begin SCF iteration
    unsigned int scfIter=0;
    double norm=1.0;
    while ((norm>1.0e-13) && (scfIter<1)){
      if(this_mpi_process==0) printf("\n\nBegin SCF Iteration:%u\n", scfIter+1);
      //Mixing scheme
      if (scfIter>0){
	if (scfIter==1) norm=mixing_simple();
	else norm=mixing_anderson();
	if(this_mpi_process==0) printf("Mixing Scheme: iter:%u, norm:%12.6e\n", scfIter+1, norm);
      }
      //Poisson solve
      computing_timer.enter_section("Solve Poisson system (CG)");
      pcout<<"Solving poisson problem with rhoIn\n";
      assemble_system (); solve_cg ();
      computing_timer.exit_section("Solve Poisson system (CG)");
      //Eigen solve
      computing_timer.enter_section("Assemble eigen system (JacobiDavidson)");
      pcout<<"Solving eigen problem\n";
      assemble_eigenSystem();
      computing_timer.exit_section("Assemble eigen system (JacobiDavidson)");
      computing_timer.enter_section("Solve eigen system (JacobiDavidson)");
      solve_eigenSystem();
      computing_timer.exit_section("Solve eigen system (JacobiDavidson)");
      //Compute fermi energy
      compute_fermienergy();
      //Compute rho out
      computeRhoOut();
      //Poisson solve
      computing_timer.enter_section("Solve Poisson system (CG)");
      pcout<<"Solving poisson problem with rhoOut\n";
      assemble_system (false); solve_cg (false);
      computing_timer.exit_section("Solve Poisson system (CG)");
      //energy calculation
      compute_energy();
      pcout<<"SCF iteration: " << scfIter+1 << " complete\n";
      scfIter++;
      //break;
    }
     //output
    output_results ();
}

// Main method.
int main (int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
    deallog.depth_console(0);
    {
      poissonProblem poissonProblem;
      poissonProblem.run ();
    }
    return 0;
}
