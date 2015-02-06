//Include header files
#include "../include/headers.h"
#include "../include/dft.h"
#include "poisson.cc"
//#include "eigen.cc"
#include "../utils/fileReaders.cc"

//dft constructor
dft::dft():
  triangulation (MPI_COMM_WORLD,
		 typename Triangulation<dim>::MeshSmoothing
		 (Triangulation<dim>::smoothing_on_refinement |
		  Triangulation<dim>::smoothing_on_coarsening)),
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
  poissonObject(&dofHandler),
  denSpline(numAtomTypes)
{}

void dft::locateAtomCoreNodes(){ 
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (FE, quadrature_formula, update_values);
  //
  unsigned int vertices_per_cell=GeometryInfo<dim>::vertices_per_cell;
  DoFHandler<dim>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int i=0; i<vertices_per_cell; ++i){
	Point<dim> feNodeGlobalCoord = cell->vertex(i);
	if (sqrt(feNodeGlobalCoord.square())<1.0e-12){  
	  originIDs[0]=cell->vertex_dof_index(i,0);
	  std::cout << "Atom core located at ("<< cell->vertex(i) << ") with node id " << cell->vertex_dof_index(i,0) << " in processor " << this_mpi_process << std::endl;
	  MPI_Bcast (originIDs, numAtomTypes, MPI_UNSIGNED, this_mpi_process, mpi_communicator);
	  return;
	}
      }
    }
  }
}

//Compute total charge
double dft::totalCharge(){
  double normValue=0.0;
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (FE, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  
  DoFHandler<dim>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
        normValue+=(*rhoInVals[0])(cellID,q_point)*fe_values.JxW(q_point);
      }
    cellID++;
    }
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}

//initialize rho
void dft::initRho(){
  //Initialize electron density table storage
  //std::cout << triangulation.n_locally_owned_active_cells() << std::endl; 
  Table<2,double> *rhoInValues=new Table<2,double>(triangulation.n_locally_owned_active_cells(),std::pow(quadratureRule,dim));
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
  
  //interpolate rho
  alglib::real_1d_array x;
  x.setcontent(numRows,xData);
  alglib::real_1d_array y;
  y.setcontent(numRows,yData);
  alglib::ae_int_t natural_bound_type = 1;
  spline1dbuildcubic(x, y, numRows, natural_bound_type, 0.0, natural_bound_type, 0.0, denSpline[0]);

  std::vector<double> outerMostPointDen(numAtomTypes);
  outerMostPointDen[0]= xData[numRows-1];
  //Initialize rho
  QGauss<dim>  quadrature_formula(quadratureRule);
  FEValues<dim> fe_values (FE, quadrature_formula, update_values);
  const unsigned int n_q_points    = quadrature_formula.size();
  typename DoFHandler<dim>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
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
  }
  //Normalize rho
  double charge=totalCharge();
  pcout << "initial charge: " << charge << std::endl;
  cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)(cellID,q)*=1.0/charge;
      }
      cellID++;
    }
  }
  
  //Initialize libxc
  int exceptParamX, exceptParamC;
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
  if(exceptParamX != 0 || exceptParamC != 0){
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"Exchange or Correlation Functional not found"<<std::endl;
    pcout<<"-------------------------------------"<<std::endl;
  }
  else{
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	 << std::endl<< funcX.info->refs << std::endl;
    pcout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	 << std::endl<< funcX.info->refs << std::endl;
    pcout<<"-------------------------------------"<<std::endl;	  
  }
  
}

void dft::init(){
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;
  
  //initialize FE objects
  dofHandler.distribute_dofs (FE);
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;
  
  //initialize poisson object
  poissonObject.init();
}

//Generate triangulation.
void dft::mesh(){
  //GridGenerator::hyper_cube (triangulation, -1, 1);
  //triangulation.refine_global (6);
  //Read mesh written out in UCD format
  static const Point<3> center = Point<3>();
  static const HyperBallBoundary<3, 3> boundary(center,radius);
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);
  //Read mesh in UCD format generated from Cubit
  std::ifstream f("meshFiles/ucd.inp");
  gridin.read_ucd(f);
  triangulation.set_boundary(0, boundary);
  triangulation.refine_global (n_refinement_steps);
  /*pcout << "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl;
  // Output the total number of cells.
  pcout << "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl;
  */
}

void dft::run ()
{
  //generate/read mesh
  mesh();
  //initialize
  init();
  initRho();
  locateAtomCoreNodes();

  //initialize poisson, eigen objects
  poissonObject.solve();
  //eigenProblem eigen(this);
  //Initialize mesh, setup, locate origin.
}
