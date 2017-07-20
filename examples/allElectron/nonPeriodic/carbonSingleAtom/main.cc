//
//deal.II header
//
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

//
//C++ headers
//
#include <list>
#include <iostream>
#include <fstream>

double radiusAtomBall;
unsigned int finiteElementPolynomialOrder = 4;
const unsigned int n_refinement_steps = 4;

double lowerEndWantedSpectrum;
unsigned int chebyshevOrder; 
unsigned int numSCFIterations   = 1;
unsigned int maxLinearSolverIterations = 5000;
double relLinearSolverTolerance        = 1.0e-14; 
double selfConsistentSolverTolerance   = 1.0e-11;


bool isPseudopotential;
std::string meshFileName;

//
//Define constants
//
const double TVal = 500.0;

//
//Mesh information
//
#define coordinatesFile "../../../../data/meshes/allElectron/carbonSingleAtom/coordinates.inp" 
#define xc_id 1


//
//dft header
//
#include "../../../../src/dft/dft.cc"




using namespace dealii;
ParameterHandler prm;

void
print_usage_message ()
{
  static const char *message
    =
    "Usage:\n"
    "    ./dftfe [-p parameter_file]\n"
    "Parameter sequences in brackets can be omitted if a parameter file is\n"
    "specified on the command line and if it provides values for these\n"
    "missing parameters.\n"
    "\n"
    "The parameter file has the following format and allows the following\n"
    "values (you can cut and paste this and use it for your own parameter\n"
    "file):\n"
    "\n";
  std::cout << message;
  prm.print_parameters (std::cout, ParameterHandler::Text);
}


void declare_parameters()
{
  prm.declare_entry("Radius Atom Ball", "3.0",
		    Patterns::Double(),
		    "The radius of the ball around an atom on which self-potential of the associated nuclear charge is solved");

  prm.declare_entry("Pseudopotential Calculation", "false",
		    Patterns::Bool(),
		    "Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed"); 

  prm.declare_entry("Mesh File Name", "",
		    Patterns::Anything(),
		    "Finite-element mesh file to be used for the given problem");

  prm.enter_subsection("Chebyshev filtering options");
  {
    prm.declare_entry("Lower Bound Wanted Spectrum", "-10.0",
		      Patterns::Double(),
		      "The lower bound of the wanted eigen spectrum");

    prm.declare_entry("Chebyshev Polynomial Degree", "50",
		      Patterns::Integer(),
		      "The degree of the Chebyshev polynomial to be employed for filtering out the unwanted spectrum");
  }
  prm.leave_subsection();
}

void parse_command_line(const int argc,
			char *const *argv)
{
  if(argc < 2)
    {
      print_usage_message();
      exit(1);
    }

  std::list<std::string> args;
  for (int i=1; i<argc; ++i)
    args.push_back (argv[i]);

  while(args.size())
    {
      if (args.front() == std::string("-p"))
	{
	  if (args.size() == 1)
	    {
	      std::cerr << "Error: flag '-p' must be followed by the "
			<< "name of a parameter file."
			<< std::endl;
	      print_usage_message ();
	      exit (1);
	    }
	  args.pop_front();
	  const std::string parameter_file = args.front();
	  args.pop_front();
	  prm.read_input(parameter_file);
	  print_usage_message();

	  radiusAtomBall = prm.get_double("Radius Atom Ball");
	  prm.enter_subsection("Chebyshev filtering options");
	  {
	    lowerEndWantedSpectrum = prm.get_double("Lower Bound Wanted Spectrum");
	    chebyshevOrder = prm.get_integer("Chebyshev Polynomial Degree");        
	  }
	  prm.leave_subsection();

	  isPseudopotential = prm.get_bool("Pseudopotential Calculation");

	  meshFileName = prm.get("Mesh File Name");

	}
      /*else
	{
	  input_file_names.push_back (args.front());
	  args.pop_front ();
	}

      if (input_file_names.size() == 0)
	{
	  std::cerr << "Error: No mesh file specified." << std::endl;
	  print_usage_message ();
	  exit (1);
	  }*/

    }//end of while loop

}//end of function



int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  try
    {
      declare_parameters ();
      parse_command_line(argc,argv);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
  deallog.depth_console(0);
  {
    //
    // set stdout precision
    //
    std::cout << std::scientific << std::setprecision(18);
    dftClass<finiteElementPolynomialOrder> problem;
    problem.numberAtomicWaveFunctions[6] = 5;
    problem.numEigenValues = 5;
    problem.run();
  }
  return 0;
}

