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

double radiusAtomBall, domainSizeX, domainSizeY, domainSizeZ;
unsigned int finiteElementPolynomialOrder;
unsigned int n_refinement_steps;
unsigned int numberEigenValues;

double lowerEndWantedSpectrum;
unsigned int chebyshevOrder; 
unsigned int numSCFIterations   = 1;
unsigned int maxLinearSolverIterations = 5000;
double relLinearSolverTolerance        = 1.0e-14; 
double selfConsistentSolverTolerance   = 1.0e-11;


bool isPseudopotential;
std::string meshFileName,coordinatesFile;

int xc_id;

//
//Define constants
//
const double TVal = 500.0;


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

  prm.declare_entry("MESH FILE", "",
		    Patterns::Anything(),
		    "Finite-element mesh file to be used for the given problem");

  prm.declare_entry("ATOMIC COORDINATES FILE", "",
		    Patterns::Anything(),
		    "File specifying the coordinates of the atoms in the given material system");

  prm.declare_entry("FINITE ELEMENT POLYNOMIAL ORDER", "2",
		    Patterns::Integer(1,12),
		    "The degree of the finite-element interpolating polynomial");


  prm.declare_entry("RADIUS ATOM BALL", "3.0",
		    Patterns::Double(),
		    "The radius of the ball around an atom on which self-potential of the associated nuclear charge is solved");

  prm.declare_entry("DOMAIN SIZE X", "20.0",
		    Patterns::Double(),
		    "Size of the domain in X-direction");

  prm.declare_entry("DOMAIN SIZE Y", "20.0",
		    Patterns::Double(),
		    "Size of the domain in Y-direction");

  prm.declare_entry("DOMAIN SIZE Z", "20.0",
		    Patterns::Double(),
		    "Size of the domain in Z-direction");
  

  prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "false",
		    Patterns::Bool(),
		    "Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed"); 

  prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
		    Patterns::Integer(1,4),
		    "Parameter specifying the type of exchange-correlation to be used");

  prm.declare_entry("NUMBER OF REFINEMENT STEPS", "4",
		    Patterns::Integer(1,4),
		    "Number of refinement steps to be used");

  prm.enter_subsection("CHEBYSHEV FILTERING OPTIONS");
  {
    prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
		      Patterns::Double(),
		      "The lower bound of the wanted eigen spectrum");

    prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "50",
		      Patterns::Integer(),
		      "The degree of the Chebyshev polynomial to be employed for filtering out the unwanted spectrum");

    prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "10",
		      Patterns::Integer(),
		      "Number of Kohn-Sham wavefunctions to be computed");
    
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

	  meshFileName                 = prm.get("MESH FILE");
	  coordinatesFile              = prm.get("ATOMIC COORDINATES FILE");
	  finiteElementPolynomialOrder = prm.get_integer("FINITE ELEMENT POLYNOMIAL ORDER");
	  radiusAtomBall               = prm.get_double("RADIUS ATOM BALL");
	  isPseudopotential            = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
	  xc_id                        = prm.get_integer("EXCHANGE CORRELATION TYPE");
	  n_refinement_steps           = prm.get_integer("NUMBER OF REFINEMENT STEPS");
	  domainSizeX                  = prm.get_double("DOMAIN SIZE X");
	  domainSizeY                  = prm.get_double("DOMAIN SIZE Y");
	  domainSizeZ                  = prm.get_double("DOMAIN SIZE Z");

	  prm.enter_subsection("CHEBYSHEV FILTERING OPTIONS");
	  {
	    numberEigenValues      = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
	    lowerEndWantedSpectrum = prm.get_double("LOWER BOUND WANTED SPECTRUM");
	    chebyshevOrder = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");        
	  }
	  prm.leave_subsection();
	}

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

    switch(finiteElementPolynomialOrder) {

    case 1:
      {
	dftClass<1> problemFEOrder1;
	problemFEOrder1.numEigenValues = numberEigenValues;
	problemFEOrder1.run();
	break;
      }

    case 2:
      {
	dftClass<2> problemFEOrder2;
	problemFEOrder2.numEigenValues = numberEigenValues;
	problemFEOrder2.run();
	break;
      }

    case 3:
      {
	dftClass<3> problemFEOrder3;
	problemFEOrder3.numEigenValues = numberEigenValues;
	problemFEOrder3.run();
	break;	
      }

    case 4:
      {
	dftClass<4> problemFEOrder4;
	problemFEOrder4.numEigenValues = numberEigenValues;
	problemFEOrder4.run();
	break;
      }

    case 5:
      {
	dftClass<5> problemFEOrder5;
	problemFEOrder5.numEigenValues = numberEigenValues;
	problemFEOrder5.run();
	break;
      }

    case 6:
      {
	dftClass<6> problemFEOrder6;
	problemFEOrder6.numEigenValues = numberEigenValues;
	problemFEOrder6.run();
	break;
      }

    case 7:
      {
	dftClass<7> problemFEOrder7;
	problemFEOrder7.numEigenValues = numberEigenValues;
	problemFEOrder7.run();
	break;
      }

    case 8:
      {
	dftClass<8> problemFEOrder8;
	problemFEOrder8.numEigenValues = numberEigenValues;
	problemFEOrder8.run();
	break;
      }

    case 9:
      {
	dftClass<9> problemFEOrder9;
	problemFEOrder9.numEigenValues = numberEigenValues;
	problemFEOrder9.run();
	break;
      }

    case 10:
      {
	dftClass<10> problemFEOrder10;
	problemFEOrder10.numEigenValues = numberEigenValues;
	problemFEOrder10.run();
	break;
      }

    case 11:
      {
	dftClass<11> problemFEOrder11;
	problemFEOrder11.numEigenValues = numberEigenValues;
	problemFEOrder11.run();
	break;
      }

    case 12:
      {
	dftClass<12> problemFEOrder12;
	problemFEOrder12.numEigenValues = numberEigenValues;
	problemFEOrder12.run();
	break;
      }

    }


  }
  return 0;
}

