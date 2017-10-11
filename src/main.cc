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

unsigned int finiteElementPolynomialOrder,n_refinement_steps,numberEigenValues,xc_id, spinPolarized, nkx,nky,nkz;
unsigned int chebyshevOrder,numPass,numSCFIterations,maxLinearSolverIterations, mixingHistory;

double radiusAtomBall, domainSizeX, domainSizeY, domainSizeZ, mixingParameter;
double lowerEndWantedSpectrum,relLinearSolverTolerance,selfConsistentSolverTolerance,TVal, start_magnetization;

bool isPseudopotential,periodicX,periodicY,periodicZ, useSymm;
std::string meshFileName,coordinatesFile,currentPath,latticeVectorsFile,kPointDataFile, symmDataFile;

//
//dft header
//
#include "./dft/dft.cc"


using namespace dealii;
ParameterHandler prm;

void
print_usage_message ()
{
  static const char *message
    =
    "Usage:\n"
    "./dftRun parameterfile.prm (OR) mpirun -np nProcs ./dftRun parameterfile.prm\n"
    "\n";
  //parallel message stream
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
    {
      std::cout << message;
      prm.print_parameters (std::cout, ParameterHandler::Text);
    }
}


void declare_parameters()
{

  prm.declare_entry("OPTIMIZED MODE", "true",
		    Patterns::Bool(),
		    "Flag to control optimized/debug modes");

  prm.declare_entry("DFT PATH", "",
		    Patterns::Anything(),
		    "Path specifying the location of the build directory");

  prm.declare_entry("MESH FILE", "",
		    Patterns::Anything(),
		    "Finite-element mesh file to be used for the given problem");

  prm.declare_entry("ATOMIC COORDINATES FILE", "",
		    Patterns::Anything(),
		    "File specifying the coordinates of the atoms in the given material system");

  prm.declare_entry("LATTICE VECTORS FILE", "",
		    Patterns::Anything(),
		    "File specifying the lattice vectors associated with the unit-cell");

  prm.declare_entry("kPOINT RULE FILE", "",
		    Patterns::Anything(),
		    "File specifying the k-Point quadrature rule to sample Brillouin zone");
  prm.declare_entry("SYMMETRY MATRIX FILE", "",
		    Patterns::Anything(),
		    "File specifying the symmetry matrices for obtaining the irreducible BZ");
  prm.declare_entry("BZ SAMPLING POINTS ALONG X", "2",
		    Patterns::Integer(1,100),
		    "Number of Monkhorts-Pack grid points to be used along X direction for BZ sampling");
  prm.declare_entry("BZ SAMPLING POINTS ALONG Y", "2",
		    Patterns::Integer(1,100),
		    "Number of Monkhorts-Pack grid points to be used along Y direction for BZ sampling");
  prm.declare_entry("BZ SAMPLING POINTS ALONG Z", "2",
		    Patterns::Integer(1,100),
		    "Number of Monkhorts-Pack grid points to be used along Z direction for BZ sampling");
  prm.declare_entry("USE GROUP SYMMETRY", "true",
		    Patterns::Bool(),
		    "Flag to control usage of space group symmetries (only for periodic calculation)");

  prm.declare_entry("FINITE ELEMENT POLYNOMIAL ORDER", "2",
		    Patterns::Integer(1,12),
		    "The degree of the finite-element interpolating polynomial");


  prm.declare_entry("RADIUS ATOM BALL", "3.0",
		    Patterns::Double(),
		    "The radius of the ball around an atom on which self-potential of the associated nuclear charge is solved");

  prm.declare_entry("DOMAIN SIZE X", "0.0",
		    Patterns::Double(),
		    "Size of the domain in X-direction");

  prm.declare_entry("DOMAIN SIZE Y", "0.0",
		    Patterns::Double(),
		    "Size of the domain in Y-direction");

  prm.declare_entry("DOMAIN SIZE Z", "0.0",
		    Patterns::Double(),
		    "Size of the domain in Z-direction");

  prm.declare_entry("SPIN POLARIZATION", "0",
		    Patterns::Integer(0,1),
		    "Is spin polarization to be included?");

  prm.declare_entry("START MAGNETIZATION", "0.0",
		     Patterns::Double(),
		    "Magnetization to start with");
  
  prm.declare_entry("PERIODIC BOUNDARY CONDITION X", "false",
		    Patterns::Bool(),
		    "Periodicity in X-direction");

  prm.declare_entry("PERIODIC BOUNDARY CONDITION Y", "false",
		    Patterns::Bool(),
		    "Periodicity in Y-direction");

  prm.declare_entry("PERIODIC BOUNDARY CONDITION Z", "false",
		    Patterns::Bool(),
		    "Periodicity in Z-direction");

  prm.declare_entry("PSEUDOPOTENTIAL CALCULATION", "false",
		    Patterns::Bool(),
		    "Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed"); 

  prm.declare_entry("EXCHANGE CORRELATION TYPE", "1",
		    Patterns::Integer(1,4),
		    "Parameter specifying the type of exchange-correlation to be used");

  prm.declare_entry("NUMBER OF REFINEMENT STEPS", "4",
		    Patterns::Integer(1,4),
		    "Number of refinement steps to be used");

  prm.declare_entry("LOWER BOUND WANTED SPECTRUM", "-10.0",
		    Patterns::Double(),
		    "The lower bound of the wanted eigen spectrum");
  
  prm.declare_entry("CHEBYSHEV POLYNOMIAL DEGREE", "0",
		    Patterns::Integer(),
		    "The degree of the Chebyshev polynomial to be employed for filtering out the unwanted spectrum (Default value is used when the input parameter value is 0");

  prm.declare_entry("CHEBYSHEV FILTER PASSES", "1",
		    Patterns::Integer(),
		    "The number of the Chebyshev filter passes per SCF  (Default value is used when the input parameter is not specified");

  prm.declare_entry("NUMBER OF KOHN-SHAM WAVEFUNCTIONS", "10",
		    Patterns::Integer(),
		    "Number of Kohn-Sham wavefunctions to be computed");

  prm.declare_entry("TEMPERATURE", "500.0",
		    Patterns::Double(),
		    "Fermi-Dirac smearing temperature");

  prm.declare_entry("SCF CONVERGENCE MAXIMUM ITERATIONS", "50",
		    Patterns::Integer(),
		    "Maximum number of iterations to be allowed for SCF convergence");
  
  prm.declare_entry("SCF CONVERGENCE TOLERANCE", "1e-08",
		    Patterns::Double(),
		    "SCF iterations stopping tolerance in terms of electron-density difference between two successive iterations");

  prm.declare_entry("ANDERSON SCHEME MIXING HISTORY", "70",
		    Patterns::Integer(),
		    "Number of SCF iterations to be considered for mixing the electron-density");

  prm.declare_entry("ANDERSON SCHEME MIXING PARAMETER", "0.5",
		    Patterns::Double(0.0,1.0),
		    "Mixing parameter to be used in Anderson scheme");

  prm.declare_entry("POISSON SOLVER CONVERGENCE MAXIMUM ITERATIONS", "5000",
		    Patterns::Integer(),
		    "Maximum number of iterations to be allowed for Poisson problem convergence");
  
  prm.declare_entry("POISSON SOLVER CONVERGENCE TOLERANCE", "1e-12",
		    Patterns::Double(),
		    "Relative tolerance as stopping criterion for Poisson problem convergence");

    

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

	  currentPath                   = prm.get("DFT PATH");
	  currentPath.erase(std::remove(currentPath.begin(),currentPath.end(),'"'),currentPath.end());
	  meshFileName                  = prm.get("MESH FILE");
	  finiteElementPolynomialOrder  = prm.get_integer("FINITE ELEMENT POLYNOMIAL ORDER");
	  n_refinement_steps            = prm.get_integer("NUMBER OF REFINEMENT STEPS");
	  coordinatesFile               = prm.get("ATOMIC COORDINATES FILE");
	  radiusAtomBall                = prm.get_double("RADIUS ATOM BALL");
	  domainSizeX                   = prm.get_double("DOMAIN SIZE X");
	  domainSizeY                   = prm.get_double("DOMAIN SIZE Y");
	  domainSizeZ                   = prm.get_double("DOMAIN SIZE Z");
	  spinPolarized                 = prm.get_integer("SPIN POLARIZATION");
	  start_magnetization           = prm.get_double("START MAGNETIZATION");
	  periodicX                     = prm.get_bool("PERIODIC BOUNDARY CONDITION X");
	  periodicY                     = prm.get_bool("PERIODIC BOUNDARY CONDITION Y");
	  periodicZ                     = prm.get_bool("PERIODIC BOUNDARY CONDITION Z");
	  latticeVectorsFile            = prm.get("LATTICE VECTORS FILE");
	  kPointDataFile                = prm.get("kPOINT RULE FILE");
          symmDataFile                  = prm.get("SYMMETRY MATRIX FILE");
          nkx				= prm.get_integer("BZ SAMPLING POINTS ALONG X");
	  nky				= prm.get_integer("BZ SAMPLING POINTS ALONG Y");
          nkz				= prm.get_integer("BZ SAMPLING POINTS ALONG Z");
          useSymm 	                = prm.get_bool("USE GROUP SYMMETRY");
	  isPseudopotential             = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
	  xc_id                         = prm.get_integer("EXCHANGE CORRELATION TYPE");
	  numberEigenValues             = prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
	  lowerEndWantedSpectrum        = prm.get_double("LOWER BOUND WANTED SPECTRUM");
	  chebyshevOrder                = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");  
	  numPass			= prm.get_integer("CHEBYSHEV FILTER PASSES");
	  numSCFIterations              = prm.get_integer("SCF CONVERGENCE MAXIMUM ITERATIONS");
	  selfConsistentSolverTolerance = prm.get_double("SCF CONVERGENCE TOLERANCE");
	  mixingHistory                 = prm.get_integer("ANDERSON SCHEME MIXING HISTORY");
	  mixingParameter               = prm.get_double("ANDERSON SCHEME MIXING PARAMETER");
	  TVal                          = prm.get_double("TEMPERATURE");	
	  maxLinearSolverIterations     = prm.get_integer("POISSON SOLVER CONVERGENCE MAXIMUM ITERATIONS");
	  relLinearSolverTolerance      = prm.get_double("POISSON SOLVER CONVERGENCE TOLERANCE");
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

