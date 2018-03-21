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
// @author Phani Motamarri (2017)
//

//
//deal.II header
//
#include <deal.II/base/data_out_base.h>

//
//dft header
//
#include "constants.h"
#include "dft.h"
#include <dftUtils.h>


//
//C++ headers
//
#include <list>
#include <iostream>
#include <fstream>


using namespace dealii;

void
print_usage_message (ParameterHandler &prm)
{
  static const char *message
    =
      "Usage:\n"
      "./dftRun parameterfile.prm (OR) mpirun -np nProcs ./dftRun parameterfile.prm\n"
      "\n";
  //parallel message stream
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)== 0)
    {
      std::cout << message;
      prm.print_parameters (std::cout, ParameterHandler::Text);
    }
}


void parse_command_line(const int argc,
                        char *const *argv,
                        ParameterHandler &prm)
{
  if (argc < 2)
    {
      print_usage_message(prm);
      exit(1);
    }

  std::list<std::string> args;
  for (int i=1; i<argc; ++i)
    args.push_back (argv[i]);

  while (args.size())
    {
      if (args.front() == std::string("-p"))
        {
          if (args.size() == 1)
            {
              std::cerr << "Error: flag '-p' must be followed by the "
                        << "name of a parameter file."
                        << std::endl;
              print_usage_message (prm);
              exit (1);
            }
          args.pop_front();
          const std::string parameter_file = args.front();
          args.pop_front();
          prm.parse_input(parameter_file);
          print_usage_message(prm);
          dftParameters::parse_parameters(prm);

	  const bool printParametersToFile=false;
	  if (printParametersToFile)
	  {
	    std::ofstream output ("demoParameterFile.prm");
	    prm.print_parameters (output, ParameterHandler::OutputStyle::Text);
	  }
        }

    }//end of while loop

}//end of function



int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  ParameterHandler prm;
  dftParameters::declare_parameters (prm);
  parse_command_line(argc,argv, prm);

  deallog.depth_console(0);

  dftUtils::Pool pool(MPI_COMM_WORLD, dftParameters::npool);

  // set stdout precision
  std::cout << std::scientific << std::setprecision(18);

  switch (dftParameters::finiteElementPolynomialOrder)
    {

    case 1:
    {
      dftClass<1> problemFEOrder1(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder1.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder1.set();
      problemFEOrder1.init();
      problemFEOrder1.run();
      break;
    }

    case 2:
    {
      dftClass<2> problemFEOrder2(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder2.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder2.set();
      problemFEOrder2.init();
      problemFEOrder2.run();
      break;
    }

    case 3:
    {
      dftClass<3> problemFEOrder3(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder3.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder3.set();
      problemFEOrder3.init();
      problemFEOrder3.run();
      break;
    }

    case 4:
    {
      dftClass<4> problemFEOrder4(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder4.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder4.set();
      problemFEOrder4.init();
      problemFEOrder4.run();
      break;
    }

    case 5:
    {
      dftClass<5> problemFEOrder5(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder5.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder5.set();
      problemFEOrder5.init();
      problemFEOrder5.run();
      break;
    }

    case 6:
    {
      dftClass<6> problemFEOrder6(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder6.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder6.set();
      problemFEOrder6.init();
      problemFEOrder6.run();
      break;
    }

    case 7:
    {
      dftClass<7> problemFEOrder7(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder7.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder7.set();
      problemFEOrder7.init();
      problemFEOrder7.run();
      break;
    }

    case 8:
    {
      dftClass<8> problemFEOrder8(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder8.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder8.set();
      problemFEOrder8.init();
      problemFEOrder8.run();
      break;
    }

    case 9:
    {
      dftClass<9> problemFEOrder9(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder9.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder9.set();
      problemFEOrder9.init();
      problemFEOrder9.run();
      break;
    }

    case 10:
    {
      dftClass<10> problemFEOrder10(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder10.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder10.set();
      problemFEOrder10.init();
      problemFEOrder10.run();
      break;
    }

    case 11:
    {
      dftClass<11> problemFEOrder11(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder11.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder11.set();
      problemFEOrder11.init();
      problemFEOrder11.run();
      break;
    }

    case 12:
    {
      dftClass<12> problemFEOrder12(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder12.numEigenValues = dftParameters::numberEigenValues;
      problemFEOrder12.set();
      problemFEOrder12.init();
      problemFEOrder12.run();
      break;
    }

    }

  return 0;
}
