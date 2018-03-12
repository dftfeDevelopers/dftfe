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
        }

    }//end of while loop

}//end of function



int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  //
  try
    {
      ParameterHandler prm;
      dftParameters::declare_parameters (prm);
      parse_command_line(argc,argv, prm);
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
  //
  int color1, color2;
  const int npool = dftParameters::npool;
  const int n_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  //double poolSizeFloat = (double)n_mpi_processes/(double)npool;
  //int poolSize = std::floor(poolSizeFloat);
  AssertThrow(n_mpi_processes % npool == 0,ExcMessage("Number of mpi processes must be a multiple of NUMBER OF POOLS"));
  const int poolSize= n_mpi_processes/npool;
  const int taskId = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ;
  MPI_Comm interpoolcomm, intrapoolcomm, mpi_comm_replica ;
  //
  if (taskId == 0)
    {
      std::cout<<"Number of pools: "<<npool<<std::endl;
      std::cout<<"Pool size: "<<poolSize<<std::endl;
    }

  color1 = taskId%poolSize ;


  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Comm_split(MPI_COMM_WORLD,
                 color1,
                 0,
                 &interpoolcomm);

  //

  color2 = taskId / poolSize ;

  MPI_Barrier(MPI_COMM_WORLD);


  MPI_Comm_split(MPI_COMM_WORLD,
                 color2,
                 0,
                 &intrapoolcomm);

  MPI_Comm_dup(intrapoolcomm , &mpi_comm_replica)  ;

  for (int i=0; i<n_mpi_processes; ++i)
    {
      if (taskId==i)
        std::cout << " My global id is " << taskId << " , pool id is " << Utilities::MPI::this_mpi_process(interpoolcomm)  <<
                  " , intrapool id is " << Utilities::MPI::this_mpi_process(intrapoolcomm) << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);

    }


  {
    //
    // set stdout precision
    //
    std::cout << std::scientific << std::setprecision(18);

    unsigned int finiteElementPolynomialOrder = dftParameters::finiteElementPolynomialOrder;
    unsigned int numberEigenValues = dftParameters::numberEigenValues;

    switch (finiteElementPolynomialOrder)
      {

      case 1:
      {
        dftClass<1> problemFEOrder1(mpi_comm_replica, interpoolcomm);
        problemFEOrder1.numEigenValues = numberEigenValues;
        problemFEOrder1.set();
        problemFEOrder1.init();
        problemFEOrder1.run();
        break;
      }

      case 2:
      {
        dftClass<2> problemFEOrder2(mpi_comm_replica, interpoolcomm);
        problemFEOrder2.numEigenValues = numberEigenValues;
        problemFEOrder2.set();
        problemFEOrder2.init();
        problemFEOrder2.run();
        break;
      }

      case 3:
      {
        dftClass<3> problemFEOrder3(mpi_comm_replica, interpoolcomm);
        problemFEOrder3.numEigenValues = numberEigenValues;
        problemFEOrder3.set();
        problemFEOrder3.init();
        problemFEOrder3.run();
        break;
      }

      case 4:
      {
        dftClass<4> problemFEOrder4(mpi_comm_replica, interpoolcomm);
        problemFEOrder4.numEigenValues = numberEigenValues;
        problemFEOrder4.set();
        problemFEOrder4.init();
        problemFEOrder4.run();
        break;
      }

      case 5:
      {
        dftClass<5> problemFEOrder5(mpi_comm_replica, interpoolcomm);
        problemFEOrder5.numEigenValues = numberEigenValues;
        problemFEOrder5.set();
        problemFEOrder5.init();
        problemFEOrder5.run();
        break;
      }

      case 6:
      {
        dftClass<6> problemFEOrder6(mpi_comm_replica, interpoolcomm);
        problemFEOrder6.numEigenValues = numberEigenValues;
        problemFEOrder6.set();
        problemFEOrder6.init();
        problemFEOrder6.run();
        break;
      }

      case 7:
      {
        dftClass<7> problemFEOrder7(mpi_comm_replica, interpoolcomm);
        problemFEOrder7.numEigenValues = numberEigenValues;
        problemFEOrder7.set();
        problemFEOrder7.init();
        problemFEOrder7.run();
        break;
      }

      case 8:
      {
        dftClass<8> problemFEOrder8(mpi_comm_replica, interpoolcomm);
        problemFEOrder8.numEigenValues = numberEigenValues;
        problemFEOrder8.set();
        problemFEOrder8.init();
        problemFEOrder8.run();
        break;
      }

      case 9:
      {
        dftClass<9> problemFEOrder9(mpi_comm_replica, interpoolcomm);
        problemFEOrder9.numEigenValues = numberEigenValues;
        problemFEOrder9.set();
        problemFEOrder9.init();
        problemFEOrder9.run();
        break;
      }

      case 10:
      {
        dftClass<10> problemFEOrder10(mpi_comm_replica, interpoolcomm);
        problemFEOrder10.numEigenValues = numberEigenValues;
        problemFEOrder10.set();
        problemFEOrder10.init();
        problemFEOrder10.run();
        break;
      }

      case 11:
      {
        dftClass<11> problemFEOrder11(mpi_comm_replica, interpoolcomm);
        problemFEOrder11.numEigenValues = numberEigenValues;
        problemFEOrder11.set();
        problemFEOrder11.init();
        problemFEOrder11.run();
        break;
      }

      case 12:
      {
        dftClass<12> problemFEOrder12(mpi_comm_replica, interpoolcomm);
        problemFEOrder12.numEigenValues = numberEigenValues;
        problemFEOrder12.set();
        problemFEOrder12.init();
        problemFEOrder12.run();
        break;
      }

      }

  }
  return 0;
}
