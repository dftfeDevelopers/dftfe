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
// @author Phani Motamarri (2017), Denis Davdov (2018)
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

int main (int argc, char *argv[])
{
  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1,
              ExcMessage("Usage:\n"
                         "mpirun -np nProcs executable parameterfile.prm\n"
                         "\n"));
  //
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  //
  int rank ;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  if (rank==0)
  {
     std::cout << "	" << std::endl ;
     std::cout << "	" << std::endl ;
     std::cout << "					Welcome to the Open Source program DFT-FE			" << std::endl;
     std::cout << "This is a C++ code for material modeling from first principles using Kohn-Sham density functional theory" << std::endl;
     std::cout << "				It is based on adaptive finite-element based methodologies.		" << std::endl ;
     std::cout << "For details and citing please refer: P.Motamarri et. al., Comp. Phys. Comm., Vol xx, Issue xx, pp xx, 2018" << std::endl ;
     std::cout << "	" << std::endl ;
     std::cout << "	" << std::endl ;
     std::cout << " 			Copyright (c) 2018 The Regents of the University of Michigan and DFT-FE authors " << std::endl ;
     std::cout << " 					DFT-FE is published under [LGPL v2.1 or newer] 			" << std::endl ;
     std::cout << "	" << std::endl ;
     std::cout << "	" << std::endl ;
  }
  //          
  ParameterHandler prm;
  dftfe::dftParameters::declare_parameters (prm);
  const std::string parameter_file = argv[1];
  prm.parse_input(parameter_file);
  dftfe::dftParameters::parse_parameters(prm);

  deallog.depth_console(0);

  dftfe::dftUtils::Pool pool(MPI_COMM_WORLD, dftfe::dftParameters::npool);

  // set stdout precision
  std::cout << std::scientific << std::setprecision(18);

  switch (dftfe::dftParameters::finiteElementPolynomialOrder)
    {

    case 1:
    {
      dftfe::dftClass<1> problemFEOrder1(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder1.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder1.set();
      problemFEOrder1.init();
      problemFEOrder1.run();
      break;
    }

    case 2:
    {
      dftfe::dftClass<2> problemFEOrder2(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder2.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder2.set();
      problemFEOrder2.init();
      problemFEOrder2.run();
      break;
    }

    case 3:
    {
      dftfe::dftClass<3> problemFEOrder3(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder3.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder3.set();
      problemFEOrder3.init();
      problemFEOrder3.run();
      break;
    }

    case 4:
    {
      dftfe::dftClass<4> problemFEOrder4(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder4.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder4.set();
      problemFEOrder4.init();
      problemFEOrder4.run();
      break;
    }

    case 5:
    {
      dftfe::dftClass<5> problemFEOrder5(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder5.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder5.set();
      problemFEOrder5.init();
      problemFEOrder5.run();
      break;
    }

    case 6:
    {
      dftfe::dftClass<6> problemFEOrder6(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder6.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder6.set();
      problemFEOrder6.init();
      problemFEOrder6.run();
      break;
    }

    case 7:
    {
      dftfe::dftClass<7> problemFEOrder7(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder7.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder7.set();
      problemFEOrder7.init();
      problemFEOrder7.run();
      break;
    }

    case 8:
    {
      dftfe::dftClass<8> problemFEOrder8(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder8.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder8.set();
      problemFEOrder8.init();
      problemFEOrder8.run();
      break;
    }

    case 9:
    {
      dftfe::dftClass<9> problemFEOrder9(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder9.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder9.set();
      problemFEOrder9.init();
      problemFEOrder9.run();
      break;
    }

    case 10:
    {
      dftfe::dftClass<10> problemFEOrder10(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder10.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder10.set();
      problemFEOrder10.init();
      problemFEOrder10.run();
      break;
    }

    case 11:
    {
      dftfe::dftClass<11> problemFEOrder11(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder11.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder11.set();
      problemFEOrder11.init();
      problemFEOrder11.run();
      break;
    }

    case 12:
    {
      dftfe::dftClass<12> problemFEOrder12(pool.get_replica_comm(), pool.get_interpool_comm());
      problemFEOrder12.numEigenValues = dftfe::dftParameters::numberEigenValues;
      problemFEOrder12.set();
      problemFEOrder12.init();
      problemFEOrder12.run();
      break;
    }

    }

  return 0;
}
