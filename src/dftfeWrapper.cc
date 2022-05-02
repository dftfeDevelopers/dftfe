// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
// @author Sambit Das
//


// deal.II header
//
#include <deal.II/base/data_out_base.h>
#include <p4est_bits.h>

#ifdef USE_PETSC
#  include <petscsys.h>
#  include <slepcsys.h>
#endif

//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <sys/stat.h>

#include "dft.h"
#include "dftParameters.h"
#include "cudaHelpers.h"
#include "dftUtils.h"
#include "dftfeWrapper.h"


namespace dftfe
{
  namespace
  {
    template <int n1, int n2>
    void
    create_dftfe(const MPI_Comm &      mpi_comm_parent,
                 const MPI_Comm &      mpi_comm_domain,
                 const MPI_Comm &      interpoolcomm,
                 const MPI_Comm &      interBandGroupComm,
                 dftfe::dftParameters &dftParams,
                 dftBase **            dftfeBaseDoublePtr)
    {
      *dftfeBaseDoublePtr = new dftfe::dftClass<n1, n2>(mpi_comm_parent,
                                                        mpi_comm_domain,
                                                        interpoolcomm,
                                                        interBandGroupComm,
                                                        dftParams);
    }

    // Dynamically create dftClass<n> objects by order.
    //  Note that we can't store a list of classes because the types differ,
    //  but we can store a list of functions that use them in an n-independent
    //  way.
    //
    //  Also note element 0 is order 1.
    //
    typedef void (*create_fn)(const MPI_Comm &      mpi_comm_parent,
                              const MPI_Comm &      mpi_comm_domain,
                              const MPI_Comm &      interpoolcomm,
                              const MPI_Comm &      interBandGroupComm,
                              dftfe::dftParameters &dftParams,
                              dftBase **            dftBaseDoublePtr);

    static create_fn order_list[] = {
#ifdef DFTFE_MINIMAL_COMPILE
      create_dftfe<2, 2>,
      create_dftfe<3, 3>,
      create_dftfe<4, 4>,
      create_dftfe<5, 5>,
      create_dftfe<6, 6>,
      create_dftfe<6, 7>,
      create_dftfe<6, 8>,
      create_dftfe<6, 9>,
      create_dftfe<7, 7>
#else
      create_dftfe<1, 1>,  create_dftfe<1, 2>,  create_dftfe<2, 2>,
      create_dftfe<2, 3>,  create_dftfe<2, 4>,  create_dftfe<3, 3>,
      create_dftfe<3, 4>,  create_dftfe<3, 5>,  create_dftfe<3, 6>,
      create_dftfe<4, 4>,  create_dftfe<4, 5>,  create_dftfe<4, 6>,
      create_dftfe<4, 7>,  create_dftfe<4, 8>,  create_dftfe<5, 5>,
      create_dftfe<5, 6>,  create_dftfe<5, 7>,  create_dftfe<5, 8>,
      create_dftfe<5, 9>,  create_dftfe<5, 10>, create_dftfe<6, 6>,
      create_dftfe<6, 7>,  create_dftfe<6, 8>,  create_dftfe<6, 9>,
      create_dftfe<6, 10>, create_dftfe<6, 11>, create_dftfe<6, 12>,
      create_dftfe<7, 7>,  create_dftfe<7, 8>,  create_dftfe<7, 9>,
      create_dftfe<7, 10>, create_dftfe<7, 11>, create_dftfe<7, 12>,
      create_dftfe<7, 13>, create_dftfe<7, 14>, create_dftfe<8, 8>,
      create_dftfe<8, 9>,  create_dftfe<8, 10>, create_dftfe<8, 11>,
      create_dftfe<8, 12>, create_dftfe<8, 13>, create_dftfe<8, 14>,
      create_dftfe<8, 15>, create_dftfe<8, 16>
#endif
    };
  } // namespace

  void
  dftfeWrapper::globalHandlesInitialize()
  {
    sc_init(MPI_COMM_WORLD, 0, 0, nullptr, SC_LP_SILENT);
    p4est_init(nullptr, SC_LP_SILENT);

#ifdef USE_PETSC
    SlepcInitializeNoArguments();
    PetscPopSignalHandler();
#endif

    if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
      {
        fprintf(stderr, "Error: ELPA API version not supported.");
        exit(1);
      }
  }

  void
  dftfeWrapper::globalHandlesFinalize()
  {
    sc_finalize();

#ifdef USE_PETSC
    SlepcFinalize();
#endif

    int error;
    elpa_uninit(&error);
    AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));
  }

  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const MPI_Comm &  mpi_comm_parent,
                             const bool        printParams,
                             const bool        setGPUToMPITaskBindingInternally)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
  {
    reinit(parameter_file,
           mpi_comm_parent,
           printParams,
           setGPUToMPITaskBindingInternally);
  }

  dftfeWrapper::~dftfeWrapper()
  {
    clear();
  }

  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const MPI_Comm &  mpi_comm_parent,
                       const bool        printParams,
                       const bool        setGPUToMPITaskBindingInternally)
  {
    clear();
    d_mpi_comm_parent = mpi_comm_parent;
    if (mpi_comm_parent != MPI_COMM_NULL)
      {
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file,
                                           mpi_comm_parent,
                                           printParams);

#ifdef DFTFE_WITH_GPU
        if (d_dftfeParamsPtr->useGPU && setGPUToMPITaskBindingInternally)
          dftfe::cudaUtils::setupGPU();
#endif

        dftfe::dftUtils::Pool kPointPool(mpi_comm_parent,
                                         d_dftfeParamsPtr->npool,
                                         d_dftfeParamsPtr->verbosity);
        dftfe::dftUtils::Pool bandGroupsPool(kPointPool.get_intrapool_comm(),
                                             d_dftfeParamsPtr->nbandGrps,
                                             d_dftfeParamsPtr->verbosity);

        std::srand(dealii::Utilities::MPI::this_mpi_process(
          bandGroupsPool.get_intrapool_comm()));

        if (d_dftfeParamsPtr->verbosity >= 1)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0));
            pcout
              << "=================================MPI Parallelization========================================="
              << std::endl;
            pcout << "Total number of MPI tasks: "
                  << Utilities::MPI::n_mpi_processes(mpi_comm_parent)
                  << std::endl;
            pcout << "k-point parallelization processor groups: "
                  << Utilities::MPI::n_mpi_processes(
                       kPointPool.get_interpool_comm())
                  << std::endl;
            pcout << "Band parallelization processor groups: "
                  << Utilities::MPI::n_mpi_processes(
                       bandGroupsPool.get_interpool_comm())
                  << std::endl;
            pcout
              << "Number of MPI tasks for finite-element domain decomposition: "
              << Utilities::MPI::n_mpi_processes(
                   bandGroupsPool.get_intrapool_comm())
              << std::endl;
            pcout
              << "============================================================================================"
              << std::endl;
          }


        // set stdout precision
        std::cout << std::scientific << std::setprecision(18);

        int order = d_dftfeParamsPtr->finiteElementPolynomialOrder;
        int orderElectro =
          d_dftfeParamsPtr->finiteElementPolynomialOrderElectrostatics;

#ifdef DFTFE_MINIMAL_COMPILE
        if (order < 2 || order > 7)
          {
            std::cout << "Invalid DFT-FE order " << order << std::endl;
            exit(1);
          }

        if (order > 5 && order < 7)
          {
            if (orderElectro < order || orderElectro > (order + 3))
              {
                std::cout << "Invalid DFT-FE order electrostatics "
                          << orderElectro << std::endl;
                exit(1);
              }
          }
        else
          {
            if (orderElectro != order)
              {
                std::cout << "Invalid DFT-FE order electrostatics "
                          << orderElectro << std::endl;
                exit(1);
              }
          }

        int listIndex = 0;
        for (int i = 2; i <= order; i++)
          {
            int maxElectroOrder = (i < order) ? (i + 3) : orderElectro;
            if (i != 6)
              maxElectroOrder = i;
            for (int j = i; j <= maxElectroOrder; j++)
              listIndex++;
          }
#else
        if (order < 1 || order > 8)
          {
            std::cout << "Invalid DFT-FE order " << order << std::endl;
            exit(1);
          }

        if (orderElectro < order || orderElectro > order * 2)
          {
            std::cout << "Invalid DFT-FE order electrostatics " << orderElectro
                      << std::endl;
            exit(1);
          }


        int listIndex = 0;
        for (int i = 1; i <= order; i++)
          {
            int maxElectroOrder = (i < order) ? 2 * i : orderElectro;
            for (int j = i; j <= maxElectroOrder; j++)
              listIndex++;
          }
#endif
        create_fn create = order_list[listIndex - 1];
        create(mpi_comm_parent,
               bandGroupsPool.get_intrapool_comm(),
               kPointPool.get_interpool_comm(),
               bandGroupsPool.get_interpool_comm(),
               *d_dftfeParamsPtr,
               &d_dftfeBasePtr);
        d_dftfeBasePtr->set();
        d_dftfeBasePtr->init();
      }
  }

  void
  dftfeWrapper::clear()
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        if (d_dftfeBasePtr != nullptr)
          delete d_dftfeBasePtr;
        if (d_dftfeParamsPtr != nullptr)
          delete d_dftfeParamsPtr;
      }
  }

  void
  dftfeWrapper::run()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->run();
  }

  double
  dftfeWrapper::computeDFTFreeEnergy()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->solve(true, true, false);
    return d_dftfeBasePtr->getFreeEnergy();
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getForcesAtoms() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> ionicForces(
      d_dftfeBasePtr->getForceonAtoms().size() / 3,
      std::vector<double>(3, 0.0));

    std::vector<double> ionicForcesVec = d_dftfeBasePtr->getForceonAtoms();
    for (unsigned int i = 0; i < ionicForces.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        ionicForces[i][j] = -ionicForcesVec[3 * i + j];
    return ionicForces;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getCellStress() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> cellStress(3, std::vector<double>(3, 0.0));
    dealii::Tensor<2, 3, double>     cellStressTensor =
      d_dftfeBasePtr->getCellStress();

    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        cellStress[i][j] = -cellStressTensor[i][j];
    return cellStress;
  }

  void
  dftfeWrapper::updateAtomPositions(
    const std::vector<std::vector<double>> atomsDisplacements)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    AssertThrow(
      atomsDisplacements.size() ==
        d_dftfeBasePtr->getAtomLocationsCart().size(),
      dealii::ExcMessage(
        "DFT-FE error: Incorrect size of atomsDisplacements vector."));
    std::vector<Tensor<1, 3, double>> dispVec(atomsDisplacements.size());
    for (unsigned int i = 0; i < dispVec.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        dispVec[i][j] = atomsDisplacements[i][j];
    d_dftfeBasePtr->updateAtomPositionsAndMoveMesh(dispVec);
  }

  void
  dftfeWrapper::deformDomain(
    const std::vector<std::vector<double>> deformationGradient)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    dealii::Tensor<2, 3, double> defGradTensor;
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        defGradTensor[i][j] = deformationGradient[i][j];
    d_dftfeBasePtr->deformDomain(defGradTensor);
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getAtomLocationsCart() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<std::vector<double>> atomLocationsCart(
      d_dftfeBasePtr->getAtomLocationsCart().size(),
      std::vector<double>(3, 0.0));
    for (unsigned int i = 0; i < atomLocationsCart.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        atomLocationsCart[i][j] = temp[i][j + 2];
    return atomLocationsCart;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getAtomLocationsFrac() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsFrac();
    std::vector<std::vector<double>> atomLocationsFrac(
      d_dftfeBasePtr->getAtomLocationsFrac().size(),
      std::vector<double>(3, 0.0));
    for (unsigned int i = 0; i < atomLocationsFrac.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        atomLocationsFrac[i][j] = temp[i][j + 2];
    return atomLocationsFrac;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getCell() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getCell();
  }

  std::vector<bool>
  dftfeWrapper::getPBC() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<bool> pbc(3, false);
    pbc[0] = d_dftfeParamsPtr->periodicX;
    pbc[1] = d_dftfeParamsPtr->periodicY;
    pbc[2] = d_dftfeParamsPtr->periodicZ;
    return pbc;
  }

  std::vector<int>
  dftfeWrapper::getAtomicNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<int> atomicNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (unsigned int i = 0; i < atomicNumbers.size(); ++i)
      atomicNumbers[i] = temp[i][0];
    return atomicNumbers;
  }


  std::vector<int>
  dftfeWrapper::getValenceElectronNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<int> valenceNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (unsigned int i = 0; i < valenceNumbers.size(); ++i)
      valenceNumbers[i] = temp[i][1];
    return valenceNumbers;
  }

  dftBase *
  dftfeWrapper::getDftfeBasePtr()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr;
  }

} // namespace dftfe
