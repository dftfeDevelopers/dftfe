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
#include <chrono>
#include <sys/time.h>
#include <ctime>

#include "dft.h"
#include "dftParameters.h"
#include "deviceKernelsGeneric.h"
#include "dftUtils.h"
#include "dftfeWrapper.h"
#include "fileReaders.h"
#include "PeriodicTable.h"

namespace dftfe
{
  namespace internalWrapper
  {
    int
    divisor_closest(int totalSize, int desiredDivisor)
    {
      int i;
      for (i = desiredDivisor; i >= 1; --i)
        {
          if (totalSize % i == 0 && i <= desiredDivisor)
            return i;
        }
      return 1;
    }


    template <int n1, int n2>
    void
    create_dftfe(const MPI_Comm &      mpi_comm_parent,
                 const MPI_Comm &      mpi_comm_domain,
                 const MPI_Comm &      interpoolcomm,
                 const MPI_Comm &      interBandGroupComm,
                 const std::string &   scratchFolderName,
                 dftfe::dftParameters &dftParams,
                 dftBase **            dftfeBaseDoublePtr)
    {
      *dftfeBaseDoublePtr = new dftfe::dftClass<n1, n2>(mpi_comm_parent,
                                                        mpi_comm_domain,
                                                        interpoolcomm,
                                                        interBandGroupComm,
                                                        scratchFolderName,
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
                              const std::string &   scratchFolderName,
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
  } // namespace internalWrapper

  void
  dftfeWrapper::globalHandlesInitialize(const MPI_Comm &mpi_comm_world)
  {
    sc_init(mpi_comm_world, 0, 0, nullptr, SC_LP_SILENT);
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
    dealii::MultithreadInfo::set_thread_limit(1);
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
  dftfeWrapper::dftfeWrapper()
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {}

  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const MPI_Comm &  mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const int         _verbosity)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(parameter_file,
           mpi_comm_parent,
           printParams,
           setDeviceToMPITaskBindingInternally,
           mode,
           restartFilesPath,
           _verbosity);
  }


  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const std::string restartCoordsFile,
                             const std::string restartDomainVectorsFile,
                             const MPI_Comm &  mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const int         _verbosity,
                             const bool        isScfRestart)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(parameter_file,
           restartCoordsFile,
           restartDomainVectorsFile,
           mpi_comm_parent,
           printParams,
           setDeviceToMPITaskBindingInternally,
           mode,
           restartFilesPath,
           _verbosity,
           isScfRestart);
  }



  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(
    const MPI_Comm &                       mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<unsigned int>        atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<unsigned int>        mpGrid,
    const std::vector<bool>                mpGridShift,
    const bool                             spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const unsigned int                     npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const int                              verbosity,
    const bool                             setDeviceToMPITaskBindingInternally)
    : d_dftfeBasePtr(nullptr)
    , d_dftfeParamsPtr(nullptr)
    , d_mpi_comm_parent(MPI_COMM_NULL)
    , d_isDeviceToMPITaskBindingSetInternally(false)
  {
    reinit(mpi_comm_parent,
           useDevice,
           atomicPositionsCart,
           atomicNumbers,
           cell,
           pbc,
           mpGrid,
           mpGridShift,
           spinPolarizedDFT,
           startMagnetization,
           fermiDiracSmearingTemp,
           npkpt,
           meshSize,
           scfMixingParameter,
           verbosity,
           setDeviceToMPITaskBindingInternally);
  }


  dftfeWrapper::~dftfeWrapper()
  {
    clear();
  }

  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const MPI_Comm &  mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const int         _verbosity)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);
    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file,
                                           d_mpi_comm_parent,
                                           printParams,
                                           mode,
                                           restartFilesPath,
                                           _verbosity);
      }
    initialize(setDeviceToMPITaskBindingInternally);
  }


  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const std::string restartCoordsFile,
                       const std::string restartDomainVectorsFile,
                       const MPI_Comm &  mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const int         _verbosity,
                       const bool        isScfRestart)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);

    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file,
                                           d_mpi_comm_parent,
                                           printParams,
                                           mode,
                                           restartFilesPath,
                                           _verbosity);
        d_dftfeParamsPtr->coordinatesFile           = restartCoordsFile;
        d_dftfeParamsPtr->domainBoundingVectorsFile = restartDomainVectorsFile;
        d_dftfeParamsPtr->loadRhoData =
          d_dftfeParamsPtr->loadRhoData && isScfRestart;
      }
    initialize(setDeviceToMPITaskBindingInternally);
  }


  void
  dftfeWrapper::reinit(
    const MPI_Comm &                       mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<unsigned int>        atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<unsigned int>        mpGrid,
    const std::vector<bool>                mpGridShift,
    const bool                             spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const unsigned int                     npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const int                              verbosity,
    const bool                             setDeviceToMPITaskBindingInternally)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      {
        int ierr = MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);
        if (ierr != 0)
          {
            throw std::runtime_error("MPI_Comm_dup failed.");
          }
      }

    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        const int totalMPIProcesses =
          dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_parent);

        std::string parameter_file_path =
          d_scratchFolderName + "/parameterFile.prm";

        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          {
            AssertThrow(
              atomicPositionsCart.size() == atomicNumbers.size(),
              dealii::ExcMessage(
                "DFT-FE Error:  Mismatch in sizes of atomicPositionsCart and atomicNumbers."));
            //
            // write pseudo.inp
            //
            std::set<unsigned int> atomicNumbersSet;
            for (unsigned int i = 0; i < atomicNumbers.size(); i++)
              atomicNumbersSet.insert(atomicNumbers[i]);

            std::vector<unsigned int> atomicNumbersUniqueVec(
              atomicNumbersSet.size());
            std::copy(atomicNumbersSet.begin(),
                      atomicNumbersSet.end(),
                      atomicNumbersUniqueVec.begin());


            const std::string dftfePspPath(getenv("DFTFE_PSP_PATH"));

            pseudoUtils::PeriodicTable periodicTable;
            const std::string          dftfePseudoFileName =
              d_scratchFolderName + "/pseudo.inp";
            std::ofstream dftfePseudoFile(dftfePseudoFileName);
            if (dftfePseudoFile.is_open())
              {
                for (unsigned int irow = 0;
                     irow < atomicNumbersUniqueVec.size();
                     ++irow)
                  {
                    const std::string upffilePath =
                      dftfePspPath + "/" +
                      periodicTable.symbol(atomicNumbersUniqueVec[irow]) +
                      ".upf";

                    dftfePseudoFile
                      << std::to_string(atomicNumbersUniqueVec[irow]);
                    dftfePseudoFile << " ";
                    dftfePseudoFile << upffilePath;
                    dftfePseudoFile << "\n";
                  }

                dftfePseudoFile.close();
              }

            //
            // write coordinates.inp
            //
            std::map<unsigned int, unsigned int> atomicNumberToValenceNumberMap;

            for (unsigned int i = 0; i < atomicNumbersUniqueVec.size(); i++)
              {
                const std::string upffilePath =
                  dftfePspPath + "/" +
                  periodicTable.symbol(atomicNumbersUniqueVec[i]) + ".upf";
                std::ifstream upffile(upffilePath);
                double        valenceNumber = 0;
                std::string   line;
                while (getline(upffile, line))
                  {
                    if (line.find("z_valence=") == std::string::npos)
                      continue;
                    std::istringstream ss(line);
                    std::string        dummy1;
                    std::string        dummy2;
                    ss >> dummy1 >> valenceNumber >> dummy2;
                    break;
                  }
                atomicNumberToValenceNumberMap[atomicNumbersUniqueVec[i]] =
                  std::round(valenceNumber);
              }

            std::vector<std::vector<double>> dftfeCoordinates(
              atomicPositionsCart.size(), std::vector<double>(5, 0));

            std::vector<double> cellVectorsFlattened(9, 0.0);
            for (unsigned int idim = 0; idim < 3; idim++)
              for (unsigned int jdim = 0; jdim < 3; jdim++)
                cellVectorsFlattened[3 * idim + jdim] = cell[idim][jdim];

            if (pbc[0] == false && pbc[1] == false && pbc[2] == false)
              {
                std::vector<double> shift(3, 0.0);
                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    shift[idim] = 0;
                    for (unsigned int jdim = 0; jdim < 3; jdim++)
                      shift[idim] -= cell[jdim][idim] / 2.0;
                  }
                for (unsigned int i = 0; i < dftfeCoordinates.size(); i++)
                  {
                    dftfeCoordinates[i][0] = atomicNumbers[i];
                    dftfeCoordinates[i][1] =
                      atomicNumberToValenceNumberMap[atomicNumbers[i]];

                    std::vector<double> coord(3, 0.0);
                    coord[0] = atomicPositionsCart[i][0];
                    coord[1] = atomicPositionsCart[i][1];
                    coord[2] = atomicPositionsCart[i][2];

                    std::vector<double> frac =
                      dftUtils::getFractionalCoordinates(cellVectorsFlattened,
                                                         coord);
                    for (unsigned int idim = 0; idim < 3; idim++)
                      AssertThrow(
                        frac[idim] > 1e-7 && frac[idim] < (1.0 - 1e-7),
                        dealii::ExcMessage(
                          "DFT-FE Error: all coordinates are not inside the cell. Please check input atomicPositionsCart."));

                    dftfeCoordinates[i][2] =
                      atomicPositionsCart[i][0] + shift[0];
                    dftfeCoordinates[i][3] =
                      atomicPositionsCart[i][1] + shift[1];
                    dftfeCoordinates[i][4] =
                      atomicPositionsCart[i][2] + shift[2];
                  }
              }
            else
              {
                for (unsigned int i = 0; i < dftfeCoordinates.size(); i++)
                  {
                    dftfeCoordinates[i][0] = atomicNumbers[i];
                    dftfeCoordinates[i][1] =
                      atomicNumberToValenceNumberMap[atomicNumbers[i]];
                    std::vector<double> coord(3, 0.0);
                    coord[0] = atomicPositionsCart[i][0];
                    coord[1] = atomicPositionsCart[i][1];
                    coord[2] = atomicPositionsCart[i][2];

                    std::vector<double> frac =
                      dftUtils::getFractionalCoordinates(cellVectorsFlattened,
                                                         coord);
                    for (unsigned int idim = 0; idim < 3; idim++)
                      AssertThrow(
                        frac[idim] > -1e-7 && frac[idim] < (1.0 + 1e-7),
                        dealii::ExcMessage(
                          "DFT-FE Error: fractional coordinates doesn't lie in [0,1]. Please check input atomicPositionsCart."));

                    dftfeCoordinates[i][2] = frac[0];
                    dftfeCoordinates[i][3] = frac[1];
                    dftfeCoordinates[i][4] = frac[2];
                  }
              }

            const std::string dftfeCoordsFileName =
              d_scratchFolderName + "/coordinates.inp";
            dftUtils::writeDataIntoFile(dftfeCoordinates, dftfeCoordsFileName);
            //
            // write domainVectors.inp
            //
            const std::string dftfeCellFileName =
              d_scratchFolderName + "/domainVectors.inp";
            dftUtils::writeDataIntoFile(cell, dftfeCellFileName);



            std::string dftfePath = DFTFE_PATH;
            std::string sourceFilePath =
              dftfePath + "/helpers/parameterFile.prm";

            std::string cmd;

            cmd = std::string("cp '") + sourceFilePath + "' '" +
                  parameter_file_path + "'";
            system(cmd.c_str());

            cmd = "sed -i 's/set NATOMS=.*/set NATOMS=" +
                  std::to_string(atomicPositionsCart.size()) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set NATOM TYPES=.*/set NATOM TYPES=" +
                  std::to_string(atomicNumbersUniqueVec.size()) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            const std::string dftfeCoordsFileNameForSed =
              d_scratchFolderName + "\\\/coordinates.inp";
            cmd =
              "sed -i 's/set ATOMIC COORDINATES FILE=.*/set ATOMIC COORDINATES FILE=" +
              dftfeCoordsFileNameForSed + "/g' " + parameter_file_path;
            system(cmd.c_str());

            const std::string dftfeCellFileNameForSed =
              d_scratchFolderName + "\\\/domainVectors.inp";
            cmd =
              "sed -i 's/set DOMAIN VECTORS FILE=.*/set DOMAIN VECTORS FILE=" +
              dftfeCellFileNameForSed + "/g' " + parameter_file_path;
            system(cmd.c_str());

            const std::string dftfePseudoFileNameForSed =
              d_scratchFolderName + "\\\/pseudo.inp";
            cmd =
              "sed -i 's/set PSEUDOPOTENTIAL FILE NAMES LIST=.*/set PSEUDOPOTENTIAL FILE NAMES LIST=" +
              dftfePseudoFileNameForSed + "/g' " + parameter_file_path;
            system(cmd.c_str());

            if (pbc[0] == false && pbc[1] == false && pbc[2] == false)
              {
                const std::string option = "false";
                cmd = "sed -i 's/set CELL STRESS=.*/set CELL STRESS=" + option +
                      "/g' " + parameter_file_path;
                system(cmd.c_str());
              }

            const std::string pbc1 = pbc[0] ? "true" : "false";
            cmd = "sed -i 's/set PERIODIC1=.*/set PERIODIC1=" + pbc1 + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            const std::string pbc2 = pbc[1] ? "true" : "false";
            cmd = "sed -i 's/set PERIODIC2=.*/set PERIODIC2=" + pbc2 + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            const std::string pbc3 = pbc[2] ? "true" : "false";
            cmd = "sed -i 's/set PERIODIC3=.*/set PERIODIC3=" + pbc3 + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING POINTS 1=.*/set SAMPLING POINTS 1=" +
                  std::to_string(mpGrid[0]) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING POINTS 2=.*/set SAMPLING POINTS 2=" +
                  std::to_string(mpGrid[1]) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING POINTS 3=.*/set SAMPLING POINTS 3=" +
                  std::to_string(mpGrid[2]) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING SHIFT 1=.*/set SAMPLING SHIFT 1=" +
                  std::to_string(mpGridShift[0] ? 1 : 0) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING SHIFT 2=.*/set SAMPLING SHIFT 2=" +
                  std::to_string(mpGridShift[1] ? 1 : 0) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set SAMPLING SHIFT 3=.*/set SAMPLING SHIFT 3=" +
                  std::to_string(mpGridShift[2] ? 1 : 0) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            const int spin = spinPolarizedDFT ? 1 : 0;
            cmd = "sed -i 's/set SPIN POLARIZATION=.*/set SPIN POLARIZATION=" +
                  std::to_string(spin) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd =
              "sed -i 's/set START MAGNETIZATION=.*/set START MAGNETIZATION=" +
              std::to_string(startMagnetization) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set TEMPERATURE=.*/set TEMPERATURE=" +
                  std::to_string(fermiDiracSmearingTemp) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set MIXING PARAMETER=.*/set MIXING PARAMETER=" +
                  std::to_string(scfMixingParameter) + "/g' " +
                  parameter_file_path;
            system(cmd.c_str());

            const int totalIrreducibleKpt =
              mpGrid[0] * mpGrid[1] * mpGrid[2] / 2;
            const int npkptSet =
              npkpt > 0 ? 1 :
                          internalWrapper::divisor_closest(totalMPIProcesses,
                                                           totalIrreducibleKpt);
            cmd =
              "sed -i 's/set NPKPT=.*/set NPKPT=" + std::to_string(npkptSet) +
              "/g' " + parameter_file_path;
            system(cmd.c_str());


            cmd =
              "sed -i 's/set MESH SIZE AROUND ATOM=.*/set MESH SIZE AROUND ATOM=" +
              std::to_string(meshSize) + "/g' " + parameter_file_path;
            system(cmd.c_str());

            cmd = "sed -i 's/set VERBOSITY=.*/set VERBOSITY=" +
                  std::to_string(verbosity) + "/g' " + parameter_file_path;
            system(cmd.c_str());
          }
        MPI_Barrier(d_mpi_comm_parent);
        d_dftfeParamsPtr = new dftfe::dftParameters;
        d_dftfeParamsPtr->parse_parameters(parameter_file_path,
                                           d_mpi_comm_parent,
                                           false,
                                           "GS");
#ifdef DFTFE_WITH_DEVICE
        d_dftfeParamsPtr->useDevice = useDevice;
#endif
      }
    initialize(setDeviceToMPITaskBindingInternally);
  }


  void
  dftfeWrapper::createScratchFolder()
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          {
            d_scratchFolderName =
              "dftfeScratch" +
              std::to_string(
                dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
              "t" +
              std::to_string(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
          }

        int line_size = d_scratchFolderName.size();
        MPI_Bcast(&line_size, 1, MPI_INT, 0, d_mpi_comm_parent);
        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) != 0)
          d_scratchFolderName.resize(line_size);
        MPI_Bcast(const_cast<char *>(d_scratchFolderName.data()),
                  line_size,
                  MPI_CHAR,
                  0,
                  d_mpi_comm_parent);

        if (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)
          mkdir(d_scratchFolderName.c_str(), ACCESSPERMS);

        MPI_Barrier(d_mpi_comm_parent);
      }
  }

  void
  dftfeWrapper::initialize(const bool setDeviceToMPITaskBindingInternally)
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
#ifdef DFTFE_WITH_DEVICE
        if (d_dftfeParamsPtr->useDevice &&
            setDeviceToMPITaskBindingInternally &&
            !d_isDeviceToMPITaskBindingSetInternally)
          {
            dftfe::utils::deviceKernelsGeneric::setupDevice();
            d_isDeviceToMPITaskBindingSetInternally = true;
          }
#endif

        dftfe::dftUtils::Pool kPointPool(d_mpi_comm_parent,
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
              (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) ==
               0));
            pcout
              << "=================================MPI Parallelization========================================="
              << std::endl;
            pcout << "Total number of MPI tasks: "
                  << dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_parent)
                  << std::endl;
            pcout << "k-point parallelization processor groups: "
                  << dealii::Utilities::MPI::n_mpi_processes(
                       kPointPool.get_interpool_comm())
                  << std::endl;
            pcout << "Band parallelization processor groups: "
                  << dealii::Utilities::MPI::n_mpi_processes(
                       bandGroupsPool.get_interpool_comm())
                  << std::endl;
            pcout
              << "Number of MPI tasks for finite-element domain decomposition: "
              << dealii::Utilities::MPI::n_mpi_processes(
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
        internalWrapper::create_fn create =
          internalWrapper::order_list[listIndex - 1];
        create(d_mpi_comm_parent,
               bandGroupsPool.get_intrapool_comm(),
               kPointPool.get_interpool_comm(),
               bandGroupsPool.get_interpool_comm(),
               d_scratchFolderName,
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
          {
            delete d_dftfeBasePtr;

            if (!d_dftfeParamsPtr->keepScratchFolder &&
                dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) ==
                  0)
              {
                std::string command = "rm -rf " + d_scratchFolderName;
                system(command.c_str());
              }
            MPI_Barrier(d_mpi_comm_parent);
          }
        if (d_dftfeParamsPtr != nullptr)
          delete d_dftfeParamsPtr;
        MPI_Comm_free(&d_mpi_comm_parent);
      }
    d_dftfeBasePtr    = nullptr;
    d_dftfeParamsPtr  = nullptr;
    d_mpi_comm_parent = MPI_COMM_NULL;
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

  void
  dftfeWrapper::writeMesh()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->writeMesh();
  }


  std::tuple<double, bool, double>
  dftfeWrapper::computeDFTFreeEnergy(const bool computeIonForces,
                                     const bool computeCellStress)
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::tuple<bool, double> t =
      d_dftfeBasePtr->solve(computeIonForces, computeCellStress);
    return std::make_tuple(d_dftfeBasePtr->getFreeEnergy(),
                           std::get<0>(t),
                           std::get<1>(t));
  }

  void
  dftfeWrapper::computeStress()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));

    d_dftfeBasePtr->computeStress();
  }

  double
  dftfeWrapper::getDFTFreeEnergy() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getFreeEnergy();
  }


  double
  dftfeWrapper::getElectronicEntropicEnergy() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeBasePtr->getEntropicEnergy();
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
    std::vector<dealii::Tensor<1, 3, double>> dispVec(
      atomsDisplacements.size());
    for (unsigned int i = 0; i < dispVec.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        dispVec[i][j] = atomsDisplacements[i][j];
    d_dftfeBasePtr->updateAtomPositionsAndMoveMesh(dispVec);
  }

  void
  dftfeWrapper::deformCell(
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
  dftfeWrapper::getAtomPositionsCart() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    // dftfe stores cell centered coordinates
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<std::vector<double>> atomLocationsCart(
      d_dftfeBasePtr->getAtomLocationsCart().size(),
      std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> cell = d_dftfeBasePtr->getCell();
    std::vector<double>              shift(3, 0.0);
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        shift[idim] = 0;
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          shift[idim] += cell[jdim][idim] / 2.0;
      }

    for (unsigned int i = 0; i < atomLocationsCart.size(); ++i)
      for (unsigned int j = 0; j < 3; ++j)
        atomLocationsCart[i][j] = temp[i][j + 2] + shift[j];
    return atomLocationsCart;
  }

  std::vector<std::vector<double>>
  dftfeWrapper::getAtomPositionsFrac() const
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


  void
  dftfeWrapper::writeDomainAndAtomCoordinates(const std::string Path) const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    d_dftfeBasePtr->writeDomainAndAtomCoordinates(Path);
  }



} // namespace dftfe
