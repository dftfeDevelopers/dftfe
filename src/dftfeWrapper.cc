// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
#include <deal.II/base/multithread_info.h>
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
#include "MemorySpaceType.h"

namespace dftfe
{
  namespace internalWrapper
  {
    dftfe::Int
    divisor_closest(dftfe::Int totalSize, dftfe::Int desiredDivisor)
    {
      dftfe::Int i;
      for (i = desiredDivisor; i >= 1; --i)
        {
          if (totalSize % i == 0 && i <= desiredDivisor)
            return i;
        }
      return 1;
    }


    template <dftfe::utils::MemorySpace memory>
    void
    create_dftfe(const MPI_Comm       &mpi_comm_parent,
                 const MPI_Comm       &mpi_comm_domain,
                 const MPI_Comm       &interpoolcomm,
                 const MPI_Comm       &interBandGroupComm,
                 const std::string    &scratchFolderName,
                 dftfe::dftParameters &dftParams,
                 dftBase             **dftfeBaseDoublePtr)
    {
      *dftfeBaseDoublePtr = new dftfe::dftClass<memory>(mpi_comm_parent,
                                                        mpi_comm_domain,
                                                        interpoolcomm,
                                                        interBandGroupComm,
                                                        scratchFolderName,
                                                        dftParams);
    }
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
                             const MPI_Comm   &mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const dftfe::Int  _verbosity,
                             const bool        useDevice)
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
           _verbosity,
           useDevice);
  }


  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(const std::string parameter_file,
                             const std::string restartCoordsFile,
                             const std::string restartDomainVectorsFile,
                             const MPI_Comm   &mpi_comm_parent,
                             const bool        printParams,
                             const bool setDeviceToMPITaskBindingInternally,
                             const std::string mode,
                             const std::string restartFilesPath,
                             const dftfe::Int  _verbosity,
                             const bool        useDevice,
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
           useDevice,
           isScfRestart);
  }



  //
  // constructor
  //
  dftfeWrapper::dftfeWrapper(
    const MPI_Comm                        &mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<dftfe::uInt>         atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<dftfe::uInt>         mpGrid,
    const std::vector<bool>                mpGridShift,
    const bool                             spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const dftfe::uInt                      npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const dftfe::Int                       verbosity,
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
                       const MPI_Comm   &mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const dftfe::Int  _verbosity,
                       const bool        useDevice)
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
                                           _verbosity,
                                           useDevice);
      }
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
  }


  void
  dftfeWrapper::reinit(const std::string parameter_file,
                       const std::string restartCoordsFile,
                       const std::string restartDomainVectorsFile,
                       const MPI_Comm   &mpi_comm_parent,
                       const bool        printParams,
                       const bool        setDeviceToMPITaskBindingInternally,
                       const std::string mode,
                       const std::string restartFilesPath,
                       const dftfe::Int  _verbosity,
                       const bool        useDevice,
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
                                           _verbosity,
                                           useDevice);
        d_dftfeParamsPtr->coordinatesFile           = restartCoordsFile;
        d_dftfeParamsPtr->domainBoundingVectorsFile = restartDomainVectorsFile;
        d_dftfeParamsPtr->loadQuadData =
          d_dftfeParamsPtr->loadQuadData && isScfRestart;
      }
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
  }


  void
  dftfeWrapper::reinit(
    const MPI_Comm                        &mpi_comm_parent,
    const bool                             useDevice,
    const std::vector<std::vector<double>> atomicPositionsCart,
    const std::vector<dftfe::uInt>         atomicNumbers,
    const std::vector<std::vector<double>> cell,
    const std::vector<bool>                pbc,
    const std::vector<dftfe::uInt>         mpGrid,
    const std::vector<bool>                mpGridShift,
    const bool                             spinPolarizedDFT,
    const double                           startMagnetization,
    const double                           fermiDiracSmearingTemp,
    const dftfe::uInt                      npkpt,
    const double                           meshSize,
    const double                           scfMixingParameter,
    const dftfe::Int                       verbosity,
    const bool                             setDeviceToMPITaskBindingInternally)
  {
    clear();
    if (mpi_comm_parent != MPI_COMM_NULL)
      {
        dftfe::Int ierr = MPI_Comm_dup(mpi_comm_parent, &d_mpi_comm_parent);
        if (ierr != 0)
          {
            throw std::runtime_error("MPI_Comm_dup failed.");
          }
      }

    createScratchFolder();

    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
        const dftfe::Int totalMPIProcesses =
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
            std::set<dftfe::uInt> atomicNumbersSet;
            for (dftfe::uInt i = 0; i < atomicNumbers.size(); i++)
              atomicNumbersSet.insert(atomicNumbers[i]);

            std::vector<dftfe::uInt> atomicNumbersUniqueVec(
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
                for (dftfe::uInt irow = 0; irow < atomicNumbersUniqueVec.size();
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
            std::map<dftfe::uInt, dftfe::uInt> atomicNumberToValenceNumberMap;

            for (dftfe::uInt i = 0; i < atomicNumbersUniqueVec.size(); i++)
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
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
                cellVectorsFlattened[3 * idim + jdim] = cell[idim][jdim];

            if (pbc[0] == false && pbc[1] == false && pbc[2] == false)
              {
                std::vector<double> shift(3, 0.0);
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  {
                    shift[idim] = 0;
                    for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
                      shift[idim] -= cell[jdim][idim] / 2.0;
                  }
                for (dftfe::uInt i = 0; i < dftfeCoordinates.size(); i++)
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
                    for (dftfe::uInt idim = 0; idim < 3; idim++)
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
                for (dftfe::uInt i = 0; i < dftfeCoordinates.size(); i++)
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
                    for (dftfe::uInt idim = 0; idim < 3; idim++)
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

            const dftfe::Int spin = spinPolarizedDFT ? 1 : 0;
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

            const dftfe::Int totalIrreducibleKpt =
              mpGrid[0] * mpGrid[1] * mpGrid[2] / 2;
            const dftfe::Int npkptSet =
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
    initialize(setDeviceToMPITaskBindingInternally, useDevice);
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

        dftfe::Int line_size = d_scratchFolderName.size();
        MPI_Bcast(&line_size,
                  1,
                  dftfe::dataTypes::mpi_type_id(&line_size),
                  0,
                  d_mpi_comm_parent);
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
  dftfeWrapper::initialize(const bool setDeviceToMPITaskBindingInternally,
                           const bool useDevice)
  {
    if (d_mpi_comm_parent != MPI_COMM_NULL)
      {
#ifdef DFTFE_WITH_DEVICE
        if (useDevice && setDeviceToMPITaskBindingInternally &&
            !d_isDeviceToMPITaskBindingSetInternally)
          {
            dftfe::utils::deviceKernelsGeneric::setupDevice(
              dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent));
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

        dftfe::Int order = d_dftfeParamsPtr->finiteElementPolynomialOrder;
        dftfe::Int orderElectro =
          d_dftfeParamsPtr->finiteElementPolynomialOrderElectrostatics;

        if (!useDevice)
          {
            dftfe::internalWrapper::create_dftfe<
              dftfe::utils::MemorySpace::HOST>(
              d_mpi_comm_parent,
              bandGroupsPool.get_intrapool_comm(),
              kPointPool.get_interpool_comm(),
              bandGroupsPool.get_interpool_comm(),
              d_scratchFolderName,
              *d_dftfeParamsPtr,
              &d_dftfeBasePtr);
          }
#ifdef DFTFE_WITH_DEVICE
        else if (useDevice)
          {
            dftfe::internalWrapper::create_dftfe<
              dftfe::utils::MemorySpace::DEVICE>(
              d_mpi_comm_parent,
              bandGroupsPool.get_intrapool_comm(),
              kPointPool.get_interpool_comm(),
              bandGroupsPool.get_interpool_comm(),
              d_scratchFolderName,
              *d_dftfeParamsPtr,
              &d_dftfeBasePtr);
          }
#endif
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
    for (dftfe::uInt i = 0; i < ionicForces.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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

    for (dftfe::uInt i = 0; i < 3; ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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
    for (dftfe::uInt i = 0; i < dispVec.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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
    for (dftfe::uInt i = 0; i < 3; ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        shift[idim] = 0;
        for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
          shift[idim] += cell[jdim][idim] / 2.0;
      }

    for (dftfe::uInt i = 0; i < atomLocationsCart.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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
    for (dftfe::uInt i = 0; i < atomLocationsFrac.size(); ++i)
      for (dftfe::uInt j = 0; j < 3; ++j)
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

  std::vector<dftfe::Int>
  dftfeWrapper::getAtomicNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<dftfe::Int> atomicNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (dftfe::uInt i = 0; i < atomicNumbers.size(); ++i)
      atomicNumbers[i] = temp[i][0];
    return atomicNumbers;
  }


  std::vector<dftfe::Int>
  dftfeWrapper::getValenceElectronNumbers() const
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    std::vector<std::vector<double>> temp =
      d_dftfeBasePtr->getAtomLocationsCart();
    std::vector<dftfe::Int> valenceNumbers(
      d_dftfeBasePtr->getAtomLocationsCart().size(), 0);
    for (dftfe::uInt i = 0; i < valenceNumbers.size(); ++i)
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


  dftParameters *
  dftfeWrapper::getDftfeParamsPtr()
  {
    AssertThrow(
      d_mpi_comm_parent != MPI_COMM_NULL,
      dealii::ExcMessage(
        "DFT-FE Error: dftfeWrapper cannot be used on MPI_COMM_NULL."));
    return d_dftfeParamsPtr;
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
