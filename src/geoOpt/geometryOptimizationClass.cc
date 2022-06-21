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
// @author Nikhil Kodali
//



#include <dftUtils.h>
#include <fileReaders.h>
#include <geometryOptimizationClass.h>
#include <sys/stat.h>


namespace dftfe
{
  geometryOptimizationClass::geometryOptimizationClass(
    const std::string parameter_file,
    const MPI_Comm &  mpi_comm_parent,
    const bool        restart)
    : d_mpiCommParent(mpi_comm_parent)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isRestart(restart)
  {
    init(parameter_file);
  }

  void
  geometryOptimizationClass::init(const std::string parameter_file)
  {
    if (d_isRestart)
      {
        std::string coordinatesFile, domainVectorsFile, restartPath;
        std::vector<std::vector<double>> tmp, optData;
        pcout << "Restarting Optimization using saved data." << std::endl;
        optData.clear();
        dftUtils::readFile(1, optData, "optRestart/geometryOptimization.dat");
        d_optMode              = (int)optData[0][0];
        bool        isPeriodic = optData[1][0] > 1e-6;
        std::string chkPath    = "optRestart/";
        dftUtils::readFile(1, tmp, chkPath + "/cycle.chk");
        d_cycle = tmp[0][0];
        chkPath += "cycle" + std::to_string(d_cycle);
        tmp.clear();
        dftUtils::readFile(1, tmp, chkPath + "/status.chk");
        d_status               = tmp[0][0];
        bool restartFilesFound = false;
        while (d_cycle >= 0)
          {
            std::string tempfolder =
              "optRestart/cycle" + std::to_string(d_cycle);
            tempfolder += d_status == 0 ? "/ionRelax/step" : "/cellRelax/step";
            tmp.clear();
            dftUtils::readFile(1, tmp, tempfolder + ".chk");
            int lastSavedStep = tmp[0][0];
            while (lastSavedStep >= 0)
              {
                std::string path = tempfolder + std::to_string(lastSavedStep);
                pcout << "Looking for geometry files of step " << lastSavedStep
                      << " at: " << path << std::endl;
                std::string file1 = isPeriodic ?
                                      path + "/atomsFracCoordCurrent.chk" :
                                      path + "/atomsCartCoordCurrent.chk";
                std::string file2 = path + "/domainBoundingVectorsCurrent.chk";
                std::ifstream readFile1(file1.c_str());
                std::ifstream readFile2(file2.c_str());
                if (!readFile1.fail() && !readFile2.fail())
                  {
                    coordinatesFile   = file1;
                    domainVectorsFile = file2;
                    tmp.clear();
                    tmp.resize(1, std::vector<double>(1, lastSavedStep));
                    dftUtils::writeDataIntoFile(tmp,
                                                tempfolder + ".chk",
                                                d_mpiCommParent);

                    pcout << "Geometry restart files are found in: " << path
                          << std::endl;
                    restartFilesFound = true;
                    restartPath       = path;
                    break;
                  }

                else
                  pcout << "----Error opening restart files present in: "
                        << path << std::endl
                        << "Switching to step: " << --lastSavedStep << " ----"
                        << std::endl;
              }
            if (restartFilesFound)
              break;
            else if (d_optMode == 2)
              {
                d_cycle  = d_status == 0 ? d_cycle - 1 : d_cycle;
                d_status = d_status == 0 ? 1 : 0;
              }
          }
        if (!restartFilesFound)
          {
            AssertThrow(
              false, ExcMessage("DFT-FE Error: Unable to find restart files."));
          }
        d_dftfeWrapper = std::make_unique<dftfeWrapper>(parameter_file,
                                                        coordinatesFile,
                                                        domainVectorsFile,
                                                        MPI_COMM_WORLD,
                                                        true,
                                                        true);
        d_dftPtr       = d_dftfeWrapper->getDftfeBasePtr();
        d_geoOptIonPtr =
          std::make_unique<geoOptIon>(d_dftPtr, d_mpiCommParent, d_status == 0);
        d_geoOptCellPtr = std::make_unique<geoOptCell>(d_dftPtr,
                                                       d_mpiCommParent,
                                                       d_status == 1);
      }
    else
      {
        d_dftfeWrapper = std::make_unique<dftfeWrapper>(parameter_file,
                                                        d_mpiCommParent,
                                                        true,
                                                        true);
        d_dftPtr       = d_dftfeWrapper->getDftfeBasePtr();
        if (d_dftPtr->getParametersObject().isIonOpt &&
            !d_dftPtr->getParametersObject().isCellOpt)
          d_optMode = 0;
        else if (!d_dftPtr->getParametersObject().isIonOpt &&
                 d_dftPtr->getParametersObject().isCellOpt)
          d_optMode = 1;
        else if (d_dftPtr->getParametersObject().isIonOpt &&
                 d_dftPtr->getParametersObject().isCellOpt)
          d_optMode = 2;
        else
          AssertThrow(
            false,
            ExcMessage(
              "DFT-FE Error: SOLVER MODE is set to OPT but ION OPT and CELL OPT are set to FALSE."));
        mkdir("optRestart", ACCESSPERMS);
        std::vector<std::vector<double>> optData(2,
                                                 std::vector<double>(1, 0.0));
        optData[0][0] = d_optMode;
        optData[1][0] = d_dftPtr->getParametersObject().periodicX ||
                            d_dftPtr->getParametersObject().periodicY ||
                            d_dftPtr->getParametersObject().periodicZ ?
                          1 :
                          0;
        dftUtils::writeDataIntoFile(optData,
                                    "optRestart/geometryOptimization.dat",
                                    d_mpiCommParent);
        d_cycle  = 0;
        d_status = d_optMode == 1 ? 1 : 0;
        d_geoOptIonPtr =
          std::make_unique<geoOptIon>(d_dftPtr, d_mpiCommParent, d_isRestart);
        d_geoOptCellPtr =
          std::make_unique<geoOptCell>(d_dftPtr, d_mpiCommParent, d_isRestart);
      }
  }


  void
  geometryOptimizationClass::runOpt()
  {
    bool                             isConverged = false;
    std::vector<std::vector<double>> tmpData(1, std::vector<double>(1, 0.0));
    // Define max cycles?
    while (d_cycle < 100)
      {
        pcout << "Starting optimization cycle: " << d_cycle << std::endl;
        tmpData[0][0] = d_cycle;
        dftUtils::writeDataIntoFile(tmpData,
                                    "optRestart/cycle.chk",
                                    d_mpiCommParent);

        std::string restartPath = "optRestart/cycle" + std::to_string(d_cycle);
        mkdir(restartPath.c_str(), ACCESSPERMS);
        tmpData[0][0] = d_status;
        dftUtils::writeDataIntoFile(tmpData,
                                    restartPath + "/status.chk",
                                    d_mpiCommParent);
        if (d_cycle == 0 && !d_isRestart)
          {
            d_dftPtr->solve(true, true, false);
          }

        if (d_status == 0)
          {
            pcout << "Starting ion optimization" << std::endl;
            d_geoOptIonPtr->init(restartPath);
            int geoOptStatus = d_geoOptIonPtr->run();
            if (d_optMode == 0)
              {
                isConverged = geoOptStatus >= 0;
                break;
              }
            else
              {
                isConverged = geoOptStatus == 0 && d_cycle != 0;
                if (isConverged)
                  break;
                d_status = 1;
              }
          }
        tmpData[0][0] = d_status;
        dftUtils::writeDataIntoFile(tmpData,
                                    restartPath + "/status.chk",
                                    d_mpiCommParent);
        if (d_optMode == 2 && d_status == 1)
          {
            d_dftPtr->trivialSolveForStress();
          }
        if (d_status == 1)
          {
            d_geoOptCellPtr->init(restartPath);
            int geoOptStatus = d_geoOptCellPtr->run();
            if (d_optMode == 1)
              {
                isConverged = geoOptStatus >= 0;
                break;
              }
            else
              {
                isConverged = geoOptStatus == 0;
                if (isConverged)
                  break;
                d_status = 0;
                ++d_cycle;
              }
          }
      }
  }

} // namespace dftfe
