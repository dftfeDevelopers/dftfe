
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
// @author Kartick Ramakrishnan
//
#include <cgPRPNonLinearSolver.h>
#include <BFGSNonLinearSolver.h>
#include <LBFGSNonLinearSolver.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include "nudgedElasticBandClass.h"
#include <sys/stat.h>


namespace dftfe
{
  nudgedElasticBandClass::nudgedElasticBandClass(
    const std::string  parameter_file,
    const std::string  restartFilesPath,
    const MPI_Comm &   mpi_comm_parent,
    const bool         restart,
    const int          verbosity,
    int                d_numberOfImages,
    bool               imageFreeze,
    double             Kmax,
    double             Kmin,
    double             pathThreshold,
    int                maximumNEBIteration,
    const std::string &coordinatesFileNEB,
    const std::string &domainVectorsFileNEB)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_restartFilesPath(restartFilesPath)
    , d_isRestart(restart)
    , d_verbosity(verbosity)
    , d_numberOfImages(d_numberOfImages)
    , d_imageFreeze(imageFreeze)
    , d_kmax(Kmax)
    , d_kmin(Kmin)
    , d_optimizertolerance(pathThreshold)
    , d_maximumNEBIteration(maximumNEBIteration)

  {
    // Read Coordinates file and create coordinates for each image

    MPI_Barrier(d_mpiCommParent);
    if (!d_isRestart)
      {
        if (d_this_mpi_process == 0)
          {
            if (d_restartFilesPath != ".")
              {
                mkdir(d_restartFilesPath.c_str(), ACCESSPERMS);
              }
            else
              {
                d_restartFilesPath = "./nebRestart";
                mkdir(d_restartFilesPath.c_str(), ACCESSPERMS);
              }
          }


        std::string Folder = d_restartFilesPath + "/Step0";
        if (Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          mkdir(Folder.c_str(), ACCESSPERMS);


        std::vector<std::vector<double>> initialatomLocations;
        std::vector<std::vector<double>> LatticeVectors;
        dftUtils::readFile(5, initialatomLocations, coordinatesFileNEB);
        dftUtils::readFile(3, LatticeVectors, domainVectorsFileNEB);
        d_numberGlobalCharges = initialatomLocations.size() / d_numberOfImages;
        for (int Image = 0; Image < d_numberOfImages; Image++)
          {
            std::string coordinatesFile, domainVectorsFile;
            coordinatesFile = d_restartFilesPath +
                              "/Step0/Image"+ std::to_string(Image) +
                              "coordinates.inp";
            domainVectorsFile = d_restartFilesPath + "/Step0/Image" +
                                std::to_string(Image) + "domainVectors.inp";
            std::vector<std::vector<double>> coordinates, domainVectors;
            for (int i = Image * d_numberGlobalCharges;
                 i < (Image + 1) * d_numberGlobalCharges;
                 i++)
              coordinates.push_back(initialatomLocations[i]);
            if (LatticeVectors.size() == 3)
              {
                for (int i = 0; i < 3; i++)
                  domainVectors.push_back(LatticeVectors[i]);
              }
            else
              {
                for (int i = Image * 3; i < (Image + 1) * 3; i++)
                  domainVectors.push_back(LatticeVectors[i]);
              }

            dftUtils::writeDataIntoFile(coordinates,
                                        coordinatesFile,
                                        d_mpiCommParent);
            dftUtils::writeDataIntoFile(domainVectors,
                                        domainVectorsFile,
                                        d_mpiCommParent);

            d_dftfeWrapper.push_back(
              std::make_unique<dftfe::dftfeWrapper>(parameter_file,
                                                    coordinatesFile,
                                                    domainVectorsFile,
                                                    d_mpiCommParent,
                                                    Image == 0 ? true : false,
                                                    true,
                                                    "NEB",
                                                    d_restartFilesPath,
                                                    false));
          }
      }
    d_dftPtr = d_dftfeWrapper[0]->getDftfeBasePtr();
    AssertThrow(
      d_dftPtr->getParametersObject().natoms == d_numberGlobalCharges,
      ExcMessage(
        "DFT-FE Error: The number atoms"
        "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
        "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
        "blank row at the end can cause this issue too."));

    std::vector<std::vector<double>> temp_domainBoundingVectors;
    dftUtils::readFile(
      3,
      temp_domainBoundingVectors,
      d_dftPtr->getParametersObject().domainBoundingVectorsFile);

    for (int i = 0; i < 3; i++)
      {
        double temp =
          temp_domainBoundingVectors[i][0] * temp_domainBoundingVectors[i][0] +
          temp_domainBoundingVectors[i][1] * temp_domainBoundingVectors[i][1] +
          temp_domainBoundingVectors[i][2] * temp_domainBoundingVectors[i][2];
        d_Length.push_back(pow(temp, 0.5));
      }
    pcout << "--$ Domain Length$ --" << std::endl;
    pcout << "Lx:= " << d_Length[0] << " Ly:=" << d_Length[1]
          << " Lz:=" << d_Length[2] << std::endl;



    init();
  }


  const MPI_Comm &
  nudgedElasticBandClass::getMPICommunicator()
  {
    return d_mpiCommParent;
  }


  void
  nudgedElasticBandClass::CalculatePathTangent(int                  image,
                                               std::vector<double> &tangent)
  {
    unsigned int count = 0;
    if (image != 0 && image != d_numberOfImages - 1)
      {
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiminus,
          atomLocationsiplus;
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiminus =
          (d_dftfeWrapper[image - 1])->getAtomPositionsCart();
        atomLocationsiplus =
          (d_dftfeWrapper[image + 1])->getAtomPositionsCart();
        double GSEnergyminus, GSEnergyplus, GSEnergy;
        GSEnergyminus = (d_dftfeWrapper[image - 1])->getDFTFreeEnergy();
        GSEnergyplus  = (d_dftfeWrapper[image + 1])->getDFTFreeEnergy();
        GSEnergy      = (d_dftfeWrapper[image])->getDFTFreeEnergy();
        if (GSEnergyplus > GSEnergy && GSEnergy > GSEnergyminus)
          {
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
              {
                for (int j = 0; j < 3; j++)
                  {
                    if (d_relaxationFlags[3 * iCharge + j] == 1)
                      {
                        double temp = atomLocationsiplus[iCharge][j + 2] -
                                      atomLocationsi[iCharge][j + 2];
                        if (temp > d_Length[j] / 2)
                          {
                            // pcout<<"Before: "<<temp;
                            temp -= d_Length[j];
                            temp = -temp;
                            // pcout<<" After: "<<temp<<std::endl;
                          }
                        else if (temp < -d_Length[j] / 2)
                          {
                            // pcout<<"Before: "<<temp;
                            temp += d_Length[j];
                            temp = -temp;
                            // pcout<<" After: "<<temp<<std::endl;
                          }
                        tangent[count] = temp;
                        count++;
                      }
                  }
              }
          }
        else if (GSEnergyminus > GSEnergy && GSEnergy > GSEnergyplus)
          {
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
              {
                for (int j = 0; j < 3; j++)
                  {
                    if (d_relaxationFlags[3 * iCharge + j] == 1)
                      {
                        double temp = atomLocationsi[iCharge][j + 2] -
                                      atomLocationsiminus[iCharge][j + 2];
                        if (temp > d_Length[j] / 2)
                          {
                            // pcout<<"Before: "<<temp;
                            temp -= d_Length[j];
                            temp = -temp;
                            // pcout<<" After: "<<temp<<std::endl;
                          }
                        else if (temp < -d_Length[j] / 2)
                          {
                            // pcout<<"Before: "<<temp;
                            temp += d_Length[j];
                            temp = -temp;
                            // pcout<<" After: "<<temp<<std::endl;
                          }
                        tangent[count] = temp;
                        count++;
                      }
                  }
              }
          }
        else
          {
            double deltaVmax, deltaVmin;
            deltaVmax = std::max(std::fabs(GSEnergyplus - GSEnergy),
                                 std::fabs(GSEnergyminus - GSEnergy));
            deltaVmin = std::min(std::fabs(GSEnergyplus - GSEnergy),
                                 std::fabs(GSEnergyminus - GSEnergy));

            if (GSEnergyplus > GSEnergyminus)
              {
                for (int iCharge = 0; iCharge < d_numberGlobalCharges;
                     iCharge++)
                  {
                    for (int j = 0; j < 3; j++)
                      {
                        if (d_relaxationFlags[3 * iCharge + j] == 1)
                          {
                            double temp1 = atomLocationsiplus[iCharge][j + 2] -
                                           atomLocationsi[iCharge][j + 2];
                            double temp2 = atomLocationsi[iCharge][j + 2] -
                                           atomLocationsiminus[iCharge][j + 2];
                            if (temp1 > d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp1;
                                temp1 -= d_Length[j];
                                temp1 = -temp1;
                                // pcout<<" After: "<<temp1<<std::endl;
                              }
                            else if (temp1 < -d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp1;
                                temp1 += d_Length[j];
                                temp1 = -temp1;
                                // pcout<<" After: "<<temp1<<std::endl;
                              }
                            if (temp2 > d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp2;
                                temp2 -= d_Length[j];
                                temp2 = -temp2;
                                // pcout<<" After: "<<temp2<<std::endl;
                              }
                            else if (temp2 < -d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp2;
                                temp2 += d_Length[j];
                                temp2 = -temp2;
                                // pcout<<" After: "<<temp2<<std::endl;
                              }


                            tangent[count] =
                              deltaVmax * (temp1) + deltaVmin * (temp2);
                            count++;
                          }
                      }
                  }
              }
            else if (GSEnergyplus < GSEnergyminus)
              {
                for (int iCharge = 0; iCharge < d_numberGlobalCharges;
                     iCharge++)
                  {
                    for (int j = 0; j < 3; j++)
                      {
                        if (d_relaxationFlags[3 * iCharge + j] == 1)
                          {
                            double temp1 = atomLocationsiplus[iCharge][j + 2] -
                                           atomLocationsi[iCharge][j + 2];
                            double temp2 = atomLocationsi[iCharge][j + 2] -
                                           atomLocationsiminus[iCharge][j + 2];
                            if (temp1 > d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp1;
                                temp1 -= d_Length[j];
                                temp1 = -temp1;
                                // pcout<<" After: "<<temp1<<std::endl;
                              }
                            else if (temp1 < -d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp1;
                                temp1 += d_Length[j];
                                temp1 = -temp1;
                                // pcout<<" After: "<<temp1<<std::endl;
                              }
                            if (temp2 > d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp2;
                                temp2 -= d_Length[j];
                                temp2 = -temp2;
                                // pcout<<" After: "<<temp2<<std::endl;
                              }
                            else if (temp2 < -d_Length[j] / 2)
                              {
                                // pcout<<"Before: "<<temp2;
                                temp2 += d_Length[j];
                                temp2 = -temp2;
                                // pcout<<" After: "<<temp2<<std::endl;
                              }
                            tangent[count] =
                              deltaVmin * (temp1) + deltaVmax * (temp2);
                            count++;
                          }
                      }
                  }
              }

            else
              for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
                {
                  for (int j = 0; j < 3; j++)
                    {
                      if (d_relaxationFlags[3 * iCharge + j] == 1)
                        {
                          double temp = (atomLocationsiplus[iCharge][j + 2] -
                                         atomLocationsiminus[iCharge][j + 2]);
                          if (temp > d_Length[j] / 2)
                            {
                              // pcout<<"Before: "<<temp;
                              temp -= d_Length[j];
                              temp = -temp;
                              // pcout<<" After: "<<temp<<std::endl;
                            }
                          else if (temp < -d_Length[j] / 2)
                            {
                              // pcout<<"Before: "<<temp;
                              temp += d_Length[j];
                              temp = -temp;
                              // pcout<<" After: "<<temp<<std::endl;
                            }
                          tangent[count] = temp;
                          count++;
                        }
                    }
                }
          }

        ReturnNormedVector(tangent, d_countrelaxationFlags);
      }
    else if (image == 0)
      {
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiplus;
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiplus =
          (d_dftfeWrapper[image + 1])->getAtomPositionsCart();
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    double temp = (atomLocationsiplus[iCharge][j + 2] -
                                   atomLocationsi[iCharge][j + 2]);
                    if (temp > d_Length[j] / 2)
                      {
                        // pcout<<iCharge<<" "<<j<<"Before: "<<temp;
                        temp -= d_Length[j];
                        temp = -temp;
                        // pcout<<" After: "<<temp<<std::endl;
                      }
                    else if (temp < -d_Length[j] / 2)
                      {
                        // pcout<<iCharge<<" "<<j<<"Before: "<<temp;
                        temp += d_Length[j];
                        temp = -temp;
                        // pcout<<" After: "<<temp<<std::endl;
                      }
                    tangent[count] = temp;
                    count++;
                  }
              }
          }
        ReturnNormedVector(tangent, d_countrelaxationFlags);
      }
    else if (image == d_numberOfImages - 1)
      {
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiminus;
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiminus =
          (d_dftfeWrapper[image - 1])->getAtomPositionsCart();
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    double temp = (atomLocationsi[iCharge][j + 2] -
                                   atomLocationsiminus[iCharge][j + 2]);
                    if (temp > d_Length[j] / 2)
                      {
                        // pcout<<iCharge<<" "<<j<<"Before: "<<temp;
                        temp -= d_Length[j];
                        temp = -temp;
                        // pcout<<" After: "<<temp<<std::endl;
                      }
                    else if (temp < -d_Length[j] / 2)
                      {
                        // pcout<<iCharge<<" "<<j<<"Before: "<<temp;
                        temp += d_Length[j];
                        temp = -temp;
                        // pcout<<" After: "<<temp<<std::endl;
                      }
                    tangent[count] = temp;
                    count++;
                  }
              }
          }
        ReturnNormedVector(tangent, d_countrelaxationFlags);
      }
  }

  void
  nudgedElasticBandClass::ReturnNormedVector(std::vector<double> &v, int len)
  {
    int    i;
    double norm = 0.0000;
    for (i = 0; i < len; i++)
      {
        norm = norm + v[i] * v[i];
      }
    norm = sqrt(norm);
    // pcout<<"Norm: "<<norm<<std::endl;
    AssertThrow(
      norm > 0.000000000001,
      ExcMessage(
        "DFT-FE Error: cordinates have 0 displacement between images"));
    for (i = 0; i < len; i++)
      {
        v[i] = v[i] / norm;
      }
  }


  void
  nudgedElasticBandClass::CalculateSpringForce(int                  image,
                                               std::vector<double> &ForceSpring,
                                               std::vector<double>  tangent)
  {
    unsigned int count        = 0;
    double       innerproduct = 0.0;
    if (image != 0 && image != d_numberOfImages - 1)
      {
        double                           norm1 = 0.0;
        double                           norm2 = 0.0;
        std::vector<double>              v1(d_countrelaxationFlags, 0.0);
        std::vector<double>              v2(d_countrelaxationFlags, 0.0);
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiminus,
          atomLocationsiplus;
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiminus =
          (d_dftfeWrapper[image - 1])->getAtomPositionsCart();
        atomLocationsiplus =
          (d_dftfeWrapper[image + 1])->getAtomPositionsCart();
        int count = 0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    v1[count] = std::fabs(atomLocationsiplus[iCharge][j + 2] -
                                          atomLocationsi[iCharge][j + 2]);
                    v2[count] = std::fabs(atomLocationsiminus[iCharge][j + 2] -
                                          atomLocationsi[iCharge][j + 2]);

                    if (d_Length[j] / 2 <= v1[count])
                      {
                        // pcout<<"Before: "<<v1[count];
                        v1[count] -= d_Length[j];
                        // pcout<<" After: "<<v1[count]<<std::endl;
                      }

                    if (d_Length[j] / 2 <= v2[count])
                      {
                        // pcout<<"Before: "<<v2[count];
                        v2[count] -= d_Length[j];
                        // pcout<<" After: "<<v2[count]<<std::endl;
                      }
                    count++;
                  }
              }
          }
        LNorm(norm1, v1, 2, d_countrelaxationFlags);
        LNorm(norm2, v2, 2, d_countrelaxationFlags);

        double kplus, kminus, k;
        CalculateSpringConstant(image + 1, kplus);
        CalculateSpringConstant(image, k);
        CalculateSpringConstant(image - 1, kminus);
        innerproduct = 0.5 * (kplus + k) * norm1 - 0.5 * (k + kminus) * norm2;
      }
    else if (image == 0)
      {
        double                           norm1 = 0.0;
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiminus,
          atomLocationsiplus;
        std::vector<double> v1(d_countrelaxationFlags, 0.0);
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiplus =
          (d_dftfeWrapper[image + 1])->getAtomPositionsCart();
        int count = 0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    v1[count] = std::fabs(atomLocationsiplus[iCharge][j + 2] -
                                          atomLocationsi[iCharge][j + 2]);
                    if (d_Length[j] / 2 <= v1[count])
                      {
                        // pcout<<"Before: "<<v1[count];
                        v1[count] -= d_Length[j];
                        // pcout<<" After: "<<v1[count]<<std::endl;
                      }
                    count++;
                  }
              }
          }
        LNorm(norm1, v1, 2, d_countrelaxationFlags);
        double k, kplus;
        CalculateSpringConstant(image + 1, kplus);
        CalculateSpringConstant(image, k);
        innerproduct = 0.5 * (kplus + k) * norm1;
      }
    else if (image == d_numberOfImages - 1)
      {
        double                           norm2 = 0.0;
        std::vector<std::vector<double>> atomLocationsi, atomLocationsiminus,
          atomLocationsplus;
        std::vector<double> v2(d_countrelaxationFlags, 0.0);
        atomLocationsi = (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsiminus =
          (d_dftfeWrapper[image - 1])->getAtomPositionsCart();
        int count = 0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    v2[count] = std::fabs(atomLocationsiminus[iCharge][j + 2] -
                                          atomLocationsi[iCharge][j + 2]);
                    if (d_Length[j] / 2 <= v2[count])
                      {
                        // pcout<<"Before: "<<v2[count];
                        v2[count] -= d_Length[j];
                        // pcout<<" After: "<<v2[count]<<std::endl;
                      }
                    count++;
                  }
              }
          }
        LNorm(norm2, v2, 2, d_countrelaxationFlags);
        double k, kminus;
        CalculateSpringConstant(image, k);
        CalculateSpringConstant(image - 1, kminus);
        innerproduct = -0.5 * (k + kminus) * norm2;
      }
    pcout << "Spring Force on image: " << image << std::endl;
    for (count = 0; count < d_countrelaxationFlags; count++)
      {
        ForceSpring[count] = innerproduct * tangent[count];
        // pcout<<ForceSpring[count]<<"  "<<tangent[count]<<std::endl;
      }
  }


  void
  nudgedElasticBandClass::CalculateForceparallel(
    int                  image,
    std::vector<double> &Forceparallel,
    std::vector<double>  tangent)
  {
    if (true)
      {
        std::vector<std::vector<double>> forceonAtoms =
          (d_dftfeWrapper[image])->getForcesAtoms();
        double       Innerproduct = 0.0;
        unsigned int count        = 0;

        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            for (int j = 0; j < 3; j++)
              {
                if (d_relaxationFlags[3 * iCharge + j] == 1)
                  {
                    Innerproduct = Innerproduct - forceonAtoms[3 * iCharge][j] *
                                                    tangent[count];
                    count++;
                  }
              }
          }
        for (count = 0; count < d_countrelaxationFlags; count++)
          {
            Forceparallel[count] = Innerproduct * tangent[count];
          }
      }
  }


  void
  nudgedElasticBandClass::CalculateForceperpendicular(
    int                  image,
    std::vector<double> &Forceperpendicular,
    std::vector<double>  Forceparallel,
    std::vector<double>  tangent)
  {
    std::vector<std::vector<double>> forceonAtoms =
      (d_dftfeWrapper[image])->getForcesAtoms();
    unsigned int count = 0;

    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        for (int j = 0; j < 3; j++)
          {
            if (d_relaxationFlags[3 * iCharge + j] == 1)
              {
                Forceperpendicular[count] =
                  -forceonAtoms[3 * iCharge][j] - Forceparallel[count];
                count++;
              }
          }
      }
  }


  int
  nudgedElasticBandClass::run()
  {
    nonLinearSolver::ReturnValueType solverReturn =
      d_nonLinearSolverPtr->solve(*this,
                                  d_restartFilesPath + "/ionRelax.chk",
                                  false);

    if (solverReturn == nonLinearSolver::SUCCESS &&
        d_dftPtr->getParametersObject().verbosity >= 1)
      {
        pcout
          << " ...Ion force relaxation completed as maximum force magnitude is less than FORCE TOL: "
          << d_dftPtr->getParametersObject().forceRelaxTol
          << ", total number of ion position updates: " << d_totalUpdateCalls
          << std::endl;
        pcout << "--------------Final Ground State Results-------------"
              << std::endl;
        for (int i = 0; i < d_numberOfImages; i++)
          {
            pcout << "Free Energy of Image: " << i + 1 << "  = "
                  << (d_dftfeWrapper[i])->getDFTFreeEnergy() << std::endl;
          }
        pcout << "--------------Final Error Results(Ha/bohr)-------------"
              << std::endl;
        for (int i = 0; i < d_numberOfImages; i++)
          {
            d_NEBImageno = i;
            std::vector<double> tangent(d_countrelaxationFlags, 0.0);
            std::vector<double> Forceparallel(d_countrelaxationFlags, 0.0);
            std::vector<double> Forceperpendicular(d_countrelaxationFlags, 0.0);
            double              Force = 0.0;

            CalculatePathTangent(d_NEBImageno, tangent);
            CalculateForceparallel(d_NEBImageno, Forceparallel, tangent);
            CalculateForceperpendicular(d_NEBImageno,
                                        Forceperpendicular,
                                        Forceparallel,
                                        tangent);
            LNorm(Force, Forceperpendicular, 0, d_countrelaxationFlags);
            pcout << "Error of Image: " << i + 1 << "  = " << Force
                  << " Ha/bohr" << std::endl;
          }
        return d_totalUpdateCalls;
      }

    else if (solverReturn == nonLinearSolver::FAILURE)
      {
        pcout << " ...Ion force relaxation failed " << std::endl;
        d_totalUpdateCalls = -1;
      }
    else if (solverReturn == nonLinearSolver::MAX_ITER_REACHED)
      {
        pcout << " ...Maximum iterations reached " << std::endl;
        d_totalUpdateCalls = -2;
      }



    pcout << "--------------Ground State Results-------------" << std::endl;
    for (int i = 0; i < d_numberOfImages; i++)
      {
        pcout << "Free Energy of Image: " << i + 1 << "  = "
              << (d_dftfeWrapper[i])->getDFTFreeEnergy() << std::endl;
      }
    pcout << "--------------Error Results(Ha/bohr)-------------" << std::endl;
    for (int i = 0; i < d_numberOfImages; i++)
      {
        d_NEBImageno = i;
        std::vector<double> tangent(d_countrelaxationFlags, 0.0);
        std::vector<double> Forceparallel(d_countrelaxationFlags, 0.0);
        std::vector<double> Forceperpendicular(d_countrelaxationFlags, 0.0);
        double              Force = 0.0;

        CalculatePathTangent(d_NEBImageno, tangent);
        CalculateForceparallel(d_NEBImageno, Forceparallel, tangent);
        CalculateForceperpendicular(d_NEBImageno,
                                    Forceperpendicular,
                                    Forceparallel,
                                    tangent);
        LNorm(Force, Forceperpendicular, 0, d_countrelaxationFlags);
        pcout << "Error of Image: " << i + 1 << "  = " << Force << " Ha/bohr"
              << std::endl;
      }
    return d_totalUpdateCalls;
  }



  void
  nudgedElasticBandClass::LNorm(double &            norm,
                                std::vector<double> v,
                                int                 L,
                                int                 len)
  {
    norm = 0.0;
    if (L == 2)
      {
        for (int i = 0; i < len; i++)
          norm = norm + v[i] * v[i];
        norm = sqrt(norm);
      }
    if (L == 0)
      {
        norm = -1;
        for (int i = 0; i < len; i++)
          norm = std::max(norm, fabs(v[i]));
      }
    if (L == 1)
      {
        norm = 0.0;
        for (int i = 0; i < len; i++)
          norm = norm + std::fabs(v[i]);
      }
  }


  void
  nudgedElasticBandClass::gradient(std::vector<double> &gradient)
  {
    gradient.clear();
    std::vector<int> flagmultiplier(d_numberOfImages, 1);
    bool             flag = false;
    pcout << "    "
          << " Image No "
          << "    "
          << "Internal Energy in Ha"
          << "    "
          << "Free Energy in Ha"
          << "    " << std::endl;
    for (int i = 0; i < d_numberOfImages; i++)
      {
        double FreeEnergy = (d_dftfeWrapper[i])->getDFTFreeEnergy();
        double InternalEnergy =
          (d_dftfeWrapper[i])->getDFTFreeEnergy() +
          (d_dftfeWrapper[i])->getElectronicEntropicEnergy();
        pcout << "    " << i << "    " << InternalEnergy << "    " << FreeEnergy
              << "    " << std::endl;
      }
    pcout << "************Error in gradient***************" << std::endl;
    double Force = 0.0;
    ImageError(0, Force);
    d_ImageError[0] = Force;

    for (int i = 1; i < d_numberOfImages - 1; i++)
      {
        d_NEBImageno = i;
        ImageError(d_NEBImageno, Force);
        d_ImageError[i] = Force;

        pcout << "The Force on image no. " << d_NEBImageno << " is " << Force
              << " in Ha/bohr" << std::endl;
        if (Force < 0.95 * d_optimizertolerance && d_imageFreeze)
          flagmultiplier[i] = 0;
        if (Force <= d_optimizertolerance)
          {
            flag = true;
            pcout << "Image no. " << i + 1 << " has converged with value of"
                  << Force << " vs tolerance of" << d_optimizertolerance
                  << std::endl;
          }
        else if (Force > d_optimizertolerance)
          flag = false;
      }

    ImageError(d_numberOfImages - 1, Force);
    d_ImageError[d_numberOfImages - 1] = Force;

    if (flag == true)
      pcout << "Optimization Criteria Met!!" << std::endl;

    pcout << "Image No. Norm of F_per   Norm of Spring Force" << std::endl;
    for (int image = 1; image < d_numberOfImages - 1; image++)
      {
        std::vector<double> tangent(d_countrelaxationFlags, 0.0);
        std::vector<double> Forceparallel(d_countrelaxationFlags, 0.0);
        std::vector<double> Forceperpendicular(d_countrelaxationFlags, 0.0);
        std::vector<double> SpringForce(d_countrelaxationFlags, 0.0);
        std::vector<double> ForceonImage(d_countrelaxationFlags, 0.0);
        CalculatePathTangent(image, tangent);
        CalculateForceparallel(image, Forceparallel, tangent);
        CalculateForceperpendicular(image,
                                    Forceperpendicular,
                                    Forceparallel,
                                    tangent);
        CalculateSpringForce(image, SpringForce, tangent);
        CalculateForceonImage(Forceperpendicular, SpringForce, ForceonImage);
        double F_spring = 0.0;
        double F_per    = 0.0;
        LNorm(F_per, Forceperpendicular, 0, d_countrelaxationFlags);
        LNorm(F_spring, SpringForce, 0, d_countrelaxationFlags);
        pcout << image << "  " << F_per << "  " << F_spring << std::endl;

        // pcout<<"Before start of optimization Image Force: "<<Force<<"
        // Ha/Bohr" <<std:endl;

        for (int i = 0; i < d_countrelaxationFlags; i++)
          {
            if (flag == false)
              gradient.push_back(-ForceonImage[i] * flagmultiplier[image]);
            else
              gradient.push_back(-Forceperpendicular[i] *
                                 flagmultiplier[image]);
          }
      }
    pcout << "##Frozen images are: ";
    for (int image = 1; image < d_numberOfImages - 1; image++)
      {
        if (flagmultiplier[image] == 0)
          pcout << " " << image << " ";
      }


    d_maximumAtomForceToBeRelaxed = -1.0;

    for (unsigned int i = 0; i < gradient.size(); ++i)
      {
        const double temp = std::sqrt(gradient[i] * gradient[i]);
        // pcout<<i<<"   "<<temp<<std::endl;
        if (temp > d_maximumAtomForceToBeRelaxed)
          d_maximumAtomForceToBeRelaxed = temp;
      }
    pcout << std::endl
          << "Maximum Force " << d_maximumAtomForceToBeRelaxed << "in Ha/bohr"
          << std::endl;
  }


  void
  nudgedElasticBandClass::CalculateForceonImage(
    std::vector<double>  Forceperpendicular,
    std::vector<double>  SpringForce,
    std::vector<double> &ForceonImage)
  {
    unsigned int count = 0;
    // pcout<<"Forces on Image "<<d_NEBImageno<<std::endl;
    for (count = 0; count < d_countrelaxationFlags; count++)
      {
        if (d_NEBImageno > 0 && d_NEBImageno < d_numberOfImages)
          ForceonImage[count] = SpringForce[count] + Forceperpendicular[count];
        else
          ForceonImage[count] = Forceperpendicular[count];
        // pcout<<count<<"  "<<SpringForce[count]<<"
        // "<<Forceperpendicular[count] <<"  "<<ForceonImage[count]<<std::endl;
      }
    // pcout<<"****************************"<<std::endl;
  }


  void
  nudgedElasticBandClass::update(const std::vector<double> &solution,
                                 const bool                 computeForces,
                                 const bool useSingleAtomSolutionsInitialGuess)
  {
    std::vector<Tensor<1, 3, double>> globalAtomsDisplacements(
      d_numberGlobalCharges);
    d_forceOnImages.clear();
    for (int image = 1; image < d_numberOfImages - 1; image++)
      {
        int multiplier = 1;
        pcout << "Update called for image: " << image << std::endl;

        if (d_ImageError[image] < 0.95 * d_optimizertolerance && d_imageFreeze)
          {
            multiplier = 0;
            pcout << "!!Frozen image " << image
                  << " with Image force: " << d_ImageError[image] << std::endl;
          }
        MPI_Bcast(&multiplier, 1, MPI_INT, 0, d_mpiCommParent);
        int count = 0;
        pcout << "###Displacements for image: " << image << std::endl;
        for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                if (d_this_mpi_process == 0)
                  {
                    globalAtomsDisplacements[i][j] = 0.0;
                    if (d_relaxationFlags[3 * i + j] == 1)
                      {
                        globalAtomsDisplacements[i][j] =
                          solution[(image - 1) * d_countrelaxationFlags +
                                   count] *
                          multiplier;
                        if (globalAtomsDisplacements[i][j] > 0.4)
                          globalAtomsDisplacements[i][j] = 0.4;
                        else if (globalAtomsDisplacements[i][j] < -0.4)
                          globalAtomsDisplacements[i][j] = -0.4;

                        count++;
                      }
                  }
              }
            pcout << globalAtomsDisplacements[i][0] << " "
                  << globalAtomsDisplacements[i][1] << " "
                  << globalAtomsDisplacements[i][2] << std::endl;
            MPI_Bcast(&(globalAtomsDisplacements[i][0]),
                      3,
                      MPI_DOUBLE,
                      0,
                      d_mpiCommParent);
          }



        /*double factor;
        if (d_maximumAtomForceToBeRelaxed >= 1e-03)
          factor = 1.30; // Modified
        else if (d_maximumAtomForceToBeRelaxed < 1e-03 &&
                 d_maximumAtomForceToBeRelaxed >= 1e-04)
          factor = 1.25;
        else if (d_maximumAtomForceToBeRelaxed < 1e-04)
          factor = 1.15;
        // MPI_Barrier required here...
        factor = 1.0;*/
        if (multiplier == 1)
          {
            /*MPI_Barrier(d_mpiCommParent);
            (d_dftfeWrapper[image]->getDftfeBasePtr())
              ->updateAtomPositionsAndMoveMesh(
                globalAtomsDisplacements,
                factor,
                useSingleAtomSolutionsInitialGuess); */

            pcout << "--Positions of image: " << image << " updated--"
                  << std::endl;
            MPI_Barrier(d_mpiCommParent);
            std::tuple<double, bool, double> groundStateOutput =
              (d_dftfeWrapper[image])->computeDFTFreeEnergy(true, false);
            if (!std::get<1>(groundStateOutput))
              pcout << " NEB Warning!!: Ground State of Image: " << d_NEBImageno
                    << " did not converge" << std::endl;
          }
      }
    for (int image = 0; image < d_numberOfImages; image++)
      {
        double Force = 0.0;
        d_NEBImageno = image;
        ImageError(d_NEBImageno, Force);
        d_forceOnImages.push_back(Force);
      }
    d_totalUpdateCalls += 1;
  }



  void
  nudgedElasticBandClass::precondition(std::vector<double> &      s,
                                       const std::vector<double> &gradient)
  {
    s.clear();
    s.resize(getNumberUnknowns() * getNumberUnknowns(), 0.0);
    for (auto i = 0; i < getNumberUnknowns(); ++i)
      {
        s[i + i * getNumberUnknowns()] = 1.0;
      }
  }



  void
  nudgedElasticBandClass::solution(std::vector<double> &solution)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
    /*solution.clear();
    pcout << "The size of solution vector is: " << solution.size() << std::endl;
    pcout << "Size of relaxation flags: " << d_relaxationFlags.size()
          << std::endl;
    for (int image = 1; image < d_numberOfImages - 1; image++)
      {
        pcout << "Image no.: " << image << std::endl;
        std::vector<std::vector<double>> atomLocations, atomLocationsInitial;
        atomLocations =
          (d_dftfeWrapper[image])->getAtomPositionsCart();
        atomLocationsInitial = d_atomLocationsInitial[image];
        pcout << "AtomLocation size  " << atomLocations.size() << " "
              << atomLocationsInitial.size() << std::endl;
        for (int i = 0; i < d_numberGlobalCharges; ++i)
          {
            for (int j = 0; j < 3; ++j)
              {
                if (d_relaxationFlags[3 * i + j] == 1)
                  {
                    solution.push_back(atomLocations[i][j + 2] -
                                       atomLocationsInitial[i][j + 2]);
                  }
              }
          }
      }
    // pcout<<"The size of solution vector is: "<<solution.size()<<std::endl; */
  }



  void
  nudgedElasticBandClass::save()
  {
    if (!d_dftPtr->getParametersObject().reproducible_output)
      {
        std::string savePath =
          d_restartFilesPath + "/Step" + std::to_string(d_totalUpdateCalls);
        if (d_this_mpi_process == 0)
          mkdir(savePath.c_str(), ACCESSPERMS);

        std::vector<std::vector<double>> forceData(1,
                                                   std::vector<double>(1, 0.0));
        forceData[0][0] = d_maximumAtomForceToBeRelaxed;
        dftUtils::writeDataIntoFile(forceData,
                                    savePath + "/maxForce.chk",
                                    d_mpiCommParent);
        for(int i = 0; i < d_numberOfImages; i++)
          {
            d_dftfeWrapper[i]->writeDomainAndAtomCoordinatesFloatingCharges(savePath + "/Image"+std::to_string(i));
          }        
        d_nonLinearSolverPtr->save(savePath + "/ionRelax.chk");
        std::vector<std::vector<double>> tmpData(1,
                                                 std::vector<double>(1, 0.0));
        tmpData[0][0] = d_totalUpdateCalls;
        dftUtils::writeDataIntoFile(tmpData,
                                    d_restartFilesPath + "/step.chk",
                                    d_mpiCommParent);
      }
  }



  std::vector<unsigned int>
  nudgedElasticBandClass::getUnknownCountFlag() const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  void
  nudgedElasticBandClass::value(std::vector<double> &functionValue)
  {
    functionValue.clear();

    int midImage = d_numberOfImages / 2;
    functionValue.push_back((d_dftfeWrapper[midImage])->getDFTFreeEnergy());



    // functionValue.push_back( dftPtr[3]->getInternalEnergy());
  }



  unsigned int
  nudgedElasticBandClass::getNumberUnknowns() const
  {
    return (d_countrelaxationFlags * (d_numberOfImages - 2));
  }



  void
  nudgedElasticBandClass::CalculatePathLength(double &length) const
  {
    length = 0.0;
    std::vector<std::vector<double>> atomLocations, atomLocationsInitial;

    for (int i = 0; i < d_numberOfImages - 1; i++)
      {
        atomLocations        = (d_dftfeWrapper[i + 1])->getAtomPositionsCart();
        atomLocationsInitial = (d_dftfeWrapper[i])->getAtomPositionsCart();
        double tempx, tempy, tempz, temp;
        temp = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            tempx = std::fabs(atomLocations[iCharge][2] -
                              atomLocationsInitial[iCharge][2]);
            tempy = std::fabs(atomLocations[iCharge][3] -
                              atomLocationsInitial[iCharge][3]);
            tempz = std::fabs(atomLocations[iCharge][4] -
                              atomLocationsInitial[iCharge][4]);
            if (d_Length[0] / 2 <= tempx)
              tempx -= d_Length[0];
            if (d_Length[1] / 2 <= tempy)
              tempy -= d_Length[1];
            if (d_Length[2] / 2 <= tempz)
              tempz -= d_Length[2];
            temp += tempx * tempx + tempy * tempy + tempz * tempz;
          }
        length += std::sqrt(temp);
      }
  }

  void
  nudgedElasticBandClass::CalculateSpringConstant(int     NEBImage,
                                                  double &SpringConstant)
  {
    SpringConstant = 0.0;
    double Emin, ksum, kdiff, deltaE, Emax;
    ksum  = d_kmax + d_kmin;
    kdiff = d_kmax - d_kmin;
    double Ei;
    Emax = -5000000;
    Emin = 500;
    for (int image = 0; image < d_numberOfImages - 1; image++)
      {
        Emax = std::max(Emax, (d_dftfeWrapper[image])->getDFTFreeEnergy());
        Emin = std::min(Emin, (d_dftfeWrapper[image])->getDFTFreeEnergy());
      }
    deltaE = Emax - Emin;

    Ei = (d_dftfeWrapper[NEBImage])->getDFTFreeEnergy();



    SpringConstant =
      0.5 * (ksum - kdiff * std::cos(pi * (Ei - Emin) / (deltaE)));

    pcout << "Image number " << NEBImage
          << " Spring Constant: " << SpringConstant << std::endl;
  }


  void
  nudgedElasticBandClass::WriteRestartFiles(int step)
  {
    /*std::vector<std::vector<double>> stepIndexData(1, std::vector<double>(1,
    0)); stepIndexData[0][0] = double(step); pcout<<"Writing restart files for
    step: "<<step<<std::endl; std::string Folder = "nebRestart/Step";
    std::string tempfolder = Folder +  std::to_string(step);
      mkdir(tempfolder.c_str(), ACCESSPERMS);
      Folder = "nebRestart";
      std::string newFolder3 = Folder + "/" + "step.chk";
      dftUtils::writeDataIntoFile(stepIndexData, newFolder3,d_mpiCommParent);
      std::string cordFolder = tempfolder + "/";
      for(int i=0; i < d_numberOfImages; i++)
      {
        dftPtr[i]->NEBwriteDomainAndAtomCoordinates(cordFolder,std::to_string(i));
      }
      dftUtils::writeDataIntoFile(stepIndexData, newFolder3,d_mpiCommParent);
          if(d_this_mpi_process == 0)
          {
            std::ofstream outfile;
            outfile.open(tempfolder+"/coordinates.inp", std::ios_base::app);
            for(int i=0; i < d_numberOfImages; i++)
            {
              std::vector<std::vector<double>> atomLocations;
              std::string coordinatesfolder =
    tempfolder+"/coordinates.inp"+std::to_string(i);
              dftUtils::readFile(5,atomLocations,coordinatesfolder);
              for(int iCharge = 0; iCharge < numberGlobalCharges; iCharge++)
              {
                outfile<<atomLocations[iCharge][0]<<"
    "<<atomLocations[iCharge][1]<< "  "<<atomLocations[iCharge][2]
                        <<"  "<<atomLocations[iCharge][3]<<"
    "<<atomLocations[iCharge][4]<<std::endl;
              }
            }
            outfile.close();
          }
    */
  }

  void
  nudgedElasticBandClass::ImageError(int image, double &Force)
  {
    Force = 0.0;
    std::vector<double> tangent(d_countrelaxationFlags, 0.0);
    std::vector<double> Forceparallel(d_countrelaxationFlags, 0.0);
    std::vector<double> Forceperpendicular(d_countrelaxationFlags, 0.0);
    CalculatePathTangent(image, tangent);
    CalculateForceparallel(image, Forceparallel, tangent);
    CalculateForceperpendicular(image,
                                Forceperpendicular,
                                Forceparallel,
                                tangent);
    LNorm(Force, Forceperpendicular, 0, d_countrelaxationFlags);
  }

  bool
  nudgedElasticBandClass::isConverged() const
  {
    bool flag = true;
    pcout
      << std::endl
      << "-------------------------------------------------------------------------------"
      << std::endl;
    pcout << " --------------------NEB Attempt Completed "
          << "---------------------------------------" << std::endl;
    pcout << "    "
          << " Image No "
          << "    "
          << "Force perpendicular in Ha/bohr"
          << "    "
          << "Free Energy in Ha"
          << "    " << std::endl;


    for (int i = 0; i < d_numberOfImages; i++)
      {
        double Force  = d_forceOnImages[i];
        double Energy = (d_dftfeWrapper[i])->getDFTFreeEnergy();
        pcout << "    " << i << "    " << Force << "    " << Energy << "    "
              << std::endl;
        if (Force > d_optimizertolerance && i > 0 && i < d_numberOfImages - 1)
          {
            flag = false;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    double Length = 0.0;
    CalculatePathLength(Length);
    pcout << std::endl << "--Path Length: " << Length << " Bohr" << std::endl;
    pcout
      << std::endl
      << "-------------------------------------------------------------------------------"
      << std::endl;

    return flag;
  }


  void
  nudgedElasticBandClass::init()
  {
    d_solverRestart = d_isRestart;
    if (d_dftPtr->getParametersObject().ionOptSolver == "BFGS")
      d_solver = 0;
    else if (d_dftPtr->getParametersObject().ionOptSolver == "LBFGS")
      d_solver = 1;
    else if (d_dftPtr->getParametersObject().ionOptSolver == "CGPRP")
      d_solver = 2;


    double step_time;


    if (d_dftPtr->getParametersObject().ionRelaxFlagsFile != "")
      {
        std::vector<std::vector<int>>    tempRelaxFlagsData;
        std::vector<std::vector<double>> tempForceData;
        dftUtils::readRelaxationFlagsFile(
          6,
          tempRelaxFlagsData,
          tempForceData,
          d_dftPtr->getParametersObject().ionRelaxFlagsFile);
        AssertThrow(tempRelaxFlagsData.size() == d_numberGlobalCharges,
                    ExcMessage(
                      "Incorrect number of entries in relaxationFlags file"));
        d_relaxationFlags.clear();
        d_externalForceOnAtom.clear();



        for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                d_relaxationFlags.push_back(tempRelaxFlagsData[i][j]);
                d_externalForceOnAtom.push_back(tempForceData[i][j]);
              }
          }
        // print relaxation flags
        pcout << " --------------Ion force relaxation flags----------------"
              << std::endl;
        for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
          {
            pcout << tempRelaxFlagsData[i][0] << "  "
                  << tempRelaxFlagsData[i][1] << "  "
                  << tempRelaxFlagsData[i][2] << std::endl;
          }
        pcout << " --------------------------------------------------"
              << std::endl;
      }
    else
      {
        d_relaxationFlags.clear();
        d_externalForceOnAtom.clear();
        for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                d_relaxationFlags.push_back(1.0);
                d_externalForceOnAtom.push_back(0.0);
              }
          }
      }
    d_countrelaxationFlags = 0;
    for (int i = 0; i < d_relaxationFlags.size(); i++)
      {
        if (d_relaxationFlags[i] == 1)
          d_countrelaxationFlags++;
      }


    d_totalUpdateCalls = 0;
    if (Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      mkdir(d_restartFilesPath.c_str(), ACCESSPERMS);
    std::vector<std::vector<double>> ionOptData(2 + d_numberGlobalCharges * 3,
                                                std::vector<double>(1, 0.0));
    ionOptData[0][0] = d_solver;
    ionOptData[1][0] =
      d_dftPtr->getParametersObject().usePreconditioner ? 1 : 0;
    for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            ionOptData[i * 3 + j + 2][0] = d_relaxationFlags[i * 3 + j];
          }
      }
    /*if (!d_dftPtr->getParametersObject().reproducible_output)
      dftUtils::writeDataIntoFile(ionOptData,
                                  d_restartFilesPath + "/ionOpt.dat",
                                  mpi_communicator);*/
    d_ImageError.resize(d_numberOfImages);
    double Force;
    MPI_Barrier(d_mpiCommParent);
    step_time = MPI_Wtime();

    for (int i = 0; i < d_numberOfImages; i++)
      {
        d_NEBImageno = i;
        auto groundState =
          (d_dftfeWrapper[d_NEBImageno])->computeDFTFreeEnergy(true, false);
        if (!std::get<1>(groundState))
          pcout << " NEB Warning!!: Ground State of Image: " << d_NEBImageno
                << " did not converge" << std::endl;
        pcout << "##Completed initial GS of image: " << d_NEBImageno
              << std::endl;
      }
    bool flag = true;
    pcout
      << std::endl
      << "-------------------------------------------------------------------------------"
      << std::endl;
    pcout << " --------------------Initial NEB Data "
          << "---------------------------------------" << std::endl;
    pcout << "    "
          << " Image No "
          << "    "
          << "Force perpendicular in Ha/bohr"
          << "    "
          << "Internal Energy in Ha"
          << "    " << std::endl;

    d_forceOnImages.clear();

    int count = 0;
    for (int i = 0; i < d_numberOfImages; i++)
      {
        d_NEBImageno = i;
        std::vector<std::vector<double>> atomLocations;
        atomLocations = (d_dftfeWrapper[i])->getAtomPositionsCart();
        Force         = 0.0;
        ImageError(d_NEBImageno, Force);
        double Energy = (d_dftfeWrapper[i])->getDFTFreeEnergy();
        pcout << "    " << i << "    " << Force << "    " << Energy << "    "
              << std::endl;
        d_forceOnImages.push_back(Force);
        if (Force > Forcecutoff && i > 0 && i < d_numberOfImages - 1)
          {
            flag = false;
          }
      }
    MPI_Barrier(d_mpiCommParent);
    double Length = 0.0;
    CalculatePathLength(Length);
    pcout << std::endl << "--Path Length: " << Length << " Bohr" << std::endl;
    step_time = MPI_Wtime() - step_time;
    pcout << "Time taken for initial dft solve of all images: " << step_time
          << std::endl;
    pcout
      << std::endl
      << "-------------------------------------------------------------------------------"
      << std::endl;



    if (d_solver == 0)
      d_nonLinearSolverPtr = std::make_unique<BFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().bfgsStepMethod == "RFO",
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().verbosity,
        d_mpiCommParent,
        d_dftPtr->getParametersObject().maxIonUpdateStep,
        true);
    else if (d_solver == 1)
      d_nonLinearSolverPtr = std::make_unique<LBFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().maxIonUpdateStep,
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().lbfgsNumPastSteps,
        d_dftPtr->getParametersObject().verbosity,
        d_mpiCommParent,
        true);
    else
      d_nonLinearSolverPtr = std::make_unique<cgPRPNonLinearSolver>(
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().verbosity,
        d_mpiCommParent,
        1e-4,
        d_dftPtr->getParametersObject().maxLineSearchIterCGPRP,
        0.8,
        d_dftPtr->getParametersObject().maxIonUpdateStep,
        true);
    // print relaxation flags
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        pcout << " --------------Ion force relaxation flags----------------"
              << std::endl;
        for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
          {
            pcout << d_relaxationFlags[i * 3] << "  "
                  << d_relaxationFlags[i * 3 + 1] << "  "
                  << d_relaxationFlags[i * 3 + 2] << std::endl;
          }
        pcout << " --------------------------------------------------------"
              << std::endl;
        pcout << " Total No. of relaxation flags: " << d_countrelaxationFlags
              << std::endl;
        pcout << " --------------------------------------------------"
              << std::endl;
      }
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        if (d_solver == 0)
          {
            pcout << "   ---Non-linear BFGS Parameters-----------  "
                  << std::endl;
            pcout << "      stopping tol: " << d_optimizertolerance
                  << std::endl;

            pcout << "      maxIter: " << d_maximumNEBIteration << std::endl;

            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;

            pcout << "      step method: "
                  << d_dftPtr->getParametersObject().bfgsStepMethod
                  << std::endl;

            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;


            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 1)
          {
            pcout << "   ---Non-linear LBFGS Parameters----------  "
                  << std::endl;
            pcout << "      stopping tol: " << d_optimizertolerance
                  << std::endl;
            pcout << "      maxIter: " << d_maximumNEBIteration << std::endl;
            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;
            pcout << "      lbfgs history: "
                  << d_dftPtr->getParametersObject().lbfgsNumPastSteps
                  << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 2)
          {
            pcout << "   ---Non-linear CG Parameters--------------  "
                  << std::endl;
            pcout << "      stopping tol: " << d_optimizertolerance
                  << std::endl;
            pcout << "      maxIter: " << d_maximumNEBIteration << std::endl;
            pcout << "      lineSearch tol: " << 1e-4 << std::endl;
            pcout << "      lineSearch maxIter: "
                  << d_dftPtr->getParametersObject().maxLineSearchIterCGPRP
                  << std::endl;
            pcout << "      lineSearch damping parameter: " << 0.8 << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_isRestart)
          {
            if (d_solver == 2)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
        else
          {
            if (d_solver == 2)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
      }
    d_isRestart = false;
  }


} // namespace dftfe
