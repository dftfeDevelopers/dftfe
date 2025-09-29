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
//============================================================================================================================================
//============================================================================================================================================
//    This is the source file for generating Monkhorst-Pack Brillouin zone (BZ)
//    grid and creating the irreducible BZ
//	                             Only relevant for calculations using multiple
// k-points
//
//                                        Author : Krishnendu Ghosh,
//                                        krisg@umich.edu
//
//============================================================================================================================================
//============================================================================================================================================
//
#include <dftParameters.h>
#include <dft.h>
#include <fileReaders.h>

//
namespace dftfe
{
  namespace internaldft
  {
    //============================================================================================================================================
    //============================================================================================================================================
    //			           Just a quick snippet to compute cross product of two
    // vectors
    //============================================================================================================================================
    //============================================================================================================================================
    std::vector<double>
    cross_product(const std::vector<double> &a, const std::vector<double> &b)
    {
      std::vector<double> crossProduct(a.size(), 0.0);
      crossProduct[0] = a[1] * b[2] - a[2] * b[1];
      crossProduct[1] = a[2] * b[0] - a[0] * b[2];
      crossProduct[2] = a[0] * b[1] - a[1] * b[0];
      return crossProduct;
    }
    //============================================================================================================================================
    //============================================================================================================================================
    //			          Following routine computes the reciprocal lattice vectors
    // for the given system
    //============================================================================================================================================
    //============================================================================================================================================
    std::vector<std::vector<double>>
    getReciprocalLatticeVectors(
      const std::vector<std::vector<double>> &latticeVectors,
      const std::array<dftfe::uInt, 3>        periodicity)
    {
      std::vector<std::vector<double>> reciprocalLatticeVectors(
        3, std::vector<double>(3, 0.0));
      dftfe::uInt                      periodicitySum = 0;
      std::vector<double>              cross(3, 0.0);
      std::vector<std::vector<double>> latticeVectorsToBeUsed;
      std::vector<dftfe::uInt>         latticeVectorsToBeUsedIndex;
      double                           scalarConst;
      std::vector<double>              unitVectorOutOfPlane(3, 0.0);
      //
      for (dftfe::uInt i = 0; i < 3; ++i)
        periodicitySum += periodicity[i];
      //
      switch (periodicitySum)
        {
          //=========================================================================================================================================
          case 3: //				      all directions periodic
            //==========================================================================================================================================
            for (dftfe::uInt i = 0; i < 2; ++i)
              {
                cross =
                  internaldft::cross_product(latticeVectors[i + 1],
                                             latticeVectors[3 - (2 * i + 1)]);
                scalarConst = latticeVectors[i][0] * cross[0] +
                              latticeVectors[i][1] * cross[1] +
                              latticeVectors[i][2] * cross[2];
                for (dftfe::uInt d = 0; d < 3; ++d)
                  reciprocalLatticeVectors[i][d] =
                    (2. * M_PI / scalarConst) * cross[d];
              }
            //
            cross =
              internaldft::cross_product(latticeVectors[0], latticeVectors[1]);
            scalarConst = latticeVectors[2][0] * cross[0] +
                          latticeVectors[2][1] * cross[1] +
                          latticeVectors[2][2] * cross[2];
            for (dftfe::uInt d = 0; d < 3; ++d)
              reciprocalLatticeVectors[2][d] =
                (2 * M_PI / scalarConst) * cross[d];
            break;
            //==========================================================================================================================================
          case 2: //				two directions periodic, one direction non-periodic
            //==========================================================================================================================================
            for (dftfe::uInt i = 0; i < 3; ++i)
              {
                if (periodicity[i] == 1)
                  {
                    latticeVectorsToBeUsed.push_back(latticeVectors[i]);
                    latticeVectorsToBeUsedIndex.push_back(i);
                  }
              }
            //
            cross = internaldft::cross_product(latticeVectorsToBeUsed[0],
                                               latticeVectorsToBeUsed[1]);
            for (dftfe::uInt d = 0; d < 3; ++d)
              unitVectorOutOfPlane[d] =
                cross[d] / (sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                                 cross[2] * cross[2]));
            //
            for (dftfe::uInt i = 0; i < 2; ++i)
              {
                cross =
                  internaldft::cross_product(latticeVectorsToBeUsed[1 - i],
                                             unitVectorOutOfPlane);
                scalarConst = latticeVectorsToBeUsed[i][0] * cross[0] +
                              latticeVectorsToBeUsed[i][1] * cross[1] +
                              latticeVectorsToBeUsed[i][2] * cross[2];
                for (dftfe::uInt d = 0; d < 3; ++d)
                  reciprocalLatticeVectors[latticeVectorsToBeUsedIndex[i]][d] =
                    (2. * M_PI / scalarConst) * cross[d];
              }
            break;
            //============================================================================================================================================
          case 1: //				two directions non-periodic, one direction periodic
            //============================================================================================================================================
            for (dftfe::uInt i = 0; i < 3; ++i)
              {
                if (periodicity[i] == 1)
                  {
                    const double scalarConst =
                      sqrt(latticeVectors[i][0] * latticeVectors[i][0] +
                           latticeVectors[i][1] * latticeVectors[i][1] +
                           latticeVectors[i][2] * latticeVectors[i][2]);
                    for (dftfe::uInt d = 0; d < 3; ++d)
                      reciprocalLatticeVectors[i][d] =
                        (2. * M_PI / scalarConst) * latticeVectors[i][d];
                  }
              }
        } // end switch
      //
      return reciprocalLatticeVectors;
      //
    }
  } // namespace internaldft

  //============================================================================================================================================
  //============================================================================================================================================
  //			           Following routine can read k-points supplied through
  // external
  // file 				Not required in general, as one can use MP grid samplings to
  // generate the k-grid
  //============================================================================================================================================
  //============================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::readkPointData()
  {
    const dftfe::Int                 numberColumnskPointDataFile = 4;
    std::vector<std::vector<double>> kPointData;
    char                             kPointRuleFile[256];
    strcpy(kPointRuleFile, d_dftParamsPtr->kPointDataFile.c_str());
    // sprintf(kPointRuleFile,
    //      "%s/data/kPointList/%s",
    //    DFTFE_PATH,
    //  d_dftParamsPtr->kPointDataFile.c_str());
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Reading data from file: " << kPointRuleFile << std::endl;
    dftUtils::readFile(numberColumnskPointDataFile, kPointData, kPointRuleFile);
    d_kPointCoordinates.clear();
    d_kPointWeights.clear();
    dftfe::uInt maxkPoints = kPointData.size();
    d_kPointCoordinates.resize(maxkPoints * 3, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    d_kPointCoordinatesFrac = d_kPointCoordinates;
    //
    const std::array<dftfe::uInt, 3> periodic = {d_dftParamsPtr->periodicX,
                                                 d_dftParamsPtr->periodicY,
                                                 d_dftParamsPtr->periodicZ};
    d_reciprocalLatticeVectors =
      internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors,
                                               periodic);
    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout
          << "-----------Reciprocal vectors along which the MP grid is to be generated-------------"
          << std::endl;
        for (dftfe::Int i = 0; i < 3; ++i)
          pcout << "G" << i + 1 << " : " << d_reciprocalLatticeVectors[i][0]
                << " " << d_reciprocalLatticeVectors[i][1] << " "
                << d_reciprocalLatticeVectors[i][2] << std::endl;
      }
    //
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      {
        for (dftfe::uInt d = 0; d < 3; ++d)
          d_kPointCoordinatesFrac[3 * i + d] = kPointData[i][d];
        d_kPointWeights[i] = kPointData[i][3];
      }
    pcout << "Reduced k-Point-coordinates and weights: " << std::endl;
    //
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      pcout << d_kPointCoordinatesFrac[3 * i + 0] << " "
            << d_kPointCoordinatesFrac[3 * i + 1] << " "
            << d_kPointCoordinatesFrac[3 * i + 2] << " " << d_kPointWeights[i]
            << std::endl;
    //
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      {
        for (dftfe::uInt d1 = 0; d1 < 3; ++d1)
          d_kPointCoordinates[3 * i + d1] =
            d_kPointCoordinatesFrac[3 * i + 0] *
              d_reciprocalLatticeVectors[0][d1] +
            d_kPointCoordinatesFrac[3 * i + 1] *
              d_reciprocalLatticeVectors[1][d1] +
            d_kPointCoordinatesFrac[3 * i + 2] *
              d_reciprocalLatticeVectors[2][d1];
      }
    //
    AssertThrow(
      maxkPoints >= d_dftParamsPtr->npool,
      dealii::ExcMessage(
        "Number of k-points should be higher than or equal to number of pools"));
    const dftfe::uInt this_mpi_pool(
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm));
    std::vector<double> d_kPointCoordinatesGlobal(3 * maxkPoints, 0.0);
    std::vector<double> d_kPointWeightsGlobal(maxkPoints, 0.0);
    std::vector<double> d_kPointCoordinatesFracGlobal(3 * maxkPoints, 0.0);
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      {
        for (dftfe::uInt d = 0; d < 3; ++d)
          {
            d_kPointCoordinatesGlobal[3 * i + d] =
              d_kPointCoordinates[3 * i + d];
            d_kPointCoordinatesFracGlobal[3 * i + d] =
              d_kPointCoordinatesFrac[3 * i + d];
          }
        d_kPointWeightsGlobal[i] = d_kPointWeights[i];
      }
    //
    const dftfe::uInt maxkPointsGlobal = maxkPoints;
    d_kPointCoordinates.clear();
    d_kPointCoordinatesFrac.clear();
    d_kPointWeights.clear();
    maxkPoints             = maxkPointsGlobal / d_dftParamsPtr->npool;
    const dftfe::uInt rest = maxkPointsGlobal % d_dftParamsPtr->npool;
    if (this_mpi_pool < rest)
      maxkPoints = maxkPoints + 1;
    //
    pcout << " check 0.1	" << std::endl;
    //
    d_kPointCoordinates.resize(3 * maxkPoints, 0.0);
    d_kPointCoordinatesFrac.resize(3 * maxkPoints, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    //
    std::vector<int> sendSizekPoints1(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints1(d_dftParamsPtr->npool, 0);
    std::vector<int> sendSizekPoints2(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints2(d_dftParamsPtr->npool, 0);
    if (this_mpi_pool == 0)
      {
        //
        for (dftfe::uInt i = 0; i < d_dftParamsPtr->npool; ++i)
          {
            sendSizekPoints1[i] =
              3 * (maxkPointsGlobal / d_dftParamsPtr->npool);
            sendSizekPoints2[i] = (maxkPointsGlobal / d_dftParamsPtr->npool);
            if (i < rest)
              {
                sendSizekPoints1[i] = sendSizekPoints1[i] + 3;
                sendSizekPoints2[i] = sendSizekPoints2[i] + 1;
              }
            if (i > 0)
              {
                mpiOffsetskPoints1[i] =
                  mpiOffsetskPoints1[i - 1] + sendSizekPoints1[i - 1];
                mpiOffsetskPoints2[i] =
                  mpiOffsetskPoints2[i - 1] + sendSizekPoints2[i - 1];
              }
          }
      }
    //
    pcout << " check 0.2	" << std::endl;
    // pcout << sendSizekPoints[0] << "  " << sendSizekPoints[1] << " " <<
    // maxkPoints << std::endl;
    //
    MPI_Scatterv(&(d_kPointCoordinatesGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(d_kPointCoordinates[0]),
                 3 * maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    MPI_Scatterv(&(d_kPointWeightsGlobal[0]),
                 &(sendSizekPoints2[0]),
                 &(mpiOffsetskPoints2[0]),
                 MPI_DOUBLE,
                 &(d_kPointWeights[0]),
                 maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    MPI_Scatterv(&(d_kPointCoordinatesFracGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(d_kPointCoordinatesFrac[0]),
                 3 * maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
  }
  //============================================================================================================================================
  //============================================================================================================================================
  //			           Following routine recomputes the cartesian k-points between
  // successive relaxation steps
  //============================================================================================================================================
  //============================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::recomputeKPointCoordinates()
  {
    if (d_dftParamsPtr->verbosity >= 4)
      {
        // FIXME: Print all k points across all pools
        pcout
          << "-------------------k points reduced coordinates and weights-----------------------------"
          << std::endl;
        for (dftfe::uInt i = 0; i < d_kPointWeights.size(); ++i)
          {
            pcout << " [" << d_kPointCoordinatesFrac[3 * i + 0] << ", "
                  << d_kPointCoordinatesFrac[3 * i + 1] << ", "
                  << d_kPointCoordinatesFrac[3 * i + 2] << "] "
                  << d_kPointWeights[i] << std::endl;
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

    const std::array<dftfe::uInt, 3> periodic = {d_dftParamsPtr->periodicX,
                                                 d_dftParamsPtr->periodicY,
                                                 d_dftParamsPtr->periodicZ};
    d_reciprocalLatticeVectors =
      internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors,
                                               periodic);
    for (dftfe::uInt i = 0; i < d_kPointWeights.size(); ++i)
      for (dftfe::uInt d = 0; d < 3; ++d)
        d_kPointCoordinates[3 * i + d] =
          d_kPointCoordinatesFrac[3 * i + 0] *
            d_reciprocalLatticeVectors[0][d] +
          d_kPointCoordinatesFrac[3 * i + 1] *
            d_reciprocalLatticeVectors[1][d] +
          d_kPointCoordinatesFrac[3 * i + 2] * d_reciprocalLatticeVectors[2][d];
  }
  //============================================================================================================================================
  //============================================================================================================================================
  //			           Main driver routine to generate the MP grid, reduce BZ
  // using
  // point group symmetries 				                        and scatter the
  // k-points across pools
  //============================================================================================================================================
  //============================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::generateMPGrid()
  {
    dftfe::uInt nkx = d_dftParamsPtr->nkx;
    dftfe::uInt nky = d_dftParamsPtr->nky;
    dftfe::uInt nkz = d_dftParamsPtr->nkz;
    //
    dftfe::uInt offsetFlagX = d_dftParamsPtr->offsetFlagX;
    dftfe::uInt offsetFlagY = d_dftParamsPtr->offsetFlagY;
    dftfe::uInt offsetFlagZ = d_dftParamsPtr->offsetFlagZ;
    //
    double dkx = 0.0;
    double dky = 0.0;
    double dkz = 0.0;
    //
    std::vector<double> del(3);
    dftfe::uInt         maxkPoints = (nkx * nky) * nkz;
    pcout << "Total number of k-points " << maxkPoints << std::endl;
    //=============================================================================================================================================
    //			                                        Generate MP grid
    //=============================================================================================================================================
    del[0] = 1.0 / double(nkx);
    del[1] = 1.0 / double(nky);
    del[2] = 1.0 / double(nkz);
    //
    if (d_dftParamsPtr->offsetFlagX == 1)
      dkx = 0.5 * del[0];
    if (d_dftParamsPtr->offsetFlagY == 1)
      dky = 0.5 * del[1];
    if (d_dftParamsPtr->offsetFlagZ == 1)
      dkz = 0.5 * del[2];
    //
    if (nkx == 1)
      del[0] = 0.0;
    if (nky == 1)
      del[1] = 0.0;
    if (nkz == 1)
      del[2] = 0.0;
    //
    d_kPointCoordinates.resize(maxkPoints * 3, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    //
    d_kPointCoordinatesFrac = d_kPointCoordinates;
    //
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      {
        d_kPointCoordinatesFrac[3 * i + 2] = del[2] * (i % nkz) + dkz;
        d_kPointCoordinatesFrac[3 * i + 1] =
          del[1] * (std::floor((i % (nkz * nky)) / nkz)) + dky;
        d_kPointCoordinatesFrac[3 * i + 0] =
          del[0] * (std::floor((i / (nkz * nky)))) + dkx;
        for (dftfe::uInt dir = 0; dir < 3; ++dir)
          {
            if (d_kPointCoordinatesFrac[3 * i + dir] >= 0.5)
              d_kPointCoordinatesFrac[3 * i + dir] =
                d_kPointCoordinatesFrac[3 * i + dir] - 1.0;
          }
        d_kPointWeights[i] = 1.0 / maxkPoints;
      }
    //
    const std::array<dftfe::uInt, 3> periodic = {d_dftParamsPtr->periodicX,
                                                 d_dftParamsPtr->periodicY,
                                                 d_dftParamsPtr->periodicZ};
    d_reciprocalLatticeVectors =
      internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors,
                                               periodic);
    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout
          << "-----------Reciprocal vectors along which the MP grid is to be generated-------------"
          << std::endl;
        for (dftfe::Int i = 0; i < 3; ++i)
          pcout << "G" << i + 1 << " : " << d_reciprocalLatticeVectors[i][0]
                << " " << d_reciprocalLatticeVectors[i][1] << " "
                << d_reciprocalLatticeVectors[i][2] << std::endl;
      }
    //=============================================================================================================================================
    //			                                         Create irreducible BZ
    //=============================================================================================================================================
    groupSymmetryPtr->reduceKPointGrid(d_kPointCoordinatesFrac,
                                       d_kPointWeights);
    maxkPoints = d_kPointWeights.size();
    if (!d_dftParamsPtr->reproducible_output &&
        (d_dftParamsPtr->useSymm || d_dftParamsPtr->timeReversal))
      {
        pcout << " number of irreducible k-points " << maxkPoints << std::endl;
        pcout << "Reduced k-Point-coordinates and weights: " << std::endl;
        char buffer[100];
        for (dftfe::Int i = 0; i < maxkPoints; ++i)
          {
            sprintf(buffer,
                    "  %5u:  %12.5f  %12.5f %12.5f %12.5f\n",
                    i + 1,
                    d_kPointCoordinatesFrac[3 * i + 0],
                    d_kPointCoordinatesFrac[3 * i + 1],
                    d_kPointCoordinatesFrac[3 * i + 2],
                    d_kPointWeights[i]);
            pcout << buffer;
          }
      }
    for (dftfe::Int i = 0; i < maxkPoints; ++i)
      for (dftfe::uInt d = 0; d < 3; ++d)
        d_kPointCoordinates[3 * i + d] =
          d_kPointCoordinatesFrac[3 * i + 0] *
            d_reciprocalLatticeVectors[0][d] +
          d_kPointCoordinatesFrac[3 * i + 1] *
            d_reciprocalLatticeVectors[1][d] +
          d_kPointCoordinatesFrac[3 * i + 2] * d_reciprocalLatticeVectors[2][d];
    //=============================================================================================================================================
    //			Scatter the irreducible k-points across pools
    //=============================================================================================================================================
    AssertThrow(
      maxkPoints >= d_dftParamsPtr->npool,
      dealii::ExcMessage(
        "Number of k-points should be higher than or equal to number of pools"));
    const dftfe::uInt this_mpi_pool(
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm));
    std::vector<double> d_kPointCoordinatesGlobal(3 * maxkPoints, 0.0);
    std::vector<double> d_kPointWeightsGlobal(maxkPoints, 0.0);
    std::vector<double> d_kPointCoordinatesFracGlobal(3 * maxkPoints, 0.0);
    for (dftfe::uInt i = 0; i < maxkPoints; ++i)
      {
        for (dftfe::uInt d = 0; d < 3; ++d)
          {
            d_kPointCoordinatesGlobal[3 * i + d] =
              d_kPointCoordinates[3 * i + d];
            d_kPointCoordinatesFracGlobal[3 * i + d] =
              d_kPointCoordinatesFrac[3 * i + d];
          }
        d_kPointWeightsGlobal[i] = d_kPointWeights[i];
      }
    //
    const dftfe::uInt maxkPointsGlobal = maxkPoints;
    d_kPointCoordinates.clear();
    d_kPointCoordinatesFrac.clear();
    d_kPointWeights.clear();
    maxkPoints             = maxkPointsGlobal / d_dftParamsPtr->npool;
    const dftfe::uInt rest = maxkPointsGlobal % d_dftParamsPtr->npool;
    if (this_mpi_pool < rest)
      maxkPoints = maxkPoints + 1;
    //
    d_kPointCoordinates.resize(3 * maxkPoints, 0.0);
    d_kPointCoordinatesFrac.resize(3 * maxkPoints, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    //
    std::vector<int> sendSizekPoints1(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints1(d_dftParamsPtr->npool, 0);
    std::vector<int> sendSizekPoints2(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints2(d_dftParamsPtr->npool, 0);
    if (this_mpi_pool == 0)
      {
        //
        for (dftfe::uInt i = 0; i < d_dftParamsPtr->npool; ++i)
          {
            sendSizekPoints1[i] =
              3 * (maxkPointsGlobal / d_dftParamsPtr->npool);
            sendSizekPoints2[i] = maxkPointsGlobal / d_dftParamsPtr->npool;
            if (i < rest)
              {
                sendSizekPoints1[i] = sendSizekPoints1[i] + 3;
                sendSizekPoints2[i] = sendSizekPoints2[i] + 1;
              }
            if (i > 0)
              {
                mpiOffsetskPoints1[i] =
                  mpiOffsetskPoints1[i - 1] + sendSizekPoints1[i - 1];
                mpiOffsetskPoints2[i] =
                  mpiOffsetskPoints2[i - 1] + sendSizekPoints2[i - 1];
              }
          }
      }
    //
    std::vector<int> arrayOfOne(d_dftParamsPtr->npool, 1),
      arrayOffsetOne(d_dftParamsPtr->npool, 1);
    for (dftfe::uInt ipool = 0; ipool < d_dftParamsPtr->npool; ++ipool)
      arrayOffsetOne[ipool] = ipool;
    //
    MPI_Scatterv(&(mpiOffsetskPoints2[0]),
                 &(arrayOfOne[0]),
                 (&arrayOffsetOne[0]),
                 dftfe::dataTypes::mpi_type_id(mpiOffsetskPoints2.data()),
                 &lowerBoundKindex,
                 1,
                 dftfe::dataTypes::mpi_type_id(&lowerBoundKindex),
                 0,
                 interpoolcomm);
    MPI_Scatterv(&(d_kPointCoordinatesGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(d_kPointCoordinates[0]),
                 3 * maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    MPI_Scatterv(&(d_kPointWeightsGlobal[0]),
                 &(sendSizekPoints2[0]),
                 &(mpiOffsetskPoints2[0]),
                 MPI_DOUBLE,
                 &(d_kPointWeights[0]),
                 maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    MPI_Scatterv(&(d_kPointCoordinatesFracGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(d_kPointCoordinatesFrac[0]),
                 3 * maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    //
  }
  // #include "dft.inst.cc"

} // namespace dftfe
