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
#include <symmetry.h>
#include <spglib.h>

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
      const std::array<unsigned int, 3>       periodicity)
    {
      std::vector<std::vector<double>> reciprocalLatticeVectors(
        3, std::vector<double>(3, 0.0));
      unsigned int                     periodicitySum = 0;
      std::vector<double>              cross(3, 0.0);
      std::vector<std::vector<double>> latticeVectorsToBeUsed;
      std::vector<unsigned int>        latticeVectorsToBeUsedIndex;
      double                           scalarConst;
      std::vector<double>              unitVectorOutOfPlane(3, 0.0);
      //
      for (unsigned int i = 0; i < 3; ++i)
        periodicitySum += periodicity[i];
      //
      switch (periodicitySum)
        {
          //=========================================================================================================================================
          case 3: //				      all directions periodic
            //==========================================================================================================================================
            for (unsigned int i = 0; i < 2; ++i)
              {
                cross =
                  internaldft::cross_product(latticeVectors[i + 1],
                                             latticeVectors[3 - (2 * i + 1)]);
                scalarConst = latticeVectors[i][0] * cross[0] +
                              latticeVectors[i][1] * cross[1] +
                              latticeVectors[i][2] * cross[2];
                for (unsigned int d = 0; d < 3; ++d)
                  reciprocalLatticeVectors[i][d] =
                    (2. * M_PI / scalarConst) * cross[d];
              }
            //
            cross =
              internaldft::cross_product(latticeVectors[0], latticeVectors[1]);
            scalarConst = latticeVectors[2][0] * cross[0] +
                          latticeVectors[2][1] * cross[1] +
                          latticeVectors[2][2] * cross[2];
            for (unsigned int d = 0; d < 3; ++d)
              reciprocalLatticeVectors[2][d] =
                (2 * M_PI / scalarConst) * cross[d];
            break;
            //==========================================================================================================================================
          case 2: //				two directions periodic, one direction non-periodic
            //==========================================================================================================================================
            for (unsigned int i = 0; i < 3; ++i)
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
            for (unsigned int d = 0; d < 3; ++d)
              unitVectorOutOfPlane[d] =
                cross[d] / (sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                                 cross[2] * cross[2]));
            //
            for (unsigned int i = 0; i < 2; ++i)
              {
                cross =
                  internaldft::cross_product(latticeVectorsToBeUsed[1 - i],
                                             unitVectorOutOfPlane);
                scalarConst = latticeVectorsToBeUsed[i][0] * cross[0] +
                              latticeVectorsToBeUsed[i][1] * cross[1] +
                              latticeVectorsToBeUsed[i][2] * cross[2];
                for (unsigned int d = 0; d < 3; ++d)
                  reciprocalLatticeVectors[latticeVectorsToBeUsedIndex[i]][d] =
                    (2. * M_PI / scalarConst) * cross[d];
              }
            break;
            //============================================================================================================================================
          case 1: //				two directions non-periodic, one direction periodic
            //============================================================================================================================================
            for (unsigned int i = 0; i < 3; ++i)
              {
                if (periodicity[i] == 1)
                  {
                    const double scalarConst =
                      sqrt(latticeVectors[i][0] * latticeVectors[i][0] +
                           latticeVectors[i][1] * latticeVectors[i][1] +
                           latticeVectors[i][2] * latticeVectors[i][2]);
                    for (unsigned int d = 0; d < 3; ++d)
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
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::readkPointData()
  {
    const int                        numberColumnskPointDataFile = 4;
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
    unsigned int maxkPoints = kPointData.size();
    d_kPointCoordinates.resize(maxkPoints * 3, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    kPointReducedCoordinates = d_kPointCoordinates;
    //
    const std::array<unsigned int, 3> periodic = {d_dftParamsPtr->periodicX,
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
        for (int i = 0; i < 3; ++i)
          pcout << "G" << i + 1 << " : " << d_reciprocalLatticeVectors[i][0]
                << " " << d_reciprocalLatticeVectors[i][1] << " "
                << d_reciprocalLatticeVectors[i][2] << std::endl;
      }
    //
    for (unsigned int i = 0; i < maxkPoints; ++i)
      {
        for (unsigned int d = 0; d < 3; ++d)
          kPointReducedCoordinates[3 * i + d] = kPointData[i][d];
        d_kPointWeights[i] = kPointData[i][3];
      }
    pcout << "Reduced k-Point-coordinates and weights: " << std::endl;
    //
    for (unsigned int i = 0; i < maxkPoints; ++i)
      pcout << kPointReducedCoordinates[3 * i + 0] << " "
            << kPointReducedCoordinates[3 * i + 1] << " "
            << kPointReducedCoordinates[3 * i + 2] << " " << d_kPointWeights[i]
            << std::endl;
    //
    for (unsigned int i = 0; i < maxkPoints; ++i)
      {
        for (unsigned int d1 = 0; d1 < 3; ++d1)
          d_kPointCoordinates[3 * i + d1] =
            kPointReducedCoordinates[3 * i + 0] *
              d_reciprocalLatticeVectors[0][d1] +
            kPointReducedCoordinates[3 * i + 1] *
              d_reciprocalLatticeVectors[1][d1] +
            kPointReducedCoordinates[3 * i + 2] *
              d_reciprocalLatticeVectors[2][d1];
      }
    //
    AssertThrow(
      maxkPoints >= d_dftParamsPtr->npool,
      dealii::ExcMessage(
        "Number of k-points should be higher than or equal to number of pools"));
    const unsigned int this_mpi_pool(
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm));
    std::vector<double> d_kPointCoordinatesGlobal(3 * maxkPoints, 0.0);
    std::vector<double> d_kPointWeightsGlobal(maxkPoints, 0.0);
    std::vector<double> kPointReducedCoordinatesGlobal(3 * maxkPoints, 0.0);
    for (unsigned int i = 0; i < maxkPoints; ++i)
      {
        for (unsigned int d = 0; d < 3; ++d)
          {
            d_kPointCoordinatesGlobal[3 * i + d] =
              d_kPointCoordinates[3 * i + d];
            kPointReducedCoordinatesGlobal[3 * i + d] =
              kPointReducedCoordinates[3 * i + d];
          }
        d_kPointWeightsGlobal[i] = d_kPointWeights[i];
      }
    //
    const unsigned int maxkPointsGlobal = maxkPoints;
    d_kPointCoordinates.clear();
    kPointReducedCoordinates.clear();
    d_kPointWeights.clear();
    maxkPoints              = maxkPointsGlobal / d_dftParamsPtr->npool;
    const unsigned int rest = maxkPointsGlobal % d_dftParamsPtr->npool;
    if (this_mpi_pool < rest)
      maxkPoints = maxkPoints + 1;
    //
    pcout << " check 0.1	" << std::endl;
    //
    d_kPointCoordinates.resize(3 * maxkPoints, 0.0);
    kPointReducedCoordinates.resize(3 * maxkPoints, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    //
    std::vector<int> sendSizekPoints1(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints1(d_dftParamsPtr->npool, 0);
    std::vector<int> sendSizekPoints2(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints2(d_dftParamsPtr->npool, 0);
    if (this_mpi_pool == 0)
      {
        //
        for (unsigned int i = 0; i < d_dftParamsPtr->npool; ++i)
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
    MPI_Scatterv(&(kPointReducedCoordinatesGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(kPointReducedCoordinates[0]),
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
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::recomputeKPointCoordinates()
  {
    if (d_dftParamsPtr->verbosity >= 4)
      {
        // FIXME: Print all k points across all pools
        pcout
          << "-------------------k points reduced coordinates and weights-----------------------------"
          << std::endl;
        for (unsigned int i = 0; i < d_kPointWeights.size(); ++i)
          {
            pcout << " [" << kPointReducedCoordinates[3 * i + 0] << ", "
                  << kPointReducedCoordinates[3 * i + 1] << ", "
                  << kPointReducedCoordinates[3 * i + 2] << "] "
                  << d_kPointWeights[i] << std::endl;
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

    const std::array<unsigned int, 3> periodic = {d_dftParamsPtr->periodicX,
                                                  d_dftParamsPtr->periodicY,
                                                  d_dftParamsPtr->periodicZ};
    d_reciprocalLatticeVectors =
      internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors,
                                               periodic);
    for (unsigned int i = 0; i < d_kPointWeights.size(); ++i)
      for (unsigned int d = 0; d < 3; ++d)
        d_kPointCoordinates[3 * i + d] = kPointReducedCoordinates[3 * i + 0] *
                                           d_reciprocalLatticeVectors[0][d] +
                                         kPointReducedCoordinates[3 * i + 1] *
                                           d_reciprocalLatticeVectors[1][d] +
                                         kPointReducedCoordinates[3 * i + 2] *
                                           d_reciprocalLatticeVectors[2][d];
  }
  //============================================================================================================================================
  //============================================================================================================================================
  //			           Main driver routine to generate the MP grid, reduce BZ
  // using
  // point group symmetries 				                        and scatter the
  // k-points across pools
  //============================================================================================================================================
  //============================================================================================================================================
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::generateMPGrid()
  {
    unsigned int nkx = d_dftParamsPtr->nkx;
    unsigned int nky = d_dftParamsPtr->nky;
    unsigned int nkz = d_dftParamsPtr->nkz;
    //
    unsigned int offsetFlagX = d_dftParamsPtr->offsetFlagX;
    unsigned int offsetFlagY = d_dftParamsPtr->offsetFlagY;
    unsigned int offsetFlagZ = d_dftParamsPtr->offsetFlagZ;
    //
    double dkx = 0.0;
    double dky = 0.0;
    double dkz = 0.0;
    //
    std::vector<double> del(3);
    unsigned int        maxkPoints = (nkx * nky) * nkz;
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
    kPointReducedCoordinates = d_kPointCoordinates;
    //
    for (unsigned int i = 0; i < maxkPoints; ++i)
      {
        kPointReducedCoordinates[3 * i + 2] = del[2] * (i % nkz) + dkz;
        kPointReducedCoordinates[3 * i + 1] =
          del[1] * (std::floor((i % (nkz * nky)) / nkz)) + dky;
        kPointReducedCoordinates[3 * i + 0] =
          del[0] * (std::floor((i / (nkz * nky)))) + dkx;
        for (unsigned int dir = 0; dir < 3; ++dir)
          {
            if (kPointReducedCoordinates[3 * i + dir] > (0.5 + 1.0E-10))
              kPointReducedCoordinates[3 * i + dir] =
                kPointReducedCoordinates[3 * i + dir] - 1.0;
          }
        d_kPointWeights[i] = 1.0 / maxkPoints;
      }
    //
    const std::array<unsigned int, 3> periodic = {d_dftParamsPtr->periodicX,
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
        for (int i = 0; i < 3; ++i)
          pcout << "G" << i + 1 << " : " << d_reciprocalLatticeVectors[i][0]
                << " " << d_reciprocalLatticeVectors[i][1] << " "
                << d_reciprocalLatticeVectors[i][2] << std::endl;
      }
    //=============================================================================================================================================
    //			                                         Create irreducible BZ
    //=============================================================================================================================================
    if (d_dftParamsPtr->useSymm || d_dftParamsPtr->timeReversal)
      {
        //
        const int                        numberColumnsSymmDataFile = 3;
        std::vector<std::vector<int>>    symmUnderGroupTemp;
        std::vector<std::vector<double>> symmData;
        std::vector<std::vector<std::vector<double>>> symmMatTemp, symmMatTemp2;
        const int                                     max_size = 500;
        int                                           rotation[max_size][3][3];
        //
        if (d_dftParamsPtr->useSymm)
          {
            const int num_atom = atomLocationsFractional.size();
            double    lattice[3][3], position[num_atom][3];
            int       types[num_atom];
            const int mesh[3] = {static_cast<int>(nkx),
                                 static_cast<int>(nky),
                                 static_cast<int>(nkz)};
            int       grid_address[nkx * nky * nkz][3];
            int       grid_mapping_table[nkx * nky * nkz];
            //
            for (unsigned int i = 0; i < 3; ++i)
              {
                for (unsigned int j = 0; j < 3; ++j)
                  lattice[i][j] = d_domainBoundingVectors[i][j];
              }
            std::set<unsigned int>::iterator it = atomTypes.begin();
            for (unsigned int i = 0; i < num_atom; ++i)
              {
                std::advance(it, i);
                types[i] = atomLocationsFractional[i][0];
                for (unsigned int j = 0; j < 3; ++j)
                  position[i][j] = atomLocationsFractional[i][j + 2];
              }
            //
            if (!d_dftParamsPtr->reproducible_output)
              pcout << " getting space group symmetries from spg " << std::endl;
            symmetryPtr->numSymm = spg_get_symmetry(rotation,
                                                    (symmetryPtr->translation),
                                                    max_size,
                                                    lattice,
                                                    position,
                                                    types,
                                                    num_atom,
                                                    1e-5);
            if (!d_dftParamsPtr->reproducible_output &&
                d_dftParamsPtr->verbosity > 3)
              {
                pcout << " number of symmetries allowed for the lattice "
                      << symmetryPtr->numSymm << std::endl;
                for (unsigned int iSymm = 0; iSymm < symmetryPtr->numSymm;
                     ++iSymm)
                  {
                    pcout << " Symmetry " << iSymm + 1 << std::endl;
                    pcout << " Rotation " << std::endl;
                    for (unsigned int ipol = 0; ipol < 3; ++ipol)
                      pcout << rotation[iSymm][ipol][0] << "  "
                            << rotation[iSymm][ipol][1] << "  "
                            << rotation[iSymm][ipol][2] << std::endl;
                    pcout << " translation " << std::endl;
                    pcout << symmetryPtr->translation[iSymm][0] << "  "
                          << symmetryPtr->translation[iSymm][1] << "  "
                          << symmetryPtr->translation[iSymm][2] << std::endl;
                    pcout << "	" << std::endl;
                  }
              }
          }
        //
        else
          {
            symmetryPtr->numSymm = 1;
            for (unsigned int j = 0; j < 3; ++j)
              {
                for (unsigned int k = 0; k < 3; ++k)
                  {
                    if (j == k)
                      rotation[0][j][k] = 1;
                    else
                      rotation[0][j][k] = 0;
                  }
                symmetryPtr->translation[0][j] = 0.0;
              }
            if (!d_dftParamsPtr->reproducible_output &&
                d_dftParamsPtr->verbosity > 3)
              pcout << " Only time reversal symmetry to be used " << std::endl;
          }
        //
        if (d_dftParamsPtr->timeReversal)
          {
            for (unsigned int iSymm = symmetryPtr->numSymm;
                 iSymm < 2 * symmetryPtr->numSymm;
                 ++iSymm)
              {
                for (unsigned int j = 0; j < 3; ++j)
                  {
                    for (unsigned int k = 0; k < 3; ++k)
                      rotation[iSymm][j][k] =
                        -1 * rotation[iSymm - symmetryPtr->numSymm][j][k];
                    symmetryPtr->translation[iSymm][j] =
                      symmetryPtr->translation[iSymm - symmetryPtr->numSymm][j];
                  }
              }
            symmetryPtr->numSymm = 2 * symmetryPtr->numSymm;
          }
        //
        symmMatTemp.resize(symmetryPtr->numSymm);
        symmMatTemp2.resize(symmetryPtr->numSymm);
        symmetryPtr->symmMat.resize(symmetryPtr->numSymm);
        symmUnderGroupTemp.resize(symmetryPtr->numSymm);
        for (unsigned int i = 0; i < symmetryPtr->numSymm; ++i)
          {
            symmMatTemp[i].resize(3, std::vector<double>(3, 0.0));
            symmMatTemp2[i].resize(3, std::vector<double>(3, 0.0));
            for (unsigned int j = 0; j < 3; ++j)
              {
                for (unsigned int k = 0; k < 3; ++k)
                  {
                    symmMatTemp[i][j][k]  = double(rotation[i][j][k]);
                    symmMatTemp2[i][j][k] = double(rotation[i][j][k]);
                    if (d_dftParamsPtr->timeReversal &&
                        i >= symmetryPtr->numSymm / 2)
                      symmMatTemp2[i][j][k] = -double(rotation[i][j][k]);
                  }
              }
          }
        //
        std::vector<double> kPointAllCoordinates, kPointTemp(3);
        std::vector<int>    discard(maxkPoints, 0),
          countedSymm(symmetryPtr->numSymm, 0),
          usedSymmNum(symmetryPtr->numSymm, 1);
        kPointAllCoordinates = kPointReducedCoordinates;
        const int nk         = maxkPoints;
        maxkPoints           = 0;
        //
        double translationTemp[symmetryPtr->numSymm][3];
        for (unsigned int i = 0; i < (symmetryPtr->numSymm); ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              translationTemp[i][j] = (symmetryPtr->translation)[i][j];
          }
        //
        symmetryPtr->symmMat[0] = symmMatTemp[0];
        unsigned int usedSymm   = 1,
                     ik = 0; // note usedSymm is initialized to 1 and not 0.
                             // Because identity is always present
        countedSymm[0] = 1;
        //
        while (ik < nk)
          {
            for (unsigned int d = 0; d < 3; ++d)
              kPointReducedCoordinates[3 * maxkPoints + d] =
                kPointAllCoordinates[3 * ik + d];
            maxkPoints = maxkPoints + 1;
            //
            for (unsigned int iSymm = 1; iSymm < symmetryPtr->numSymm;
                 ++iSymm) // iSymm begins from 1. because identity is always
                          // present and is taken care of.
              {
                for (unsigned int d = 0; d < 3; ++d)
                  kPointTemp[d] =
                    kPointAllCoordinates[3 * ik + 0] *
                      symmMatTemp[iSymm][0][d] +
                    kPointAllCoordinates[3 * ik + 1] *
                      symmMatTemp[iSymm][1][d] +
                    kPointAllCoordinates[3 * ik + 2] * symmMatTemp[iSymm][2][d];
                //
                for (unsigned int dir = 0; dir < 3; ++dir)
                  {
                    while (kPointTemp[dir] > (1.0 - 1.0E-5))
                      kPointTemp[dir] = kPointTemp[dir] - 1.0;
                    while (kPointTemp[dir] < -1.0E-5)
                      kPointTemp[dir] = kPointTemp[dir] + 1.0;
                  }
                //
                unsigned int ikx =
                  (round(kPointTemp[0] * (1 + offsetFlagX) * nkx) -
                   offsetFlagX);
                unsigned int iky =
                  (round(kPointTemp[1] * (1 + offsetFlagY) * nky) -
                   offsetFlagY);
                unsigned int ikz =
                  (round(kPointTemp[2] * (1 + offsetFlagZ) * nkz) -
                   offsetFlagZ);
                //
                if (ikx % (1 + offsetFlagX) == 0)
                  ikx = (round(kPointTemp[0] * (1 + offsetFlagX) * nkx) -
                         offsetFlagX) /
                        (1 + offsetFlagX);
                else
                  ikx = 1001;
                if (iky % (1 + offsetFlagY) == 0)
                  iky = (round(kPointTemp[1] * (1 + offsetFlagY) * nky) -
                         offsetFlagY) /
                        (1 + offsetFlagY);
                else
                  iky = 1001;
                if (ikz % (1 + offsetFlagZ) == 0)
                  ikz = (round(kPointTemp[2] * (1 + offsetFlagZ) * nkz) -
                         offsetFlagZ) /
                        (1 + offsetFlagZ);
                else
                  ikz = 1001;
                //
                const unsigned int jk = ikx * nky * nkz + iky * nkz + ikz;
                if (jk != ik && jk < nk && discard[jk] != 1)
                  {
                    d_kPointWeights[maxkPoints - 1] =
                      d_kPointWeights[maxkPoints - 1] + 1.0 / nk;
                    discard[jk] = 1;
                    if (countedSymm[iSymm] == 0)
                      {
                        usedSymmNum[iSymm]               = usedSymm;
                        (symmetryPtr->symmMat)[usedSymm] = symmMatTemp2[iSymm];
                        for (unsigned int j = 0; j < 3; ++j)
                          (symmetryPtr->translation)[usedSymm][j] =
                            translationTemp[iSymm][j];
                        usedSymm++;
                        countedSymm[iSymm] = 1;
                      }
                    symmUnderGroupTemp[usedSymmNum[iSymm]].push_back(
                      maxkPoints - 1);
                  }
              }
            //
            discard[ik] = 1;
            ik          = ik + 1;
            if (ik < nk)
              {
                while (discard[ik] == 1)
                  {
                    ik = ik + 1;
                    if (ik == nk)
                      break;
                  }
              }
          }
        //
        symmetryPtr->numSymm = usedSymm;
        symmetryPtr->symmUnderGroup.resize(
          maxkPoints, std::vector<int>(symmetryPtr->numSymm, 0));
        symmetryPtr->numSymmUnderGroup.resize(
          maxkPoints,
          1); // minimum should be 1, because identity is always present
        for (unsigned int i = 0; i < maxkPoints; ++i)
          {
            symmetryPtr->symmUnderGroup[i][0] = 1;
            for (unsigned int iSymm = 1; iSymm < (symmetryPtr->numSymm);
                 ++iSymm)
              {
                if (std::find(symmUnderGroupTemp[iSymm].begin(),
                              symmUnderGroupTemp[iSymm].end(),
                              i) != symmUnderGroupTemp[iSymm].end())
                  {
                    symmetryPtr->symmUnderGroup[i][iSymm] = 1;
                    symmetryPtr->numSymmUnderGroup[i] += 1;
                  }
              }
            if (d_dftParamsPtr->verbosity > 3)
              pcout << " kpoint " << i << " numSymmUnderGroup "
                    << symmetryPtr->numSymmUnderGroup[i] << std::endl;
          }
        //
        if (!d_dftParamsPtr->reproducible_output)
          {
            if (d_dftParamsPtr->verbosity > 3)
              {
                pcout << " " << usedSymm << " symmetries used to reduce BZ "
                      << std::endl;
                for (unsigned int iSymm = 0; iSymm < symmetryPtr->numSymm;
                     ++iSymm)
                  {
                    for (unsigned int ipol = 0; ipol < 3; ++ipol)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          pcout << symmetryPtr->symmMat[iSymm][ipol][0] << "  "
                                << symmetryPtr->symmMat[iSymm][ipol][1] << "  "
                                << symmetryPtr->symmMat[iSymm][ipol][2]
                                << std::endl;
                      }
                    pcout << " " << usedSymm << "  " << std::endl;
                  }
              }
            pcout << " number of irreducible k-points " << maxkPoints
                  << std::endl;
            pcout << "Reduced k-Point-coordinates and weights: " << std::endl;
            char buffer[100];
            for (int i = 0; i < maxkPoints; ++i)
              {
                sprintf(buffer,
                        "  %5u:  %12.5f  %12.5f %12.5f %12.5f\n",
                        i + 1,
                        kPointReducedCoordinates[3 * i + 0],
                        kPointReducedCoordinates[3 * i + 1],
                        kPointReducedCoordinates[3 * i + 2],
                        d_kPointWeights[i]);
                pcout << buffer;
              }
          }
      }
    for (int i = 0; i < maxkPoints; ++i)
      for (unsigned int d = 0; d < 3; ++d)
        d_kPointCoordinates[3 * i + d] = kPointReducedCoordinates[3 * i + 0] *
                                           d_reciprocalLatticeVectors[0][d] +
                                         kPointReducedCoordinates[3 * i + 1] *
                                           d_reciprocalLatticeVectors[1][d] +
                                         kPointReducedCoordinates[3 * i + 2] *
                                           d_reciprocalLatticeVectors[2][d];
    //=============================================================================================================================================
    //			Scatter the irreducible k-points across pools
    //=============================================================================================================================================
    AssertThrow(
      maxkPoints >= d_dftParamsPtr->npool,
      dealii::ExcMessage(
        "Number of k-points should be higher than or equal to number of pools"));
    const unsigned int this_mpi_pool(
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm));
    std::vector<double> d_kPointCoordinatesGlobal(3 * maxkPoints, 0.0);
    std::vector<double> d_kPointWeightsGlobal(maxkPoints, 0.0);
    std::vector<double> kPointReducedCoordinatesGlobal(3 * maxkPoints, 0.0);
    for (unsigned int i = 0; i < maxkPoints; ++i)
      {
        for (unsigned int d = 0; d < 3; ++d)
          {
            d_kPointCoordinatesGlobal[3 * i + d] =
              d_kPointCoordinates[3 * i + d];
            kPointReducedCoordinatesGlobal[3 * i + d] =
              kPointReducedCoordinates[3 * i + d];
          }
        d_kPointWeightsGlobal[i] = d_kPointWeights[i];
      }
    //
    const unsigned int maxkPointsGlobal = maxkPoints;
    d_kPointCoordinates.clear();
    kPointReducedCoordinates.clear();
    d_kPointWeights.clear();
    maxkPoints              = maxkPointsGlobal / d_dftParamsPtr->npool;
    const unsigned int rest = maxkPointsGlobal % d_dftParamsPtr->npool;
    if (this_mpi_pool < rest)
      maxkPoints = maxkPoints + 1;
    //
    d_kPointCoordinates.resize(3 * maxkPoints, 0.0);
    kPointReducedCoordinates.resize(3 * maxkPoints, 0.0);
    d_kPointWeights.resize(maxkPoints, 0.0);
    //
    std::vector<int> sendSizekPoints1(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints1(d_dftParamsPtr->npool, 0);
    std::vector<int> sendSizekPoints2(d_dftParamsPtr->npool, 0),
      mpiOffsetskPoints2(d_dftParamsPtr->npool, 0);
    if (this_mpi_pool == 0)
      {
        //
        for (unsigned int i = 0; i < d_dftParamsPtr->npool; ++i)
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
    for (unsigned int ipool = 0; ipool < d_dftParamsPtr->npool; ++ipool)
      arrayOffsetOne[ipool] = ipool;
    //
    MPI_Scatterv(&(mpiOffsetskPoints2[0]),
                 &(arrayOfOne[0]),
                 (&arrayOffsetOne[0]),
                 MPI_INT,
                 &lowerBoundKindex,
                 1,
                 MPI_INT,
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
    MPI_Scatterv(&(kPointReducedCoordinatesGlobal[0]),
                 &(sendSizekPoints1[0]),
                 &(mpiOffsetskPoints1[0]),
                 MPI_DOUBLE,
                 &(kPointReducedCoordinates[0]),
                 3 * maxkPoints,
                 MPI_DOUBLE,
                 0,
                 interpoolcomm);
    //
  }
  //#include "dft.inst.cc"

} // namespace dftfe
