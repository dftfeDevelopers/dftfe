// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020x The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//
#include <dft.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <sys/stat.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

namespace dftfe
{
  void
  loadSingleAtomPSIFiles(
    unsigned int  Z,
    unsigned int  n,
    unsigned int  l,
    unsigned int &fileReadFlag,
    double &      wfcInitTruncation,
    std::map<unsigned int,
             std::map<unsigned int,
                      std::map<unsigned int, alglib::spline1dinterpolant>>>
      &                  radValues,
    const MPI_Comm &     mpiCommParent,
    const dftParameters &dftParams)
  {
    if (radValues[Z][n].count(l) > 0)
      {
        fileReadFlag = 1;
        return;
      }

    //
    // set the paths for the Single-Atom wavefunction data
    //
    char psiFile[256];

    if (dftParams.isPseudopotential)
      {
        if (dftParams.readWfcForPdosPspFile && Z == 78)
          {
            sprintf(
              psiFile,
              "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomDataKB/psi%u%u.inp",
              DFTFE_PATH,
              Z,
              n,
              l);
          }
        else
          {
            sprintf(
              psiFile,
              "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp",
              DFTFE_PATH,
              Z,
              n,
              l);
          }
      }
    else
      sprintf(
        psiFile,
        "%s/data/electronicStructure/allElectron/z%u/singleAtomData/psi%u%u.inp",
        DFTFE_PATH,
        Z,
        n,
        l);

    std::vector<std::vector<double>> values;

    const double truncationTol = 1e-8;
    fileReadFlag               = dftUtils::readPsiFile(2, values, psiFile);

    //
    // spline fitting for single-atom wavefunctions
    //
    if (fileReadFlag > 0)
      {
        double       maxTruncationRadius = 0.0;
        unsigned int truncRowId          = 0;
        if (!dftParams.reproducible_output)
          {
            if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
              std::cout << "reading data from file: " << psiFile << std::endl;
          }

        int                 numRows = values.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        // x
        for (int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = values[irow][0];
          }
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);

        // y
        for (int irow = 0; irow < numRows; ++irow)
          {
            yData[irow] = values[irow][1];

            if (std::fabs(yData[irow]) > truncationTol)
              truncRowId = irow;
          }
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type = 0;
        alglib::spline1dbuildcubic(x,
                                   y,
                                   numRows,
                                   natural_bound_type,
                                   0.0,
                                   natural_bound_type,
                                   0.0,
                                   radValues[Z][n][l]);

        maxTruncationRadius = xData[truncRowId];
        if (maxTruncationRadius > wfcInitTruncation)
          wfcInitTruncation = maxTruncationRadius;
      }
  }



  // compute fermi energy
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::compute_tdos(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const std::string &                     dosFileName)
  {
    computing_timer.enter_subsection("DOS computation");
    std::vector<double> eigenValuesAllkPoints;
    for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    double totalEigenValues  = eigenValuesAllkPoints.size();
    double intervalSize      = 0.001;
    double sigma             = C_kb * d_dftParamsPtr->TVal;
    double lowerBoundEpsilon = 1.5 * eigenValuesAllkPoints[0];
    double upperBoundEpsilon =
      eigenValuesAllkPoints[totalEigenValues - 1] * 1.5;
    unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);

    std::vector<double> densityOfStates, densityOfStatesUp, densityOfStatesDown;


    if (d_dftParamsPtr->spinPolarized == 1)
      {
        densityOfStatesUp.resize(numberIntervals, 0.0);
        densityOfStatesDown.resize(numberIntervals, 0.0);
        for (int epsInt = 0; epsInt < numberIntervals; ++epsInt)
          {
            double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
            for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
              {
                for (unsigned int spinType = 0;
                     spinType < 1 + d_dftParamsPtr->spinPolarized;
                     ++spinType)
                  {
                    for (unsigned int statesIter = 0;
                         statesIter < d_numEigenValues;
                         ++statesIter)
                      {
                        double term1 =
                          (epsValue -
                           eigenValuesInput[kPoint]
                                           [spinType * d_numEigenValues +
                                            statesIter]);
                        double denom = term1 * term1 + sigma * sigma;
                        if (spinType == 0)
                          densityOfStatesUp[epsInt] +=
                            (sigma / M_PI) * (1.0 / denom);
                        else
                          densityOfStatesDown[epsInt] +=
                            (sigma / M_PI) * (1.0 / denom);
                      }
                  }
              }
          }
      }
    else
      {
        densityOfStates.resize(numberIntervals, 0.0);
        for (int epsInt = 0; epsInt < numberIntervals; ++epsInt)
          {
            double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
            for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
              {
                for (unsigned int statesIter = 0; statesIter < d_numEigenValues;
                     ++statesIter)
                  {
                    double term1 =
                      (epsValue - eigenValuesInput[kPoint][statesIter]);
                    double denom = term1 * term1 + sigma * sigma;
                    densityOfStates[epsInt] +=
                      2.0 * (sigma / M_PI) * (1.0 / denom);
                  }
              }
          }
      }

    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::ofstream outFile(dosFileName.c_str());
        outFile.setf(std::ios_base::fixed);

        if (outFile.is_open())
          {
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << "  " << densityOfStatesUp[epsInt] << " "
                            << densityOfStatesDown[epsInt] << std::endl;
                  }
              }
            else
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << "  " << densityOfStates[epsInt] << std::endl;
                  }
              }
          }
      }
    computing_timer.leave_subsection("DOS computation");
  }


  // compute local density of states
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::compute_ldos(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const std::string &                     ldosFileName)
  {
    computing_timer.enter_subsection("LDOS computation");
    //
    // create a map of cellId and atomId
    //

    // loop over elements
    std::vector<double> eigenValuesAllkPoints;
    for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    double totalEigenValues  = eigenValuesAllkPoints.size();
    double intervalSize      = 0.001;
    double sigma             = C_kb * d_dftParamsPtr->TVal;
    double lowerBoundEpsilon = 1.5 * eigenValuesAllkPoints[0];
    double upperBoundEpsilon =
      eigenValuesAllkPoints[totalEigenValues - 1] * 1.5;
    unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);
    unsigned int numberGlobalAtoms = atomLocations.size();

    // map each cell to an atom based on closest atom to the centroid of each
    // cell
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    std::map<dealii::CellId, unsigned int> cellToAtomIdMap;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            const dealii::Point<3> center(cell->center());

            // loop over all atoms
            double           distanceToClosestAtom = 1e8;
            dealii::Point<3> closestAtom;
            unsigned int     closestAtomId;
            for (unsigned int n = 0; n < atomLocations.size(); n++)
              {
                dealii::Point<3> atom(atomLocations[n][2],
                                      atomLocations[n][3],
                                      atomLocations[n][4]);
                if (center.distance(atom) < distanceToClosestAtom)
                  {
                    distanceToClosestAtom = center.distance(atom);
                    closestAtom           = atom;
                    closestAtomId         = n;
                  }
              }
            cellToAtomIdMap[cell->id()] = closestAtomId;
          }
      }

    std::vector<double> localDensityOfStates, localDensityOfStatesUp,
      localDensityOfStatesDown;
    localDensityOfStates.resize(numberGlobalAtoms * numberIntervals, 0.0);
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        localDensityOfStatesUp.resize(numberGlobalAtoms * numberIntervals, 0.0);
        localDensityOfStatesDown.resize(numberGlobalAtoms * numberIntervals,
                                        0.0);
      }

    // access finite-element data
    dealii::QGauss<3>   quadrature_formula(C_num1DQuad<FEOrder>());
    dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                  quadrature_formula,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int  dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
    const unsigned int  n_q_points    = quadrature_formula.size();


    const unsigned int blockSize =
      std::min(d_dftParamsPtr->wfcBlockSize, d_numEigenValues);

    std::vector<double> tempContribution(blockSize, 0.0);
    std::vector<double> tempQuadPointValues(n_q_points);

    const unsigned int localVectorSize =
      matrix_free_data.get_vector_partitioner()->locally_owned_size();
    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());
    std::vector<distributedCPUVec<dataTypes::number>>
      eigenVectorsFlattenedBlock((1 + d_dftParamsPtr->spinPolarized) *
                                 d_kPointWeights.size());

    for (unsigned int ivec = 0; ivec < d_numEigenValues; ivec += blockSize)
      {
        const unsigned int currentBlockSize =
          std::min(blockSize, d_numEigenValues - ivec);

        if (currentBlockSize != blockSize || ivec == 0)
          {
            for (unsigned int kPoint = 0;
                 kPoint <
                 (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
                 ++kPoint)
              {
                eigenVectors[kPoint].resize(currentBlockSize);
                for (unsigned int i = 0; i < currentBlockSize; ++i)
                  eigenVectors[kPoint][i].reinit(d_tempEigenVec);


                vectorTools::createDealiiVector<dataTypes::number>(
                  matrix_free_data.get_vector_partitioner(),
                  currentBlockSize,
                  eigenVectorsFlattenedBlock[kPoint]);
                eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
              }

            constraintsNoneDataInfo.precomputeMaps(
              matrix_free_data.get_vector_partitioner(),
              eigenVectorsFlattenedBlock[0].get_partitioner(),
              currentBlockSize);
          }


        std::vector<std::vector<double>> blockedEigenValues(
          d_kPointWeights.size(),
          std::vector<double>((1 + d_dftParamsPtr->spinPolarized) *
                                currentBlockSize,
                              0.0));
        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
            {
              blockedEigenValues[kPoint][iWave] =
                eigenValues[kPoint][ivec + iWave];
              if (d_dftParamsPtr->spinPolarized == 1)
                blockedEigenValues[kPoint][currentBlockSize + iWave] =
                  eigenValues[kPoint][d_numEigenValues + ivec + iWave];
            }

        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          {
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
                eigenVectorsFlattenedBlock[kPoint].local_element(
                  iNode * currentBlockSize + iWave) =
                  d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                                localVectorSize +
                                              iNode * d_numEigenValues + ivec +
                                              iWave];

            constraintsNoneDataInfo.distribute(
              eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
            eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              eigenVectors[kPoint],
              false);

            // FIXME: The underlying call to update_ghost_values
            // is required because currently localProc_dof_indicesReal
            // and localProc_dof_indicesImag are only available for
            // locally owned nodes. Once they are also made available
            // for ghost nodes- use true for the last argument in
            // copyFlattenedDealiiVecToSingleCompVec(..) above and supress
            // underlying call.
            for (unsigned int i = 0; i < currentBlockSize; ++i)
              eigenVectors[kPoint][i].update_ghost_values();
#else
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              eigenVectors[kPoint],
              true);

#endif
          }

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            for (unsigned int spinType = 0; spinType < 2; ++spinType)
              {
                typename dealii::DoFHandler<3>::active_cell_iterator
                  cellN = dofHandler.begin_active(),
                  endcN = dofHandler.end();

                for (; cellN != endcN; ++cellN)
                  {
                    if (cellN->is_locally_owned())
                      {
                        fe_values.reinit(cellN);
                        unsigned int globalAtomId =
                          cellToAtomIdMap[cellN->id()];

                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            fe_values.get_function_values(
                              eigenVectors[spinType][iEigenVec],
                              tempQuadPointValues);

                            tempContribution[iEigenVec] = 0.0;
                            for (unsigned int q_point = 0; q_point < n_q_points;
                                 ++q_point)
                              {
                                tempContribution[iEigenVec] +=
                                  tempQuadPointValues[q_point] *
                                  tempQuadPointValues[q_point] *
                                  fe_values.JxW(q_point);
                              }
                          }

                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          for (unsigned int epsInt = 0;
                               epsInt < numberIntervals;
                               ++epsInt)
                            {
                              double epsValue =
                                lowerBoundEpsilon + epsInt * intervalSize;
                              double term1 =
                                (epsValue -
                                 blockedEigenValues[0][spinType *
                                                         currentBlockSize +
                                                       iEigenVec]);
                              double smearedEnergyLevel =
                                (sigma / M_PI) *
                                (1.0 / (term1 * term1 + sigma * sigma));

                              if (spinType == 0)
                                localDensityOfStatesUp[numberIntervals *
                                                         globalAtomId +
                                                       epsInt] +=
                                  tempContribution[iEigenVec] *
                                  smearedEnergyLevel;
                              else
                                localDensityOfStatesDown[numberIntervals *
                                                           globalAtomId +
                                                         epsInt] +=
                                  tempContribution[iEigenVec] *
                                  smearedEnergyLevel;
                            }
                      }
                  }
              }
          }
        else
          {
            typename dealii::DoFHandler<3>::active_cell_iterator
              cellN = dofHandler.begin_active(),
              endcN = dofHandler.end();

            for (; cellN != endcN; ++cellN)
              {
                if (cellN->is_locally_owned())
                  {
                    fe_values.reinit(cellN);
                    unsigned int globalAtomId = cellToAtomIdMap[cellN->id()];

                    for (unsigned int iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      {
                        fe_values.get_function_values(
                          eigenVectors[0][iEigenVec], tempQuadPointValues);

                        tempContribution[iEigenVec] = 0.0;
                        for (unsigned int q_point = 0; q_point < n_q_points;
                             ++q_point)
                          {
                            tempContribution[iEigenVec] +=
                              tempQuadPointValues[q_point] *
                              tempQuadPointValues[q_point] *
                              fe_values.JxW(q_point);
                          }
                      }

                    for (unsigned int iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      for (unsigned int epsInt = 0; epsInt < numberIntervals;
                           ++epsInt)
                        {
                          double epsValue =
                            lowerBoundEpsilon + epsInt * intervalSize;
                          double term1 =
                            (epsValue - blockedEigenValues[0][iEigenVec]);
                          double smearedEnergyLevel =
                            (sigma / M_PI) *
                            (1.0 / (term1 * term1 + sigma * sigma));
                          localDensityOfStates[numberIntervals * globalAtomId +
                                               epsInt] +=
                            2.0 * tempContribution[iEigenVec] *
                            smearedEnergyLevel;
                        }
                  }
              }
          }
      } // ivec loop

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dealii::Utilities::MPI::sum(localDensityOfStatesUp,
                                    mpi_communicator,
                                    localDensityOfStatesUp);

        dealii::Utilities::MPI::sum(localDensityOfStatesDown,
                                    mpi_communicator,
                                    localDensityOfStatesDown);
      }
    else
      {
        dealii::Utilities::MPI::sum(localDensityOfStates,
                                    mpi_communicator,
                                    localDensityOfStates);
      }

    double checkSum = 0;
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::ofstream outFile(ldosFileName.c_str());
        outFile.setf(std::ios_base::fixed);

        if (outFile.is_open())
          {
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << " ";
                    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms;
                         ++iAtom)
                      {
                        outFile
                          << std::setprecision(18)
                          << localDensityOfStatesUp[numberIntervals * iAtom +
                                                    epsInt]
                          << " "
                          << localDensityOfStatesDown[numberIntervals * iAtom +
                                                      epsInt]
                          << " ";
                        ;
                        checkSum +=
                          std::fabs(
                            localDensityOfStatesUp[numberIntervals * iAtom +
                                                   epsInt]) +
                          std::fabs(
                            localDensityOfStatesDown[numberIntervals * iAtom +
                                                     epsInt]);
                      }
                    outFile << std::endl;
                  }
              }
            else
              {
                for (unsigned int epsInt = 0; epsInt < numberIntervals;
                     ++epsInt)
                  {
                    double epsValue = lowerBoundEpsilon + epsInt * intervalSize;
                    outFile << std::setprecision(18) << epsValue * 27.21138602
                            << " ";
                    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms;
                         ++iAtom)
                      {
                        outFile
                          << std::setprecision(18)
                          << localDensityOfStates[numberIntervals * iAtom +
                                                  epsInt]
                          << " ";
                        checkSum += std::fabs(
                          localDensityOfStates[numberIntervals * iAtom +
                                               epsInt]);
                      }
                    outFile << std::endl;
                  }
              }
          }
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Absolute sum of all ldos values: " << checkSum << std::endl;

    computing_timer.leave_subsection("LDOS computation");
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::compute_pdos(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const std::string &                     pdosFileName)
  {
    computing_timer.enter_subsection("PDOS computation");

    //
    // create a stencil following orbital filling order
    //
    std::vector<unsigned int>              level;
    std::vector<std::vector<unsigned int>> stencil;

    //
    // create stencil in the order of single-atom orbital filling order
    //
    // 1s
    level.clear();
    level.push_back(1);
    level.push_back(0);
    stencil.push_back(level);
    // 2s
    level.clear();
    level.push_back(2);
    level.push_back(0);
    stencil.push_back(level);
    // 2p
    level.clear();
    level.push_back(2);
    level.push_back(1);
    stencil.push_back(level);
    // 3s
    level.clear();
    level.push_back(3);
    level.push_back(0);
    stencil.push_back(level);
    // 3p
    level.clear();
    level.push_back(3);
    level.push_back(1);
    stencil.push_back(level);
    // 4s
    level.clear();
    level.push_back(4);
    level.push_back(0);
    stencil.push_back(level);
    // 3d
    level.clear();
    level.push_back(3);
    level.push_back(2);
    stencil.push_back(level);
    // 4p
    level.clear();
    level.push_back(4);
    level.push_back(1);
    stencil.push_back(level);
    // 5s
    level.clear();
    level.push_back(5);
    level.push_back(0);
    stencil.push_back(level);
    // 4d
    level.clear();
    level.push_back(4);
    level.push_back(2);
    stencil.push_back(level);
    // 5p
    level.clear();
    level.push_back(5);
    level.push_back(1);
    stencil.push_back(level);
    // 6s
    level.clear();
    level.push_back(6);
    level.push_back(0);
    stencil.push_back(level);
    // 4f
    level.clear();
    level.push_back(4);
    level.push_back(3);
    stencil.push_back(level);
    // 5d
    level.clear();
    level.push_back(5);
    level.push_back(2);
    stencil.push_back(level);
    // 6p
    level.clear();
    level.push_back(6);
    level.push_back(1);
    stencil.push_back(level);
    // 7s
    level.clear();
    level.push_back(7);
    level.push_back(0);
    stencil.push_back(level);
    // 5f
    level.clear();
    level.push_back(5);
    level.push_back(3);
    stencil.push_back(level);
    // 6d
    level.clear();
    level.push_back(6);
    level.push_back(2);
    stencil.push_back(level);
    // 7p
    level.clear();
    level.push_back(7);
    level.push_back(1);
    stencil.push_back(level);
    // 8s
    level.clear();
    level.push_back(8);
    level.push_back(0);
    stencil.push_back(level);

    const unsigned int numberGlobalAtoms = atomLocations.size();

    unsigned int errorReadFile = 0;
    unsigned int fileReadFlag  = 0;

    std::map<unsigned int,
             std::map<unsigned int,
                      std::map<unsigned int, alglib::spline1dinterpolant>>>
                                      radValues;
    std::vector<std::vector<orbital>> singleAtomInfo;
    singleAtomInfo.resize(numberGlobalAtoms);
    double wfcInitTruncation;

    for (std::vector<std::vector<unsigned int>>::iterator it = stencil.begin();
         it < stencil.end();
         ++it)
      {
        unsigned int n = (*it)[0], l = (*it)[1];
        // Think of having "m" quantum number loop as well and push it into
        // atoms
        for (int m = -l; m <= (int)l; m++)
          {
            for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                unsigned int Z = atomLocations[iAtom][0];

                //
                // load PSI files
                //
                loadSingleAtomPSIFiles(Z,
                                       n,
                                       l,
                                       fileReadFlag,
                                       wfcInitTruncation,
                                       radValues,
                                       d_mpiCommParent,
                                       *d_dftParamsPtr);

                if (fileReadFlag > 0)
                  {
                    orbital temp;
                    temp.atomID = iAtom;
                    temp.Z      = Z;
                    temp.n      = n;
                    temp.l      = l;
                    temp.m      = m;
                    temp.psi    = radValues[Z][n][l];
                    singleAtomInfo[iAtom].push_back(temp);
                    // pcout << "Atom Id: "<<iAtom<<" Z: "<<Z<<" n: "<<n<<" l:
                    // "<<l<<" m: "<<m<<std::endl;
                  }
              }
          }

        if (fileReadFlag == 0)
          errorReadFile += 1;
      } // end stencil

    unsigned int totalAtomicData = 0;
    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        for (unsigned int iSingAtomData = 0;
             iSingAtomData < singleAtomInfo[iAtom].size();
             ++iSingAtomData)
          {
            totalAtomicData += 1;
          }
      }

    // loop over elements
    std::vector<double> eigenValuesAllkPoints;
    for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    double totalEigenValues  = eigenValuesAllkPoints.size();
    double intervalSize      = 0.001;
    double sigma             = C_kb * d_dftParamsPtr->TVal;
    double lowerBoundEpsilon = 1.5 * eigenValuesAllkPoints[0];
    double upperBoundEpsilon =
      eigenValuesAllkPoints[totalEigenValues - 1] * 1.5;

    unsigned int numberIntervals =
      std::ceil((upperBoundEpsilon - lowerBoundEpsilon) / intervalSize);
    std::vector<double> partialDensityOfStates;
    partialDensityOfStates.resize(totalAtomicData * numberIntervals, 0.0);

    // access finite-element data
    dealii::QGauss<3>   quadrature_formula(C_num1DQuad<FEOrder>());
    dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                  quadrature_formula,
                                  dealii::update_values |
                                    dealii::update_JxW_values |
                                    dealii::update_quadrature_points);
    const unsigned int  dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
    const unsigned int  n_q_points    = quadrature_formula.size();


    const unsigned int blockSize =
      std::min(d_dftParamsPtr->wfcBlockSize, d_numEigenValues);

    std::vector<std::vector<double>> tempQuadPointValuesForWaveFunctions(
      blockSize);
    std::vector<double> tempQuadPointValues(n_q_points);

    const unsigned int localVectorSize =
      matrix_free_data.get_vector_partitioner()->locally_owned_size();
    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());
    std::vector<distributedCPUVec<dataTypes::number>>
      eigenVectorsFlattenedBlock((1 + d_dftParamsPtr->spinPolarized) *
                                 d_kPointWeights.size());
    // std::vector<double>
    // innerProductWaveFunctionSingAtom(d_numEigenValues*5,0.0);
    // std::vector<double> tempContribution(blockSize*,0.0);

    for (unsigned int ivec = 0; ivec < d_numEigenValues; ivec += blockSize)
      {
        const unsigned int currentBlockSize =
          std::min(blockSize, d_numEigenValues - ivec);
        std::vector<double> tempContribution(currentBlockSize * totalAtomicData,
                                             0.0);

        if (currentBlockSize != blockSize || ivec == 0)
          {
            for (unsigned int kPoint = 0;
                 kPoint <
                 (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
                 ++kPoint)
              {
                eigenVectors[kPoint].resize(currentBlockSize);
                for (unsigned int i = 0; i < currentBlockSize; ++i)
                  eigenVectors[kPoint][i].reinit(d_tempEigenVec);


                vectorTools::createDealiiVector<dataTypes::number>(
                  matrix_free_data.get_vector_partitioner(),
                  currentBlockSize,
                  eigenVectorsFlattenedBlock[kPoint]);
                eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
              }

            constraintsNoneDataInfo.precomputeMaps(
              matrix_free_data.get_vector_partitioner(),
              eigenVectorsFlattenedBlock[0].get_partitioner(),
              currentBlockSize);
          }


        std::vector<std::vector<double>> blockedEigenValues(
          d_kPointWeights.size(),
          std::vector<double>((1 + d_dftParamsPtr->spinPolarized) *
                                currentBlockSize,
                              0.0));

        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
            {
              blockedEigenValues[kPoint][iWave] =
                eigenValues[kPoint][ivec + iWave];
              if (d_dftParamsPtr->spinPolarized == 1)
                blockedEigenValues[kPoint][currentBlockSize + iWave] =
                  eigenValues[kPoint][d_numEigenValues + ivec + iWave];
            }


        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          {
            for (unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
              for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
                eigenVectorsFlattenedBlock[kPoint].local_element(
                  iNode * currentBlockSize + iWave) =
                  d_eigenVectorsFlattenedHost[kPoint * localVectorSize *
                                                d_numEigenValues +
                                              iNode * d_numEigenValues + ivec +
                                              iWave];

            constraintsNoneDataInfo.distribute(
              eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
            eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              eigenVectors[kPoint],
              false);

            // FIXME: The underlying call to update_ghost_values
            // is required because currently localProc_dof_indicesReal
            // and localProc_dof_indicesImag are only available for
            // locally owned nodes. Once they are also made available
            // for ghost nodes- use true for the last argument in
            // copyFlattenedDealiiVecToSingleCompVec(..) above and supress
            // underlying call.
            for (unsigned int i = 0; i < currentBlockSize; ++i)
              eigenVectors[kPoint][i].update_ghost_values();
#else
            vectorTools::copyFlattenedDealiiVecToSingleCompVec(
              eigenVectorsFlattenedBlock[kPoint],
              currentBlockSize,
              std::make_pair(0, currentBlockSize),
              eigenVectors[kPoint],
              true);
#endif
          }

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                "PDOS is not implemented for spin-polarized problems"));
          }
        else
          {
            typename dealii::DoFHandler<3>::active_cell_iterator
              cellN = dofHandler.begin_active(),
              endcN = dofHandler.end();
            for (; cellN != endcN; ++cellN)
              {
                if (cellN->is_locally_owned())
                  {
                    fe_values.reinit(cellN);

                    for (unsigned int iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      {
                        tempQuadPointValuesForWaveFunctions[iEigenVec].resize(
                          n_q_points);
                        fe_values.get_function_values(
                          eigenVectors[0][iEigenVec], tempQuadPointValues);
                        tempQuadPointValuesForWaveFunctions[iEigenVec] =
                          tempQuadPointValues;
                      }



                    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms;
                         ++iAtom)
                      {
                        for (unsigned int iSingAtomData = 0;
                             iSingAtomData < singleAtomInfo[iAtom].size();
                             ++iSingAtomData)
                          {
                            for (unsigned int q = 0; q < n_q_points; ++q)
                              {
                                const dealii::Point<3> &quadPoint =
                                  fe_values.quadrature_point(q);
                                double x =
                                  quadPoint[0] - atomLocations[iAtom][2];
                                double y =
                                  quadPoint[1] - atomLocations[iAtom][3];
                                double z =
                                  quadPoint[2] - atomLocations[iAtom][4];

                                double r     = sqrt(x * x + y * y + z * z);
                                double theta = acos(z / r);
                                double phi   = atan2(y, x);

                                if (r == 0)
                                  {
                                    theta = 0;
                                    phi   = 0;
                                  }

                                orbital dataOrb =
                                  singleAtomInfo[iAtom][iSingAtomData];

                                double R = 0.0;
                                double singleAtomWaveFunctionQuadValue;

                                if (r <= wfcInitTruncation)
                                  {
                                    R = alglib::spline1dcalc(dataOrb.psi, r);
                                    if (dataOrb.m > 0)
                                      singleAtomWaveFunctionQuadValue =
                                        R * std::sqrt(2) *
                                        boost::math::spherical_harmonic_r(
                                          dataOrb.l, dataOrb.m, theta, phi);
                                    else if (dataOrb.m == 0)
                                      singleAtomWaveFunctionQuadValue =
                                        R * boost::math::spherical_harmonic_r(
                                              dataOrb.l, dataOrb.m, theta, phi);
                                    else
                                      singleAtomWaveFunctionQuadValue =
                                        R * std::sqrt(2) *
                                        boost::math::spherical_harmonic_i(
                                          dataOrb.l, -dataOrb.m, theta, phi);
                                  }
                                else
                                  singleAtomWaveFunctionQuadValue = 0.0;

                                for (unsigned int iEigenVec = 0;
                                     iEigenVec < currentBlockSize;
                                     ++iEigenVec)
                                  {
                                    tempContribution
                                      [currentBlockSize *
                                         singleAtomInfo[iAtom].size() * iAtom +
                                       currentBlockSize * iSingAtomData +
                                       iEigenVec] +=
                                      tempQuadPointValuesForWaveFunctions
                                        [iEigenVec][q] *
                                      singleAtomWaveFunctionQuadValue *
                                      fe_values.JxW(q);
                                  }
                              } // quad loop

                          } // single atom wavefunction data

                      } // iAtom data

                  } // if cell

              } // cell loop

            dealii::Utilities::MPI::sum(tempContribution,
                                        mpi_communicator,
                                        tempContribution);
          } // if-else loop


        for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
          {
            for (unsigned int iSingAtomData = 0;
                 iSingAtomData < singleAtomInfo[iAtom].size();
                 ++iSingAtomData)
              {
                for (unsigned int iEigenVec = 0; iEigenVec < currentBlockSize;
                     ++iEigenVec)
                  {
                    for (unsigned int epsInt = 0; epsInt < numberIntervals;
                         ++epsInt)
                      {
                        double epsValue =
                          lowerBoundEpsilon + epsInt * intervalSize;
                        double term1 =
                          (epsValue - blockedEigenValues[0][iEigenVec]);
                        double smearedEnergyLevel =
                          (sigma / M_PI) *
                          (1.0 / (term1 * term1 + sigma * sigma));
                        double tempValue =
                          tempContribution[currentBlockSize *
                                             singleAtomInfo[iAtom].size() *
                                             iAtom +
                                           currentBlockSize * iSingAtomData +
                                           iEigenVec];
                        partialDensityOfStates[numberIntervals *
                                                 singleAtomInfo[iAtom].size() *
                                                 iAtom +
                                               numberIntervals * iSingAtomData +
                                               epsInt] +=
                          2.0 * tempValue * tempValue * smearedEnergyLevel;
                      }
                  }
              }
          }

      } // ivec block loop

    pcout << "Following is the Single atom data used for PDOS computation: "
          << std::endl;

    for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        for (unsigned int iSingAtomData = 0;
             iSingAtomData < singleAtomInfo[iAtom].size();
             ++iSingAtomData)
          {
            orbital temp = singleAtomInfo[iAtom][iSingAtomData];
            pcout << "Atom Id: " << iAtom << " Z: " << temp.Z
                  << " n: " << temp.n << " l: " << temp.l << " m: " << temp.m
                  << std::endl;
          }
      }


    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::string tempFolder = "pdosOutputFolder";
        mkdir(tempFolder.c_str(), ACCESSPERMS);

        for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
          {
            std::string outFileName = tempFolder + "/" + pdosFileName + "_" +
                                      dealii::Utilities::to_string(iAtom);
            std::ofstream outputFile(outFileName);
            outputFile.setf(std::ios_base::fixed);
            if (outputFile.is_open())
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    AssertThrow(
                      false,
                      dealii::ExcMessage(
                        "PDOS is not implemented for spin-polarized problems"));
                  }
                else
                  {
                    for (unsigned int epsInt = 0; epsInt < numberIntervals;
                         ++epsInt)
                      {
                        double epsValue =
                          lowerBoundEpsilon + epsInt * intervalSize;
                        outputFile << std::setprecision(18)
                                   << epsValue * 27.21138602 << " ";
                        for (unsigned int iSingAtomData = 0;
                             iSingAtomData < singleAtomInfo[iAtom].size();
                             ++iSingAtomData)
                          {
                            outputFile
                              << std::setprecision(18)
                              << partialDensityOfStates
                                   [numberIntervals *
                                      singleAtomInfo[iAtom].size() * iAtom +
                                    numberIntervals * iSingAtomData + epsInt]
                              << " ";
                          }
                        outputFile << std::endl;
                      }
                  }
              }
          }
      }
    computing_timer.leave_subsection("PDOS computation");
  }
#include "dft.inst.cc"

} // namespace dftfe
