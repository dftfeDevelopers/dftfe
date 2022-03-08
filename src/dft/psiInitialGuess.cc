// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::loadPSIFiles(unsigned int  Z,
                                                unsigned int  n,
                                                unsigned int  l,
                                                unsigned int &fileReadFlag)
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

  if (dftParameters::isPseudopotential)
    // if(dftParameters::pseudoProjector==2)
    // sprintf(psiFile,
    // "%s/data/electronicStructure/pseudoPotential/z%u/oncv/singleAtomData/psi%u%u.inp",
    // DFT_PATH, Z, n, l); else
    sprintf(
      psiFile,
      "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp",
      DFT_PATH,
      Z,
      n,
      l);

  else
    sprintf(
      psiFile,
      "%s/data/electronicStructure/allElectron/z%u/singleAtomData/psi%u%u.inp",
      DFT_PATH,
      Z,
      n,
      l);

  std::vector<std::vector<double>> values;

  fileReadFlag = dftUtils::readPsiFile(2, values, psiFile);

  const double truncationTol =
    dftParameters::reproducible_output ? 1e-10 : 1e-8;
  //
  // spline fitting for single-atom wavefunctions
  //
  if (fileReadFlag > 0)
    {
      double       maxTruncationRadius = 0.0;
      unsigned int truncRowId          = 0;
      if (!dftParameters::reproducible_output)
        pcout << "reading data from file: " << psiFile << std::endl;

      int                 numRows = values.size() - 1;
      std::vector<double> xData(numRows), yData(numRows);

      // x
      for (int irow = 0; irow < numRows; ++irow)
        {
          xData[irow] = values[irow][0];
        }
      outerValues[Z][n][l] = xData[numRows - 1];
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
      alglib::ae_int_t             natural_bound_type = 0;
      alglib::spline1dinterpolant *spline = new alglib::spline1dinterpolant;
      alglib::spline1dbuildcubic(x,
                                 y,
                                 numRows,
                                 natural_bound_type,
                                 0.0,
                                 natural_bound_type,
                                 0.0,
                                 *spline);

      radValues[Z][n][l] = spline;

      maxTruncationRadius = xData[truncRowId];
      if (maxTruncationRadius > d_wfcInitTruncation)
        d_wfcInitTruncation = maxTruncationRadius;
    }
}

//
// determine orbital ordering
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::determineOrbitalFilling()
{
  //
  // create a stencil following orbital filling order
  //
  std::vector<unsigned int>              level;
  std::vector<std::vector<unsigned int>> stencil;

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



  const unsigned int numberGlobalAtoms  = atomLocations.size();
  const int          numberImageCharges = d_imageIds.size();
  const int          totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

  unsigned int errorReadFile            = 0;
  unsigned int fileReadFlag             = 0;
  unsigned int waveFunctionCount        = 0;
  unsigned int totalNumberWaveFunctions = d_numEigenValues;

  for (std::vector<std::vector<unsigned int>>::iterator it = stencil.begin();
       it < stencil.end();
       it++)
    {
      unsigned int n = (*it)[0], l = (*it)[1];

      for (int m = -l; m <= (int)l; m++)
        {
          for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
            {
              unsigned int Z = atomLocations[iAtom][0];

              //
              // fill levels
              //
              if (radValues.count(Z) == 0)
                {
                  pcout << "Z:" << Z << std::endl;
                }

              //
              // load PSI files
              //
              loadPSIFiles(Z, n, l, fileReadFlag);

              if (fileReadFlag > 0)
                {
                  orbital temp;
                  temp.atomID = iAtom;
                  temp.Z      = Z;
                  temp.n      = n;
                  temp.l      = l;
                  temp.m      = m;
                  temp.psi    = radValues[Z][n][l];
                  temp.waveID = waveFunctionCount;
                  waveFunctionsVector.push_back(temp);
                  waveFunctionCount++;
                  if (waveFunctionCount >= d_numEigenValues &&
                      waveFunctionCount >= numberGlobalAtoms)
                    break;
                }
            }

          if (waveFunctionCount >= d_numEigenValues &&
              waveFunctionCount >= numberGlobalAtoms)
            break;
        }

      if (waveFunctionCount >= d_numEigenValues &&
          waveFunctionCount >= numberGlobalAtoms)
        break;

      if (fileReadFlag == 0)
        errorReadFile += 1;
    }


  if (waveFunctionsVector.size() > d_numEigenValues)
    {
      d_numEigenValues = waveFunctionsVector.size();
    }

  pcout
    << "============================================================================================================================="
    << std::endl;
  pcout << "number of electrons: " << numElectrons << std::endl;
  pcout << "number of eigen values: " << d_numEigenValues << std::endl;

  if (dftParameters::verbosity >= 1)
    pcout
      << "number of wavefunctions computed using single atom data to be used as initial guess for starting the SCF: "
      << waveFunctionCount << std::endl;

  if (errorReadFile == stencil.size())
    {
      // std::cerr<< "Error: Require single-atom wavefunctions as initial guess
      // for starting the SCF."<< std::endl; std::cerr<< "Error: Could not find
      // single-atom wavefunctions for any atom: "<< std::endl;
      if (dftParameters::verbosity >= 1)
        pcout
          << "CAUTION: Could not find single-atom wavefunctions for any atom- the starting guess for all wavefunctions will be random."
          << std::endl;
      // exit(-1);
    }
  pcout
    << "============================================================================================================================="
    << std::endl;
}

//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::readPSIRadialValues()
{
  const IndexSet &locallyOwnedSet = dofHandler.locally_owned_dofs();
  std::vector<IndexSet::size_type> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  unsigned int numberDofs = locallyOwnedDOFs.size();

  for (unsigned int kPoint = 0;
       kPoint < (1 + dftParameters::spinPolarized) * d_kPointWeights.size();
       ++kPoint)
    {
      std::fill(d_eigenVectorsFlattenedSTL[kPoint].begin(),
                d_eigenVectorsFlattenedSTL[kPoint].end(),
                0.0);
    }

  const unsigned int numberGlobalAtoms = atomLocations.size();

  if (dftParameters::verbosity >= 1)
    pcout
      << "Number of wavefunctions generated randomly to be used as initial guess for starting the SCF : "
      << d_numEigenValues - waveFunctionsVector.size() << std::endl;
  //
  // loop over nodes
  //

  std::vector<orbital>   waveFunctionsVectorTruncated;
  dealii::BoundingBox<3> boundingBoxTria(
    vectorTools::createBoundingBoxTriaLocallyOwned(dofHandler));
  dealii::Tensor<1, 3, double> tempDisp;
  tempDisp[0] = d_wfcInitTruncation;
  tempDisp[1] = d_wfcInitTruncation;
  tempDisp[2] = d_wfcInitTruncation;

  for (std::vector<orbital>::iterator it = waveFunctionsVector.begin();
       it < waveFunctionsVector.end();
       it++)
    {
      const unsigned int chargeId = it->atomID;
      Point<3>           atomCoord;

      if (chargeId < atomLocations.size())
        {
          atomCoord[0] = atomLocations[chargeId][2];
          atomCoord[1] = atomLocations[chargeId][3];
          atomCoord[2] = atomLocations[chargeId][4];
        }
      else
        {
          atomCoord[0] = d_imagePositions[chargeId - numberGlobalAtoms][0];
          atomCoord[1] = d_imagePositions[chargeId - numberGlobalAtoms][1];
          atomCoord[2] = d_imagePositions[chargeId - numberGlobalAtoms][2];
        }


      std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
        boundaryPoints;
      boundaryPoints.first  = atomCoord - tempDisp;
      boundaryPoints.second = atomCoord + tempDisp;
      dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

      if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
          NeighborType::not_neighbors)
        ;
      waveFunctionsVectorTruncated.push_back(*it);
    }

  boost::math::normal normDist;
  bool                pp = false;
  std::srand(this_mpi_process);
  for (unsigned int dof = 0; dof < numberDofs; dof++)
    {
      const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
      Point<3>                              node  = d_supportPoints[dofID];
      if (!constraintsNone.is_constrained(dofID))
        {
          //
          // loop over wave functions
          //
          for (int kPoint = 0; kPoint < (1 + dftParameters::spinPolarized) *
                                          d_kPointWeights.size();
               ++kPoint)
            {
              // unsigned int waveFunction=0;
              for (std::vector<orbital>::iterator it =
                     waveFunctionsVectorTruncated.begin();
                   it < waveFunctionsVectorTruncated.end();
                   it++)
                {
                  //
                  // get the imageIdmap information corresponding to
                  // globalChargeId (Fix me: Examine whether periodic image
                  // contributions have to be included or not) currently not
                  // including
                  std::vector<int> imageIdsList;
                  if (dftParameters::periodicX || dftParameters::periodicY ||
                      dftParameters::periodicZ)
                    {
                      imageIdsList = d_globalChargeIdToImageIdMap[it->atomID];
                    }
                  else
                    {
                      imageIdsList.push_back(it->atomID);
                    }

                  const unsigned int waveId = it->waveID;
                  for (int iImageAtomCount = 0;
                       iImageAtomCount < imageIdsList.size();
                       ++iImageAtomCount)
                    {
                      //
                      // find coordinates of atom correspoding to this wave
                      // function and imageAtom
                      //
                      int      chargeId = imageIdsList[iImageAtomCount];
                      Point<3> atomCoord;

                      if (chargeId < numberGlobalAtoms)
                        {
                          atomCoord[0] = atomLocations[chargeId][2];
                          atomCoord[1] = atomLocations[chargeId][3];
                          atomCoord[2] = atomLocations[chargeId][4];
                        }
                      else
                        {
                          atomCoord[0] =
                            d_imagePositions[chargeId - numberGlobalAtoms][0];
                          atomCoord[1] =
                            d_imagePositions[chargeId - numberGlobalAtoms][1];
                          atomCoord[2] =
                            d_imagePositions[chargeId - numberGlobalAtoms][2];
                        }

                      double x = node[0] - atomCoord[0];
                      double y = node[1] - atomCoord[1];
                      double z = node[2] - atomCoord[2];


                      double r     = sqrt(x * x + y * y + z * z);
                      double theta = acos(z / r);
                      double phi   = atan2(y, x);

                      if (r == 0)
                        {
                          theta = 0;
                          phi   = 0;
                        }

                      double R = 0.0;
                      if (
                        r <=
                        d_wfcInitTruncation) // outerValues[it->Z][it->n][it->l])
                        {
                          // radial part
                          R = alglib::spline1dcalc(*(it->psi), r);
                          // spherical part
                          if (it->m > 0)
                            {
                              d_eigenVectorsFlattenedSTL
                                [kPoint][dof * d_numEigenValues + waveId] +=
                                dataTypes::number(
                                  R * std::sqrt(2) *
                                  boost::math::spherical_harmonic_r(
                                    it->l, it->m, theta, phi));
                            }
                          else if (it->m == 0)
                            {
                              d_eigenVectorsFlattenedSTL
                                [kPoint][dof * d_numEigenValues + waveId] +=
                                dataTypes::number(
                                  R * boost::math::spherical_harmonic_r(
                                        it->l, it->m, theta, phi));
                            }
                          else
                            {
                              d_eigenVectorsFlattenedSTL
                                [kPoint][dof * d_numEigenValues + waveId] +=
                                dataTypes::number(
                                  R * std::sqrt(2) *
                                  boost::math::spherical_harmonic_i(
                                    it->l, -(it->m), theta, phi));
                            }
                        }
                    }
                  // waveFunction++;
                }

              d_nonAtomicWaveFunctions = 0;
              if (waveFunctionsVector.size() < d_numEigenValues)
                {
                  d_nonAtomicWaveFunctions =
                    d_numEigenValues - waveFunctionsVector.size();

                  //
                  // assign the rest of the wavefunctions using a standard
                  // normal distribution
                  //
                  // boost::math::normal normDist;

                  std::vector<dataTypes::number> &temp =
                    d_eigenVectorsFlattenedSTL[kPoint];
                  for (unsigned int iWave = waveFunctionsVector.size();
                       iWave < d_numEigenValues;
                       ++iWave)
                    {
                      double z     = (-0.5 + (rand() + 0.0) / (RAND_MAX)) * 3.0;
                      double value = boost::math::pdf(normDist, z);
                      if (rand() % 2 == 0)
                        value = -1.0 * value;

                      temp[dof * d_numEigenValues + iWave] =
                        dataTypes::number(value);
                    }
                }
            }
        }
    }

  // for(int kPoint = 0; kPoint <
  // (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
  //{
  //   d_eigenVectorsFlattened[kPoint].compress(VectorOperation::insert);
  //      d_eigenVectorsFlattened[kPoint].update_ghost_values();
  //  }

  if (dftParameters::startingWFCType == "RANDOM")
    {
      pcout
        << "============================================================================================================================="
        << std::endl;
      pcout << "number of electrons: " << numElectrons << std::endl;
      pcout << "number of eigen values: " << d_numEigenValues << std::endl;
      pcout
        << "============================================================================================================================="
        << std::endl;
    }
}

//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::readPSI()
{
  computing_timer.enter_subsection("initialize wave functions");
  readPSIRadialValues();
  computing_timer.leave_subsection("initialize wave functions");
}
