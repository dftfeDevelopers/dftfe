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
// @author Sambit Das
//

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  accumulateForceContributionGammaAtomsFloating(
    const std::map<unsigned int, std::vector<double>>
      &                  forceContributionLocalGammaAtoms,
    std::vector<double> &accumForcesVector)
{
  for (unsigned int iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
    {
      std::vector<double> forceContributionLocalGammaiAtomGlobal(C_DIM);
      std::vector<double> forceContributionLocalGammaiAtomLocal(C_DIM, 0.0);

      if (forceContributionLocalGammaAtoms.find(iAtom) !=
          forceContributionLocalGammaAtoms.end())
        forceContributionLocalGammaiAtomLocal =
          forceContributionLocalGammaAtoms.find(iAtom)->second;
      // accumulate value
      MPI_Allreduce(&(forceContributionLocalGammaiAtomLocal[0]),
                    &(forceContributionLocalGammaiAtomGlobal[0]),
                    3,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      for (unsigned int idim = 0; idim < 3; idim++)
        accumForcesVector[iAtom * 3 + idim] +=
          forceContributionLocalGammaiAtomGlobal[idim];
    }
}
