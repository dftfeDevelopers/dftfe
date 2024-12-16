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

// source file for restart functionality in dftClass

//
//
#include <dft.h>
#include <densityCalculator.h>
#include <kineticEnergyDensityCalculator.h>
#include <fileReaders.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <QuadDataCompositeWrite.h>
#include <MPIWriteOnFile.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeAndPrintKE(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &kineticEnergyDensityValues)
  {
    const dealii::Quadrature<3> &quadratureFormula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    const unsigned int n_q_points = quadratureFormula.size();

    //
    // compute kinetic energy density values
    //
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      computeKineticEnergyDensity(*d_BLASWrapperPtr,
                                  &d_eigenVectorsFlattenedDevice,
                                  d_numEigenValues,
                                  eigenValues,
                                  fermiEnergy,
                                  fermiEnergyUp,
                                  fermiEnergyDown,
                                  d_basisOperationsPtrDevice,
                                  d_densityQuadratureId,
                                  d_kPointCoordinates,
                                  d_kPointWeights,
                                  kineticEnergyDensityValues,
                                  d_mpiCommParent,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  mpi_communicator,
                                  *d_dftParamsPtr);
#endif
    if (!d_dftParamsPtr->useDevice)
      computeKineticEnergyDensity(*d_BLASWrapperPtrHost,
                                  &d_eigenVectorsFlattenedHost,
                                  d_numEigenValues,
                                  eigenValues,
                                  fermiEnergy,
                                  fermiEnergyUp,
                                  fermiEnergyDown,
                                  d_basisOperationsPtrHost,
                                  d_densityQuadratureId,
                                  d_kPointCoordinates,
                                  d_kPointWeights,
                                  kineticEnergyDensityValues,
                                  d_mpiCommParent,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  mpi_communicator,
                                  *d_dftParamsPtr);

    MPI_Barrier(MPI_COMM_WORLD);



    double kineticEnergy = 0;


    dealii::FEValues<3> feValues(dofHandler.get_fe(),
                                 quadratureFormula,
                                 dealii::update_values |
                                   dealii::update_JxW_values |
                                   dealii::update_quadrature_points);



    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();

    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          feValues.reinit(cell);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const dealii::Point<3> &quadPoint =
                feValues.quadrature_point(q_point);
              const double jxw = feValues.JxW(q_point);

              kineticEnergy += kineticEnergyDensityValues
                                 .data()[iElem * n_q_points + q_point] *
                               jxw;
            }
          iElem++;
        }


    MPI_Allreduce(
      MPI_IN_PLACE, &kineticEnergy, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);


    pcout << "kinetic energy: " << kineticEnergy << std::endl;

    return kineticEnergy;
  }

#include "dft.inst.cc"
} // namespace dftfe
