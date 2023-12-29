// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#include <headers.h>
#include <dftd.h>
#include <excManager.h>
#include "dftParameters.h"
#include <FEBasisOperations.h>
#ifndef energyCalculator_H_
#  define energyCalculator_H_

namespace dftfe
{
  /**
   * @brief Calculates the ksdft problem total energy and its components
   *
   * @author Sambit Das, Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
   */
  class energyCalculator
  {
  public:
    /**
     * @brief Constructor
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain mpi communicator of domain decomposition
     * @param interpool_comm mpi interpool communicator over k points
     * @param interBandGroupComm mpi interpool communicator over band groups
     */
    energyCalculator(const MPI_Comm &     mpi_comm_parent,
                     const MPI_Comm &     mpi_comm_domain,
                     const MPI_Comm &     interpool_comm,
                     const MPI_Comm &     interBandGroupComm,
                     const dftParameters &dftParams);

    /**
     * Computes total energy of the ksdft problem in the current state and also
     * prints the individual components of the energy
     *
     * @param dofHandlerElectrostatic p refined DoFHandler object used for re-computing
     * the electrostatic fields using the ground state electron density. If
     * electrostatics is not recomputed on p refined mesh, use
     * dofHandlerElectronic for this argument.
     * @param dofHandlerElectronic DoFHandler object on which the electrostatics for the
     * eigen solve are computed.
     * @param quadratureElectrostatic qudarature object for dofHandlerElectrostatic.
     * @param quadratureElectronic qudarature object for dofHandlerElectronic.
     * @param eigenValues eigenValues for each k point.
     * @param kPointWeights
     * @param fermiEnergy
     * @param funcX exchange functional object.
     * @param funcC correlation functional object.
     * @param phiTotRhoIn nodal vector field of total electrostatic potential using input
     * electron density to an eigensolve. This vector field is based on
     * dofHandlerElectronic.
     * @param phiTotRhoOut nodal vector field of total electrostatic potential using output
     * electron density to an eigensolve. This vector field is based on
     * dofHandlerElectrostatic.
     * @param rhoInValues cell quadrature data of input electron density to an eigensolve. This
     * data must correspond to quadratureElectronic.
     * @param rhoOutValues cell quadrature data of output electron density of an eigensolve. This
     * data must correspond to quadratureElectronic.
     * @param rhoOutValuesElectrostatic cell quadrature data of output electron density of an eigensolve
     * evaluated on a p refined mesh. This data corresponds to
     * quadratureElectrostatic.
     * @param gradRhoInValues cell quadrature data of input gradient electron density
     * to an eigensolve. This data must correspond to quadratureElectronic.
     * @param gradRhoOutValues cell quadrature data of output gradient electron density
     * of an eigensolve. This data must correspond to quadratureElectronic.
     * @param localVselfs peak vselfs of local atoms in each vself bin
     * @param atomElectrostaticNodeIdToChargeMap map between locally processor atom global node ids
     * of dofHandlerElectrostatic to atom charge value.
     * @param numberGlobalAtoms
     * @param lowerBoundKindex global k index of lower bound of the local k point set in the current pool
     * @param if scf is converged
     * @param print
     *
     * @return total energy
     */
    double
    computeEnergy(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                                     basisOperationsPtrElectro,
      const unsigned int                      densityQuadratureID,
      const unsigned int                      densityQuadratureIDElectro,
      const unsigned int                      smearedChargeQuadratureIDElectro,
      const unsigned int                      lpspQuadratureIDElectro,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<double> &             kPointWeights,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      const excManager *                      excManagerPtr,
      const dispersionCorrection &            dispersionCorr,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiTotRhoInValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                              phiTotRhoOutValues,
      const distributedCPUVec<double> &phiTotRhoOut,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  rhoOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<unsigned int>>
        &                                     smearedbNonTrivialAtomIds,
      const std::vector<std::vector<double>> &localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
      const std::map<dealii::types::global_dof_index, double>
        &                atomElectrostaticNodeIdToChargeMap,
      const unsigned int numberGlobalAtoms,
      const unsigned int lowerBoundKindex,
      const unsigned int scfConverged,
      const bool         print,
      const bool         smearedNuclearCharges = false);


    double
    computeXCEnergyTermsSpinPolarized(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureId,
      const excManager * excManagerPtr,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  gradDensityOutValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      double &                                             exchangeEnergy,
      double &                                             correlationEnergy,
      double &excCorrPotentialTimesRho);

    double
    computeXCEnergyTerms(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureId,
      const excManager * excManagerPtr,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  gradDensityOutValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      double &                                             exchangeEnergy,
      double &                                             correlationEnergy,
      double &excCorrPotentialTimesRho);


    double
    computeEntropicEnergy(const std::vector<std::vector<double>> &eigenValues,
                          const std::vector<double> &             kPointWeights,
                          const double                            fermiEnergy,
                          const double                            fermiEnergyUp,
                          const double fermiEnergyDown,
                          const bool   isSpinPolarized,
                          const bool   isConstraintMagnetization,
                          const double temperature) const;



  private:
    const MPI_Comm d_mpiCommParent;
    const MPI_Comm mpi_communicator;
    const MPI_Comm interpoolcomm;
    const MPI_Comm interBandGroupComm;

    const dftParameters &d_dftParams;

    /// parallel message stream
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // energyCalculator_H_
