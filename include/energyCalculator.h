// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#include <xc.h>

#ifndef energyCalculator_H_
#define energyCalculator_H_

namespace dftfe {

 /**
  * @brief Calculates the ksdft problem total energy and its components
  *
  * @author Sambit Das, Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
  */
  class energyCalculator {

    public:

	/**
	 * @brief Constructor
	 *
	 * @param mpi_comm mpi communicator of domain decomposition
	 * @param interpool_comm mpi communicator of interpool communicator
	 */
	energyCalculator(const  MPI_Comm &mpi_comm,
		         const  MPI_Comm &interpool_comm);

	/**
	 * Computes total energy of the ksdft problem in the current state and also prints the
	 * individual components of the energy
	 *
	 * @param dofHandlerElectrostatic[in] p refined DoFHandler object used for re-computing
	 * the electrostatic fields using the ground state electron density. If electrostatics is
	 * not recomputed on p refined mesh, use dofHandlerElectronic for this argument.
	 * @param dofHandlerElectronic[in] DoFHandler object on which the electrostatics for the
	 * eigen solve are computed.
	 * @param quadratureElectrostatic[in] qudarature object for dofHandlerElectrostatic.
	 * @param quadratureElectronic[in] qudarature object for dofHandlerElectronic.
	 * @param eigenValues[in] eigenValues for each k point.
	 * @param kPointWeights[in]
	 * @param fermiEnergy[in]
	 * @param funcX[in] exchange functional object.
	 * @param funcC[in] correlation functional object.
	 * @param phiTotRhoIn[in] nodal vector field of total electrostatic potential using input
	 * electron density to an eigensolve. This vector field is based on dofHandlerElectronic.
	 * @param phiTotRhoOut[in] nodal vector field of total electrostatic potential using output
	 * electron density to an eigensolve. This vector field is based on dofHandlerElectrostatic.
	 * @param rhoInValues[in] cell quadrature data of input electron density to an eigensolve. This
	 * data must correspond to quadratureElectronic.
	 * @param rhoOutValues[in] cell quadrature data of output electron density of an eigensolve. This
	 * data must correspond to quadratureElectronic.
	 * @param rhoOutValuesElectrostatic[in] cell quadrature data of output electron density of an eigensolve
	 * evaluated on a p refined mesh. This data corresponds to quadratureElectrostatic.
	 * @param gradRhoInValues[in] cell quadrature data of input gradient electron density
	 * to an eigensolve. This data must correspond to quadratureElectronic.
	 * @param gradRhoOutValues[in] cell quadrature data of output gradient electron density
	 * of an eigensolve. This data must correspond to quadratureElectronic.
	 * @param localVselfs[in] peak vselfs of local atoms in each vself bin
	 * @param atomElectrostaticNodeIdToChargeMap[in] map between locally processor atom global node ids
	 * of dofHandlerElectrostatic to atom charge value.
	 * @param numberGlobalAtoms[in]
	 * @param lowerBoundKindex global k index of lower bound of the local k point set in the current pool
	 * @param print[in]
	 *
	 * @return total energy
	 */
	double computeEnergy(const dealii::DoFHandler<3> & dofHandlerElectrostatic,
		             const dealii::DoFHandler<3> & dofHandlerElectronic,
			     const dealii::QGauss<3> & quadratureElectrostatic,
                             const dealii::QGauss<3> & quadratureElectronic,
		             const std::vector<std::vector<double> > & eigenValues,
			     const std::vector<double> & kPointWeights,
		             const double fermiEnergy,
			     const xc_func_type & funcX,
			     const xc_func_type & funcC,
			     const vectorType & phiTotRhoIn,
			     const vectorType & phiTotRhoOut,
			     const std::map<dealii::CellId, std::vector<double> > & rhoInValues,
			     const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
			     const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectrostatic,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoInValues,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
		             const std::vector<std::vector<double> > & localVselfs,
		             const std::map<dealii::types::global_dof_index, double> & atomElectrostaticNodeIdToChargeMap,
			     const unsigned int numberGlobalAtoms,
			     const unsigned int lowerBoundKindex,
		             const bool print) const;

	/**
	 * Computes total energy of the spin polarized ksdft problem in the current state and also prints the
	 * individual components of the energy
	 *
	 * @param dofHandlerElectrostatic[in] p refined DoFHandler object used for re-computing
	 * the electrostatic fields using the ground state electron density. If electrostatics is
	 * not recomputed on p refined mesh, use dofHandlerElectronic for this argument.
	 * @param dofHandlerElectronic[in] DoFHandler object on which the electrostatics for the
	 * eigen solve are computed.
	 * @param quadratureElectrostatic[in] qudarature object for dofHandlerElectrostatic.
	 * @param quadratureElectronic[in] qudarature object for dofHandlerElectronic.
	 * @param eigenValues[in] eigenValues for each k point.
	 * @param kPointWeights[in]
	 * @param fermiEnergy[in]
	 * @param funcX[in] exchange functional object.
	 * @param funcC[in] correlation functional object.
	 * @param phiTotRhoIn[in] nodal vector field of total electrostatic potential using input
	 * electron density to an eigensolve. This vector field is based on dofHandlerElectronic.
	 * @param phiTotRhoOut[in] nodal vector field of total electrostatic potential using output
	 * electron density to an eigensolve. This vector field is based on dofHandlerElectrostatic.
	 * @param rhoInValues[in] cell quadrature data of input electron density to an eigensolve. This
	 * data must correspond to quadratureElectronic.
	 * @param rhoOutValues[in] cell quadrature data of output electron density of an eigensolve. This
	 * data must correspond to quadratureElectronic.
	 * @param rhoOutValuesElectrostatic[in] cell quadrature data of output electron density of an eigensolve
	 * evaluated on a p refined mesh. This data corresponds to quadratureElectrostatic.
	 * @param gradRhoInValues[in] cell quadrature data of input gradient electron density
	 * to an eigensolve. This data must correspond to quadratureElectronic.
	 * @param gradRhoOutValues[in] cell quadrature data of output gradient electron density
	 * of an eigensolve. This data must correspond to quadratureElectronic.
	 * @param rhoInValuesSpinPolarized[in] cell quadrature data of input spin polarized
	 * electron density to an eigensolve. This data must correspond to quadratureElectronic.
	 * @param rhoOutValuesSpinPolarized[in] cell quadrature data of output spin polarized
	 * electron density of an eigensolve. This data must correspond to quadratureElectronic.
	 * @param gradRhoInValuesSpinPolarized[in] cell quadrature data of input gradient spin polarized
	 * electron density to an eigensolve. This data must correspond to quadratureElectronic.
	 * @param gradRhoOutValuesSpinPolarized[in] cell quadrature data of output gradient spin polarized
	 * electron density of an eigensolve. This data must correspond to quadratureElectronic.
	 * @param localVselfs[in] peak vselfs of local atoms in each vself bin
	 * @param atomElectrostaticNodeIdToChargeMap[in] map between locally processor atom global node ids
	 * of dofHandlerElectrostatic to atom charge value.
	 * @param numberGlobalAtoms[in]
	 * @param lowerBoundKindex global k index of lower bound of the local k point set in the current pool
	 * @param print[in]
	 *
	 * @return total energy
	 */
	double computeEnergySpinPolarized
	                    (const dealii::DoFHandler<3> & dofHandlerElectrostatic,
			     const dealii::DoFHandler<3> & dofHandlerElectronic,
			     const dealii::QGauss<3> & quadratureElectrostatic,
                             const dealii::QGauss<3> & quadratureElectronic,
			     const std::vector<std::vector<double> > & eigenValues,
			     const std::vector<double> & kPointWeights,
			     const double fermiEnergy,
			     const xc_func_type & funcX,
			     const xc_func_type & funcC,
			     const vectorType & phiTotRhoIn,
			     const vectorType & phiTotRhoOut,
			     const std::map<dealii::CellId, std::vector<double> > & rhoInValues,
			     const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
			     const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectrostatic,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoInValues,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
			     const std::map<dealii::CellId, std::vector<double> > & rhoInValuesSpinPolarized,
			     const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesSpinPolarized,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoInValuesSpinPolarized,
			     const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesSpinPolarized,
			     const std::vector<std::vector<double> > & localVselfs,
			     const std::map<dealii::types::global_dof_index, double> & atomElectrostaticNodeIdToChargeMap,
			     const unsigned int numberGlobalAtoms,
			     const unsigned int lowerBoundKindex,
			     const bool print) const;

     private:

         const MPI_Comm mpi_communicator;
	 const MPI_Comm interpoolcomm;

	 /// parallel message stream
         dealii::ConditionalOStream  pcout;

  };

}
#endif // energyCalculator_H_
