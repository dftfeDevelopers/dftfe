// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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

#ifndef densityCalculator_H_
#define densityCalculator_H_

#include <headers.h>
#include <constants.h>

namespace dftfe
{
	/**
	 * @brief Density calculator class
	 *
	 * @author Sambit Das
	 */
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		class DensityCalculator
		{

			public:

				/**
				 * @brief Constructor
				 *
				 * @param mpi_comm mpi communicator
				 */
				DensityCalculator();


				void computeRhoFromPSI
					(const std::vector<std::vector<dataTypes::number> > & eigenVectorsInput,
					 const std::vector<std::vector<dataTypes::number> > & eigenVectorsFracInput,
					 const unsigned int totalNumWaveFunctions,
					 const unsigned int Nfr,
					 const std::vector<std::vector<double>> & eigenValues,
					 const double fermiEnergy, 
					 const double fermiEnergyUp,
					 const double fermiEnergyDown,
					 const dealii::DoFHandler<3> & dofHandler,
					 const dealii::AffineConstraints<double> & constraints,
					 const dealii::MatrixFree<3,double> & mfData,
					 const unsigned int mfDofIndex,
					 const unsigned int mfQuadIndex,
					 const std::vector<dealii::types::global_dof_index> & localProc_dof_indicesReal,
					 const std::vector<dealii::types::global_dof_index> & localProc_dof_indicesImag,
					 const std::vector<double> & kPointWeights,
					 std::map<dealii::CellId, std::vector<double> > * _rhoValues,
					 std::map<dealii::CellId, std::vector<double> > * _gradRhoValues,
					 std::map<dealii::CellId, std::vector<double> > * _rhoValuesSpinPolarized,
					 std::map<dealii::CellId, std::vector<double> > * _gradRhoValuesSpinPolarized,
					 const bool isEvaluateGradRho,
					 const MPI_Comm & interpoolcomm,
					 const MPI_Comm & interBandGroupComm,
					 const bool isConsiderSpectrumSplitting,
					 const bool lobattoNodesFlag);
		};
}
#endif
