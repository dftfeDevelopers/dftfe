// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXACTEXCHANGECLASS_H
#define DFTFE_EXACTEXCHANGECLASS_H


#include "ExcSSDFunctionalBaseClass.h"
#include "MultiVectorLinearSolverProblem.h"
#include "MultiVectorCGSolver.h"
#include "constraintMatrixInfo.h"
#include "FEBasisOperations.h"
#include "MultiVectorPoissonLinearSolverProblem.h"
#include "constraintMatrixInfo.h"
#include "dftParameters.h"

#include "lapack_support.h"

#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <deviceKernelsGeneric.h>
#include <DeviceTypeConfig.h>
#include <DeviceDataTypeOverloads.h>

namespace dftfe
{

//TODO move this to blas wrapper	
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  extern "C"
  {
	  void
    dpotrf_(const char *        uplo,
            const unsigned int *n,
            double *            a,
            const unsigned int *lda,
            int *               info);


	  void
    dtrtri_(const char *        uplo,
            const char *        diag,
            const unsigned int *n,
            double *            a,
            const unsigned int *lda,
            int *               info);
  }

#endif 


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class ExactExchange
  {
  public:

    ExactExchange(const bool useTucker,
                const double factorOfExactExchange,
                const MPI_Comm &mpi_comm_parent,
                const MPI_Comm &mpi_comm_domain);


    void reinit(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> BLASWrapperPtr,
                                               std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> BLASWrapperHostPtr,
                                               std::shared_ptr<
                                                 dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                                 basisOperationsPtrPsi,
                                               std::shared_ptr<
                                                 dftfe::basis::FEBasisOperations<double, double, memorySpace>>
                                                                                         basisOperationsPtrElectroNoDBC,
											 dealii::MatrixFree<3, double> &matrixFreeDataPRefined,
                                               const dealii::AffineConstraints<double> &constraintMatrixPsi,
                                               std::vector<const dealii::AffineConstraints<double> *> & constraintMatrixSingleElectro,
                                               const dealii::AffineConstraints<double> &constraintMatrixSingleElectroHangingPeriodic,
                                               const dealii::AffineConstraints<double> &constraintMatrixSingleElectroHangingPeriodicHomogenous,
                                               const unsigned int                       matrixFreeVectorPsiComponent,
                                               const unsigned int                       matrixFreeQuadratureComponentPsiRhs,
                                               const unsigned int                       matrixFreeVectorElectroComponentNoDBC,
                                                const unsigned int                       matrixFreeVectorElectroComponentDBC,
                                               const unsigned int                       matrixFreeQuadratureComponentElectroRhs,
                                               const unsigned int                       matrixFreeQuadratureComponentElectroAX,
                                               const unsigned int                       numKpoints,
                                               const unsigned int                       numSpins,
                                               const unsigned int                       numWaveFunctions,
                                               const dftParameters &                   dftParam);
	

    void updateFockExachageFactor(double factorOfExactExchange);    
    void applyPoissonExactExchangeOperator( dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                                                        & src,
                                      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
                                      const unsigned int inputVecSize,
                                      const double factor);

    void computeApplyAtQuadPointsUsingPoisson(
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> & src,
      dftfe::utils::MemoryStorage<ValueType,memorySpace> &VFockQuadInputValues,
      dftfe::utils::MemoryStorage<ValueType,memorySpace> &cellLeveQuadValuesInput,
      const unsigned int inputPsiSize,
      const unsigned int kPointIndex,
      const unsigned int spinIndex);


    void applyTuckerExactExchangeOperator();

    void applyACEOperator(const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
                     dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>      &dst,
                     const unsigned int inputPsiSize,
                     const double factor);

    void updatePoissonExactExchangeOperator(const dftfe::utils::MemoryStorage<ValueType,memorySpace> &eigenVectors,
                                       const std::vector<std::vector<double>> & partialOccupancies);

    void updateACEOperator (const dftfe::utils::MemoryStorage<ValueType,memorySpace> &eigenVectors,
                      const std::vector<std::vector<double>> & partialOccupancies,
		      const bool updateOperator = true);

    void updateTuckerExactExchangeOperator();

    double getExactExchangeEnergy();

    double getExpectationOfExactExchangeOperator();


    double computeExactE_HFQuad(
      const dftfe::utils::MemoryStorage<ValueType,memorySpace> &wavefunctionInpuVec,
      dftfe::utils::MemoryStorage<ValueType,memorySpace> &VFockQuadInputValues,
      dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> & singleParticleExchangeEnergy);


  private:

    double d_exactExchangeEnergy;
    double d_expectationOfExactExchange;


    double d_factorOfExactExchange;

    double d_fockEnergy;

    MultiVectorPoissonLinearSolverProblem<memorySpace>  d_multiVectorPoissonSolver;

    MultiVectorCGSolver  d_multiVectorCGsolver;

    std::vector<dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>>
      d_FockOperatorVector ,  d_psiInputVectorForAce;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> d_multiVectorOutputDBC, d_boundaryValuesDBC ;
    dftfe::dftUtils::constraintMatrixInfo<memorySpace> d_constraintsMatrixOperatorDataInfo;

    dftfe::utils::MemoryStorage<ValueType,memorySpace> d_cellLeveQuadValuesInput,
      d_cellLevelQuadValuesOperator, d_cellLevelQuadValuesInputOperator;

    dftfe::utils::MemoryStorage<ValueType,memorySpace> d_mMatrixMemSpace;
    dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST> d_mMatrixHost;

    std::vector<dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>> d_orbitalOccupancyOperatorHost;
    std::vector<dftfe::utils::MemoryStorage<ValueType,memorySpace>> d_orbitalOccupancyOperatorMemSpace;
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> d_ACEOperatorVecMemSpace;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>> d_BasisOperatorMemPtrPsi, d_BasisOperatorMemPtrElectroNoDBC;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>> d_BasisOperatorMemPtrElectroDBC;


    const MPI_Comm mpi_communicator_parent; 
    const MPI_Comm mpi_communicator; 
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process; 

    dealii::ConditionalOStream pcout;

    bool d_useTucker;
    unsigned int  d_blockSizeInputVector;
    unsigned int  d_blockSizePoissonSolver; 

    const dealii::MatrixFree<3,double>  * d_matrixFreeDataPtrPsi;
    unsigned int d_matrixFreePsiVectorComponent;
    unsigned int d_matrixFreePsiQuadratureComponentRhs;
    const dealii::MatrixFree<3,double>  *d_matrixFreeDataPtrElectro;

    unsigned int d_matrixFreeElectroVectorComponentNoDBC, d_matrixFreeElectroVectorComponentDBC;
    unsigned int d_matrixFreeElectroQuadratureComponentRhs;
    unsigned int d_matrixFreeElectroQuadratureComponentAx;

    unsigned int d_numberQuadraturePointsRhs;

    unsigned int d_numKPoints;
    unsigned int d_numSpins;
    unsigned int d_localVectorSizePsi, d_localVectorSizeElectro ;
    unsigned int d_numberWaveFunctions;

     std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperMemPtr;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                           d_BLASWrapperHostPtr;

    const dealii::AffineConstraints<double> *d_constraintMatrixPtrPsi;
    const dealii::AffineConstraints<double> *d_constraintMatrixSingleElectroHangingPeriodic;
    const dealii::AffineConstraints<double> *d_constraintMatrixSingleElectroHangingPeriodicHomogenous;


    dftfe::dftUtils::constraintMatrixInfo<memorySpace> d_ConstraintMatrixElectroHangingPeriodicHomogenousInfo;
    dftfe::dftUtils::constraintMatrixInfo<memorySpace> d_ConstraintMatrixElectroHangingPeriodicInfo;

    const dftParameters *            d_dftParamsPtr;

      dftfe::utils::MemoryStorage<unsigned int // This is hardcoded  can be an issue
	      , memorySpace>
      d_mapQuadIdToProcId;

      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> d_distVecMemSpace;

      dftfe::utils::MemoryStorage<ValueType, memorySpace> d_VFockQuadInputValues;

    unsigned int d_numCells; 
    unsigned int d_nQuadsPerCell; 
    dftfe::utils::MemoryStorage<ValueType,memorySpace> d_total_charge_memSpace;

    dftfe::utils::MemoryStorage<unsigned int, // This is hardcoded  can be an issue
                                dftfe::utils::MemorySpace::HOST> quadIds;

    dftfe::utils::MemoryStorage<ValueType, memorySpace> d_alphaValuePoisson;

    bool d_exchangeOperatorExists;
  };
}

#endif // DFTFE_EXACTEXCHANGECLASS_H
