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
// ---------------------------------------------------------------------
//

#include "functionalTest.h"
#include "MultiVectorPoissonLinearSolverProblem.h"
#include "MultiVectorMinResSolver.h"
#include "MultiVectorCGSolver.h"
#include "poissonSolverProblem.h"
#include <dealiiLinearSolver.h>
#include "vectorUtilities.h"



namespace functionalTest
{
  // template <dftfe::utils::MemorySpace memorySpace>
  void
  testMultiVectorPoissonSolver(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                            basisOperationsPtr,
    dealii::MatrixFree<3, double> &matrixFreeData,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                            BLASWrapperPtr,
    std::vector<const dealii::AffineConstraints<double> *> &constraintMatrixVec,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                inputVec,
    const unsigned int matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const unsigned int verbosity,
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain)
  {
    dftfe::MultiVectorMinResSolver linearSolver(mpi_comm_parent,
                                                mpi_comm_domain);
    dftfe::MultiVectorPoissonLinearSolverProblem<
      dftfe::utils::MemorySpace::HOST>
      multiPoissonSolver(mpi_comm_parent, mpi_comm_domain);

    dealii::ConditionalOStream pcout(std::cout,
                                     (dealii::Utilities::MPI::this_mpi_process(
                                        mpi_comm_parent) == 0));
    unsigned int               blockSizeInput = 5;
    pcout << " setting block Size to " << blockSizeInput << "\n";

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoQuadInputValuesHost;

    basisOperationsPtr->reinit(blockSizeInput,
                               100,
                               matrixFreeQuadratureComponentRhsDensity,
                               true,  // TODO should this be set to true
                               true); // TODO should this be set to true
    unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    unsigned int numQuadsPerCell        = basisOperationsPtr->nQuadsPerCell();


    const dealii::DoFHandler<3> *d_dofHandler;
    d_dofHandler = &(matrixFreeData.get_dof_handler(matrixFreeVectorComponent));
    rhoQuadInputValuesHost.resize(totalLocallyOwnedCells * numQuadsPerCell *
                                  blockSizeInput);

    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      basisOpHost = std::make_shared<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>(
        BLASWrapperPtr);

    std::vector<dftfe::basis::UpdateFlags> updateFlags;
    updateFlags.resize(2);
    updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                     dftfe::basis::update_gradients |
                     dftfe::basis::update_quadpoints |
                     dftfe::basis::update_transpose;

    updateFlags[1] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                     dftfe::basis::update_gradients |
                     dftfe::basis::update_quadpoints |
                     dftfe::basis::update_transpose;

    std::vector<unsigned int> quadVec;
    quadVec.resize(2);
    quadVec[0] = matrixFreeQuadratureComponentRhsDensity;
    quadVec[1] = matrixFreeQuadratureComponentAX;

    basisOpHost->init(matrixFreeData,
                      constraintMatrixVec,
                      matrixFreeVectorComponent,
                      quadVec,
                      updateFlags);

    // set up solver functions for Poisson
    dftfe::poissonSolverProblem<2, 2> phiTotalSolverProblem(mpi_comm_domain);

    dftfe::distributedCPUVec<double>      expectedOutput;
    dftfe::distributedCPUMultiVec<double> multiExpectedOutput;
    // set up linear solver
    dftfe::dealiiLinearSolver dealiiCGSolver(mpi_comm_parent,
                                             mpi_comm_domain,
                                             dftfe::dealiiLinearSolver::CG);
    std::map<dealii::types::global_dof_index, double> atoms;
    std::map<dealii::CellId, std::vector<double>>     smearedChargeValues;
    dftfe::vectorTools::createDealiiVector<double>(
      matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
      1,
      expectedOutput);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
      blockSizeInput,
      multiExpectedOutput);

    dftfe::distributedCPUMultiVec<double> multiVectorOutput;
    dftfe::distributedCPUMultiVec<double> boundaryValues;
    multiVectorOutput.reinit(multiExpectedOutput);
    boundaryValues.reinit(multiExpectedOutput);
    //    boundaryValues = 0.0;
    boundaryValues.setValue(0.0);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      scaledInputVec;
    scaledInputVec.resize(inputVec.size());
    for (unsigned int iBlockId = 0; iBlockId < blockSizeInput; iBlockId++)
      {
        dftfe::distributedCPUVec<double> singleBoundaryCond;
        singleBoundaryCond.reinit(expectedOutput);
        singleBoundaryCond = 0.0;
        singleBoundaryCond.update_ghost_values();
        for (unsigned int iNodeId = 0;
             iNodeId < singleBoundaryCond.locally_owned_size();
             iNodeId++)
          {
            boundaryValues.data()[iNodeId * blockSizeInput + iBlockId] =
              singleBoundaryCond.local_element(iNodeId);
          }
        boundaryValues.updateGhostValues();
      }


    unsigned int                                iElem = 0;
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  d_dofHandler->begin_active(),
                                                endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int i = 0; i < numQuadsPerCell; i++)
            {
              for (unsigned int k = 0; k < blockSizeInput; k++)
                {
                  rhoQuadInputValuesHost[iElem * blockSizeInput *
                                           numQuadsPerCell +
                                         i * blockSizeInput + k] =
                    inputVec.data()[iElem * numQuadsPerCell + i] * (k + 1);
                }
            }
          iElem++;
        }

    for (unsigned int k = 0; k < blockSizeInput; k++)
      {
        expectedOutput = 0;
        double multFac = k + 1;
        BLASWrapperPtr->axpby(numQuadsPerCell * totalLocallyOwnedCells,
                              multFac,
                              inputVec.begin(),
                              0.0,
                              scaledInputVec.begin());

        phiTotalSolverProblem.reinit(
          basisOpHost,
          expectedOutput,
          *constraintMatrixVec[matrixFreeVectorComponent],
          matrixFreeVectorComponent,
          matrixFreeQuadratureComponentRhsDensity,
          matrixFreeQuadratureComponentAX,
          atoms,
          smearedChargeValues,
          matrixFreeQuadratureComponentAX,
          scaledInputVec,
          true,  // isComputeDiagonalA
          false, // isComputeMeanValueConstraint
          false, // smearedNuclearCharges
          true,  // isRhoValues
          false, // isGradSmearedChargeRhs
          0,     // smearedChargeGradientComponentId
          false, // storeSmearedChargeRhs
          false, // reuseSmearedChargeRhs
          true); // reinitializeFastConstraints

        dealiiCGSolver.solve(phiTotalSolverProblem, 1e-10, 10000, verbosity);

        dealii::types::global_dof_index indexVec;
        for (unsigned int i = 0; i < expectedOutput.locally_owned_size(); i++)
          {
            indexVec = i * blockSizeInput + k;
            multiExpectedOutput.data()[indexVec] =
              expectedOutput.local_element(i);
          }
      }



    multiPoissonSolver.reinit(BLASWrapperPtr,
                              basisOpHost,
                              *constraintMatrixVec[matrixFreeVectorComponent],
                              matrixFreeVectorComponent,
                              matrixFreeQuadratureComponentRhsDensity,
                              matrixFreeQuadratureComponentAX,
                              false);



    multiPoissonSolver.setDataForRhsVec(rhoQuadInputValuesHost);

    unsigned int locallyOwnedSize = basisOpHost->nOwnedDofs();

    linearSolver.solve(multiPoissonSolver,
                       BLASWrapperPtr,
                       multiVectorOutput,
                       boundaryValues,
                       locallyOwnedSize,
                       blockSizeInput,
                       1e-10,
                       10000,
                       verbosity,
                       true);


    double multiExpectedOutputNorm = 0.0;
    BLASWrapperPtr->xdot(locallyOwnedSize * blockSizeInput,
                         multiExpectedOutput.begin(),
                         1,
                         multiExpectedOutput.begin(),
                         1,
                         mpi_comm_domain,
                         &multiExpectedOutputNorm);

    double multiVectorOutputNorm = 0.0;
    BLASWrapperPtr->xdot(locallyOwnedSize * blockSizeInput,
                         multiVectorOutput.begin(),
                         1,
                         multiVectorOutput.begin(),
                         1,
                         mpi_comm_domain,
                         &multiVectorOutputNorm);

    BLASWrapperPtr->axpby(locallyOwnedSize * blockSizeInput,
                          -1.0,
                          multiExpectedOutput.begin(),
                          1.0,
                          multiVectorOutput.begin());

    multiVectorOutputNorm = 0.0;
    BLASWrapperPtr->xdot(locallyOwnedSize * blockSizeInput,
                         multiVectorOutput.begin(),
                         1,
                         multiVectorOutput.begin(),
                         1,
                         mpi_comm_domain,
                         &multiVectorOutputNorm);

    multiVectorOutputNorm = std::sqrt(multiVectorOutputNorm);
    if (multiVectorOutputNorm > 1e-9)
      {
        pcout << " Error in TestMultiVectorPoissonSolver = "
              << multiVectorOutputNorm << "\n";
      }
    else
      {
        pcout << " TestMultiVectorPoissonSolver successful \n";
      }
  }

} // end of namespace functionalTest
