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
// @author  Phani Motamarri, Sambit Das
//
#include <complex>
#include <vector>

namespace internal
{
  void pointWiseScaleWithDiagonal(const distributedCPUVec<double> & diagonal,
      std::vector<distributedCPUVec<double>> & fieldArray,
      dftUtils::constraintMatrixInfo & constraintsNoneEigenDataInfo)
  {
    for(unsigned int i = 0; i < fieldArray.size();++i)
    {
      auto & vec = fieldArray[i];
      vec.scale(diagonal);
      constraintsNoneEigenDataInfo.distribute(vec);
      vec.update_ghost_values();
    }

  }


  void pointWiseScaleWithDiagonal(const distributedCPUVec<double> & diagonal,
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & singleComponentPartitioner,
      const unsigned int numberFields,
      std::vector<dataTypes::number> & fieldsArrayFlattened)
  {

    const unsigned int numberDofs = fieldsArrayFlattened.size()/numberFields;
    const unsigned int inc = 1;

    for(unsigned int i = 0; i < numberDofs; ++i)
    {
#ifdef USE_COMPLEX
      double scalingCoeff =
        diagonal.local_element(i);
      zdscal_(&numberFields,
          &scalingCoeff,
          &fieldsArrayFlattened[i*numberFields],
          &inc);
#else
      double scalingCoeff = diagonal.local_element(i);
      dscal_(&numberFields,
          &scalingCoeff,
          &fieldsArrayFlattened[i*numberFields],
          &inc);
#endif
    }

  }
}

//
  template<unsigned int FEOrder>
dataTypes::number dftClass<FEOrder>::computeTraceXtHX(unsigned int numberWaveFunctionsEstimate)
{
  //
  //set up poisson solver
  //
  dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
  poissonSolverProblem<FEOrder> phiTotalSolverProblem(mpi_communicator);

  //
  //solve for vself and compute Tr(XtHX)
  //
  d_vselfBinsManager.solveVselfInBins(matrix_free_data,
      2,
      constraintsNone,
      d_imagePositions,
      d_imageIds,
      d_imageCharges,
      d_localVselfs);

  //
  //solve for potential corresponding to initial electron-density
  //
  phiTotalSolverProblem.reinit(matrix_free_data,
      d_phiTotRhoIn,
      *d_constraintsVector[phiTotDofHandlerIndex],
      phiTotDofHandlerIndex,
      d_atomNodeIdToChargeMap,
      *rhoInValues);


  dealiiCGSolver.solve(phiTotalSolverProblem,
      dftParameters::absLinearSolverTolerance,
      dftParameters::maxLinearSolverIterations,
      dftParameters::verbosity);

  //
  //create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder> kohnShamDFTEigenOperator(this,mpi_communicator);
  kohnShamDFTEigenOperator.init();

  //
  //precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
  //
  kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals();

  //
  //compute Veff
  //
  if(dftParameters::xc_id < 4)
  {
    kohnShamDFTEigenOperator.computeVEff(rhoInValues, d_phiTotRhoIn, d_phiExt, d_pseudoVLoc);
  }
  else if (dftParameters::xc_id == 4)
  {
    kohnShamDFTEigenOperator.computeVEff(rhoInValues, gradRhoInValues, d_phiTotRhoIn, d_phiExt, d_pseudoVLoc);
  }

  //
  //compute Hamiltonian matrix
  //
  kohnShamDFTEigenOperator.computeHamiltonianMatrix(0);

  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[0]);


  //
  //compute projected Hamiltonian
  //
  std::vector<dataTypes::number> ProjHam;

  kohnShamDFTEigenOperator.XtHX(d_eigenVectorsFlattenedSTL[0],
      d_numEigenValues,
      ProjHam);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[0]);


  dataTypes::number trXtHX = 0.0;
  for(unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
  {
    trXtHX += ProjHam[d_numEigenValues*i+i];
  }

  return trXtHX;

}

  template<unsigned int FEOrder>
double dftClass<FEOrder>::computeTraceXtKX(unsigned int numberWaveFunctionsEstimate)
{

  //
  //create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder> kohnShamDFTEigenOperator(this,mpi_communicator);
  kohnShamDFTEigenOperator.init();

  //
  //precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
  //
  kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals();


  //
  //compute Hamiltonian matrix
  //
  kohnShamDFTEigenOperator.computeKineticMatrix();

  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[0]);


  //
  //orthogonalize the vectors
  //
  linearAlgebraOperations::gramSchmidtOrthogonalization(d_eigenVectorsFlattenedSTL[0],
      d_numEigenValues,
      mpi_communicator);

  //
  //compute projected Hamiltonian
  //
  std::vector<dataTypes::number> ProjHam;

  kohnShamDFTEigenOperator.XtHX(d_eigenVectorsFlattenedSTL[0],
      d_numEigenValues,
      ProjHam);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[0]);

  double trXtKX = 0.0;
#ifdef USE_COMPLEX
  trXtKX = 0.0;
#else
  for(unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
  {
    trXtKX += ProjHam[d_numEigenValues*i+i];
  }
#endif

  return trXtKX;


}


  template<unsigned int FEOrder>
void dftClass<FEOrder>::solveNoSCF()
{

  //
  //create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder> kohnShamDFTEigenOperator(this,mpi_communicator);
  kohnShamDFTEigenOperator.init();

  kohnShamDFTEigenOperator.processGridOptionalELPASetup(d_numEigenValues,
      d_numEigenValuesRR);

  for(unsigned int spinType=0; spinType<(1+dftParameters::spinPolarized); ++spinType)
  {
    //
    //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
    //multiply by M^{1/2}
    for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size(); ++kPointIndex)
    {
      internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
          matrix_free_data.get_vector_partitioner(),
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);
    }


    if (dftParameters::verbosity>=2)
      pcout<<"Re-orthonormalizing before solving for ground-state after Gaussian Movement of Mesh "<< std::endl;
    //
    //orthogonalize the vectors
    //
    for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size(); ++kPointIndex)
    {
      const unsigned int flag=linearAlgebraOperations::pseudoGramSchmidtOrthogonalization
        (kohnShamDFTEigenOperator,
         d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
         d_numEigenValues,
         interBandGroupComm,
         mpi_communicator,
         false);
    }


    //
    //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
    //
    for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size(); ++kPointIndex)
    {
      internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
          matrix_free_data.get_vector_partitioner(),
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);
    }
  }

  DensityCalculator<FEOrder> densityCalc;

  densityCalc.computeRhoFromPSI(
      d_eigenVectorsFlattenedSTL,
      d_eigenVectorsRotFracDensityFlattenedSTL,
      d_numEigenValues,
      d_numEigenValuesRR,
      eigenValues,
      fermiEnergy,
      fermiEnergyUp,
      fermiEnergyDown,
      dofHandler,
      constraintsNone,
      matrix_free_data,
      eigenDofHandlerIndex,
      0,
      localProc_dof_indicesReal,
      localProc_dof_indicesImag,  
      d_kPointWeights,
      rhoOutValues,
      gradRhoOutValues,
      rhoOutValuesSpinPolarized,
      gradRhoOutValuesSpinPolarized,
      dftParameters::xc_id == 4,
      interpoolcomm,
      interBandGroupComm,
      false,
      false);
}

//chebyshev solver
  template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceCompute(const unsigned int spinType,
    const unsigned int kPointIndex,
    kohnShamDFTOperatorClass<FEOrder> & kohnShamDFTEigenOperator,
    elpaScalaManager & elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver,
    std::vector<double>                            & residualNormWaveFunctions,
    const bool isSpectrumSplit,
    const bool useMixedPrec,
    const bool isFirstScf,
    const bool useFullMassMatrixGEP)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity>=2)
  {
    pcout << "kPoint: "<< kPointIndex<<std::endl;
    if (dftParameters::spinPolarized==1)
      pcout << "spin: "<< spinType+1 <<std::endl;
  }


  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  std::vector<double> eigenValuesTemp(isSpectrumSplit?d_numEigenValuesRR
      :d_numEigenValues,0.0);

  subspaceIterationSolver.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  subspaceIterationSolver.solve(kohnShamDFTEigenOperator,
      d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      d_eigenVectorsRotFracDensityFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      d_tempEigenVec,
      d_numEigenValues,
      eigenValuesTemp,
      residualNormWaveFunctions,
      interBandGroupComm,
      useMixedPrec,
      isFirstScf,
      useFullMassMatrixGEP);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  if (!useFullMassMatrixGEP)
    internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
        matrix_free_data.get_vector_partitioner(),
        d_numEigenValues,
        d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
  {
    internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
        matrix_free_data.get_vector_partitioner(),
        d_numEigenValuesRR,
        d_eigenVectorsRotFracDensityFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);
  }

  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
  {
    for(unsigned int i = 0; i < d_numEigenValuesRR; i++)
    {
      if(dftParameters::verbosity>=4 && d_numEigenValues==d_numEigenValuesRR)
        pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;
      else if(dftParameters::verbosity>=4 && d_numEigenValues!=d_numEigenValuesRR)
        pcout<<"valence eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValuesRRSplit[kPointIndex][spinType*d_numEigenValuesRR + i] =  eigenValuesTemp[i];
    }

    for(unsigned int i = 0; i < d_numEigenValues; i++)
    {
      if (i>=(d_numEigenValues-d_numEigenValuesRR))
        eigenValues[kPointIndex][spinType*d_numEigenValues + i]
          = eigenValuesTemp[i-(d_numEigenValues-d_numEigenValuesRR)];
      else
        eigenValues[kPointIndex][spinType*d_numEigenValues + i]=-100.0;
    }
  }
  else
  {
    for(unsigned int i = 0; i < d_numEigenValues; i++)
    {
      if(dftParameters::verbosity>=4)
        pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValues[kPointIndex][spinType*d_numEigenValues + i] =  eigenValuesTemp[i];
    }
  }

  if (dftParameters::verbosity>=4)
    pcout <<std::endl;


  //set a0 and bLow
  /* a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=isSpectrumSplit?
     dftParameters::lowerEndWantedSpectrum
     :eigenValuesTemp[0];*/


  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();

  if(!isSpectrumSplit)
  {
    a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType] = eigenValuesTemp[0];
  }

  computing_timer.exit_section("Chebyshev solve");
}

#ifdef DFTFE_WITH_GPU
//chebyshev solver
  template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceCompute(const unsigned int spinType,
    const unsigned int kPointIndex,
    kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
    elpaScalaManager & elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolverCUDA & subspaceIterationSolverCUDA,
    std::vector<double>                            & residualNormWaveFunctions,
    const bool isXlBOMDLinearizedSolve,
    const unsigned int numberRayleighRitzAvoidanceXLBOMDPasses,
    const bool isSpectrumSplit,
    const bool useMixedPrec,
    const bool isFirstScf,
    const bool useFullMassMatrixGEP)
{
  computing_timer.enter_section("Chebyshev solve CUDA");

  if (dftParameters::verbosity>=2)
  {
    pcout << "kPoint: "<< kPointIndex<<std::endl;
    if (dftParameters::spinPolarized==1)
      pcout << "spin: "<< spinType+1 <<std::endl;
  }

  std::vector<double> eigenValuesTemp(isSpectrumSplit?d_numEigenValuesRR
      :d_numEigenValues,0.0);
  std::vector<double> eigenValuesDummy(isSpectrumSplit?d_numEigenValuesRR
      :d_numEigenValues,0.0);

  subspaceIterationSolverCUDA.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  const unsigned int rowsBlockSize=elpaScala.getScalapackBlockSize();
  std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;

  linearAlgebraOperations::internal::createProcessGridSquareMatrix(elpaScala.getMPICommunicator(),
      d_numEigenValues,
      processGrid,
      false);

  dealii::ScaLAPACKMatrix<double> projHamPar(d_numEigenValues,
      processGrid,
      rowsBlockSize);


  dealii::ScaLAPACKMatrix<double> overlapMatPar(d_numEigenValues,
      processGrid,
      rowsBlockSize);


  if (numberRayleighRitzAvoidanceXLBOMDPasses>0)
  {
    bool isFirstPass=false;
    if(useMixedPrec && dftParameters::useAsyncChebPGS_SR && dftParameters::useMixedPrecPGS_SR)
    {
      subspaceIterationSolverCUDA.solveNoRRMixedPrec(kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesDummy,
          interBandGroupComm,
          isXlBOMDLinearizedSolve,
          numberRayleighRitzAvoidanceXLBOMDPasses+1,
          useMixedPrec);
    }
    else
    {
      subspaceIterationSolverCUDA.solveNoRR(kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesDummy,
          interBandGroupComm,
          projHamPar,
          overlapMatPar,
          processGrid,
          isXlBOMDLinearizedSolve,
          numberRayleighRitzAvoidanceXLBOMDPasses,
          useMixedPrec);

    }
  }
  else	  
  {
#ifdef DFTFE_WITH_ELPA
    if (dftParameters::useELPA)
    {
      subspaceIterationSolverCUDA.solve(kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsRotFracFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesDummy,
          residualNormWaveFunctions,
          interBandGroupComm,
          projHamPar,
          overlapMatPar,
          processGrid,
          isXlBOMDLinearizedSolve,
          useMixedPrec,
          isFirstScf,
          useFullMassMatrixGEP,
          true,
          false);
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();

      if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
      {
        if (dftParameters::rrGEP==false)
          linearAlgebraOperations::elpaPartialDiagonalization(elpaScala,
              d_numEigenValues,
              d_numEigenValues-d_numEigenValuesRR,
              elpaScala.getMPICommunicator(),
              eigenValuesTemp,
              projHamPar,
              processGrid);
        else
          linearAlgebraOperations::elpaPartialDiagonalizationGEP(elpaScala,
              d_numEigenValues,
              d_numEigenValues-d_numEigenValuesRR,
              elpaScala.getMPICommunicator(),
              eigenValuesTemp,
              projHamPar,
              overlapMatPar,
              processGrid); 
      }
      else
      {
        if (dftParameters::rrGEP==false)
          linearAlgebraOperations::elpaDiagonalization(elpaScala,
              d_numEigenValues,
              elpaScala.getMPICommunicator(),
              eigenValuesTemp,
              projHamPar,
              processGrid);
        else
          linearAlgebraOperations::elpaDiagonalizationGEP(elpaScala,
              d_numEigenValues,
              elpaScala.getMPICommunicator(),
              eigenValuesTemp,
              projHamPar,
              overlapMatPar,
              processGrid);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==0 && dftParameters::verbosity>=2)
        if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
          std::cout<<"Time for ELPA partial eigen decomp, RR step: "<<time<<std::endl;
        else
          std::cout<<"Time for ELPA eigen decomp, RR step: "<<time<<std::endl;


      subspaceIterationSolverCUDA.solve(kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsRotFracFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesTemp,
          residualNormWaveFunctions,
          interBandGroupComm,
          projHamPar,
          overlapMatPar,
          processGrid,
          isXlBOMDLinearizedSolve,
          useMixedPrec,
          isFirstScf,
          useFullMassMatrixGEP,
          false,
          true);
    }
    else
    {
      subspaceIterationSolverCUDA.solve(kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsRotFracFlattenedCUDA.begin()
          +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesTemp,
          residualNormWaveFunctions,
          interBandGroupComm,
          projHamPar,
          overlapMatPar,
          processGrid,
          isXlBOMDLinearizedSolve,
          useMixedPrec,
          isFirstScf,
          useFullMassMatrixGEP);
    }
#else
    subspaceIterationSolverCUDA.solve(kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesTemp,
        residualNormWaveFunctions,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        isXlBOMDLinearizedSolve,
        useMixedPrec,
        isFirstScf,
        useFullMassMatrixGEP);
#endif


    //
    //copy the eigenValues and corresponding residual norms back to data members
    //
    if (isSpectrumSplit)
    {
      for(unsigned int i = 0; i < d_numEigenValuesRR; i++)
      {

        if(dftParameters::verbosity>=5 && d_numEigenValues==d_numEigenValuesRR)
          pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;
        else if(dftParameters::verbosity>=5 && d_numEigenValues!=d_numEigenValuesRR)
          pcout<<"valence eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

        eigenValuesRRSplit[kPointIndex][spinType*d_numEigenValuesRR + i] =  eigenValuesTemp[i];
      }

      for(unsigned int i = 0; i < d_numEigenValues; i++)
      {
        if (i>=(d_numEigenValues-d_numEigenValuesRR))
          eigenValues[kPointIndex][spinType*d_numEigenValues + i]
            = eigenValuesTemp[i-(d_numEigenValues-d_numEigenValuesRR)];
        else
          eigenValues[kPointIndex][spinType*d_numEigenValues + i]=-100.0;
      }
    }
    else
    {
      for(unsigned int i = 0; i < d_numEigenValues; i++)
      {

        if(dftParameters::verbosity>=5)
          pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

        eigenValues[kPointIndex][spinType*d_numEigenValues + i] =  eigenValuesTemp[i];
      }
    }

    if (dftParameters::verbosity>=4)
      pcout <<std::endl;


    bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();

    if(!isSpectrumSplit)
    {
      a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType] = eigenValuesTemp[0];
    }
  }
  computing_timer.exit_section("Chebyshev solve CUDA");
}
#endif


#ifdef DFTFE_WITH_GPU
//chebyshev solver
  template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceOnlyRRCompute(const unsigned int spinType,
    const unsigned int kPointIndex,
    kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
    elpaScalaManager & elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolverCUDA & subspaceIterationSolverCUDA,
    const bool isSpectrumSplit,
    const bool useMixedPrec)
{
  if (dftParameters::verbosity>=2)
  {
    pcout << "kPoint: "<< kPointIndex<<std::endl;
    if (dftParameters::spinPolarized==1)
      pcout << "spin: "<< spinType+1 <<std::endl;
  }

  std::vector<double> eigenValuesTemp(isSpectrumSplit?d_numEigenValuesRR
      :d_numEigenValues,0.0);
  std::vector<double> eigenValuesDummy(isSpectrumSplit?d_numEigenValuesRR
      :d_numEigenValues,0.0);


  const unsigned int rowsBlockSize=elpaScala.getScalapackBlockSize();
  std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;

  linearAlgebraOperations::internal::createProcessGridSquareMatrix(elpaScala.getMPICommunicator(),
      d_numEigenValues,
      processGrid,
      false);

  dealii::ScaLAPACKMatrix<double> projHamPar(d_numEigenValues,
      processGrid,
      rowsBlockSize);


  dealii::ScaLAPACKMatrix<double> overlapMatPar(d_numEigenValues,
      processGrid,
      rowsBlockSize);


#ifdef DFTFE_WITH_ELPA
  if (dftParameters::useELPA)
  {
    subspaceIterationSolverCUDA.onlyRR(kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesDummy,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec,
        true,
        false);
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
    {
      linearAlgebraOperations::elpaPartialDiagonalization(elpaScala,
          d_numEigenValues,
          d_numEigenValues-d_numEigenValuesRR,
          elpaScala.getMPICommunicator(),
          eigenValuesTemp,
          projHamPar,
          processGrid);
    }
    else
    {
      linearAlgebraOperations::elpaDiagonalization(elpaScala,
          d_numEigenValues,
          elpaScala.getMPICommunicator(),
          eigenValuesTemp,
          projHamPar,
          processGrid);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==0 && dftParameters::verbosity>=2)
      if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
        std::cout<<"Time for ELPA partial eigen decomp, RR step: "<<time<<std::endl;
      else
        std::cout<<"Time for ELPA eigen decomp, RR step: "<<time<<std::endl;


    subspaceIterationSolverCUDA.onlyRR(kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesTemp,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec,
        false,
        true);
  }
  else
  {
    subspaceIterationSolverCUDA.onlyRR(kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin()
        +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesTemp,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec);
  }
#else
  subspaceIterationSolverCUDA.onlyRR(kohnShamDFTEigenOperator,
      d_eigenVectorsFlattenedCUDA.begin()
      +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsFlattenedSTL[0].size(),
      d_eigenVectorsRotFracFlattenedCUDA.begin()
      +((1+dftParameters::spinPolarized)*kPointIndex+spinType)*d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
      d_eigenVectorsFlattenedSTL[0].size(),
      d_tempEigenVec,
      d_numEigenValues,
      eigenValuesTemp,
      interBandGroupComm,
      projHamPar,
      overlapMatPar,
      processGrid,
      useMixedPrec);
#endif


  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
  {
    for(unsigned int i = 0; i < d_numEigenValuesRR; i++)
    {

      if(dftParameters::verbosity>=5 && d_numEigenValues==d_numEigenValuesRR)
        pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;
      else if(dftParameters::verbosity>=5 && d_numEigenValues!=d_numEigenValuesRR)
        pcout<<"valence eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValuesRRSplit[kPointIndex][spinType*d_numEigenValuesRR + i] =  eigenValuesTemp[i];
    }

    for(unsigned int i = 0; i < d_numEigenValues; i++)
    {
      if (i>=(d_numEigenValues-d_numEigenValuesRR))
        eigenValues[kPointIndex][spinType*d_numEigenValues + i]
          = eigenValuesTemp[i-(d_numEigenValues-d_numEigenValuesRR)];
      else
        eigenValues[kPointIndex][spinType*d_numEigenValues + i]=-100.0;
    }
  }
  else
  {
    for(unsigned int i = 0; i < d_numEigenValues; i++)
    {

      if(dftParameters::verbosity>=5)
        pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValues[kPointIndex][spinType*d_numEigenValues + i] =  eigenValuesTemp[i];
    }
  }

  if (dftParameters::verbosity>=4)
    pcout <<std::endl;


  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();

  if(!isSpectrumSplit)
  {
    a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType] = eigenValuesTemp[0];
  }
}
#endif



//chebyshev solver
  template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceComputeNSCF(const unsigned int spinType,
    const unsigned int kPointIndex,
    kohnShamDFTOperatorClass<FEOrder> & kohnShamDFTEigenOperator,
    chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver,
    std::vector<double>                            & residualNormWaveFunctions,
    unsigned int ipass)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity==2)
  {
    pcout << "kPoint: "<< kPointIndex<<std::endl;
    pcout << "spin: "<< spinType+1 <<std::endl;
  }

  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  if (ipass==1)
    internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
        matrix_free_data.get_vector_partitioner(),
        d_numEigenValues,
        d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);


  std::vector<double> eigenValuesTemp(d_numEigenValues,0.0);

  subspaceIterationSolver.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);


  subspaceIterationSolver.solve(kohnShamDFTEigenOperator,
      d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
      d_tempEigenVec,
      d_numEigenValues,
      eigenValuesTemp,
      residualNormWaveFunctions,
      interBandGroupComm,
      false);

  if(dftParameters::verbosity >= 4)
  {
#ifdef USE_PETSC
    PetscLogDouble bytes;
    PetscMemoryGetCurrentUsage(&bytes);
    FILE *dummy;
    unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    PetscSynchronizedPrintf(mpi_communicator,"[%d] Memory after recreating STL vector and exiting from subspaceIteration solver  %e\n",this_mpi_process,bytes);
    PetscSynchronizedFlush(mpi_communicator,dummy);
#endif
  }



  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  for(unsigned int i = 0; i < d_numEigenValues; i++)
  {
    //if(dftParameters::verbosity==2)
    //    pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

    eigenValues[kPointIndex][spinType*d_numEigenValues + i] =  eigenValuesTemp[i];
  }

  //if (dftParameters::verbosity==2)
  //   pcout <<std::endl;


  //set a0 and bLow
  a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp[0];
  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();
  //


  computing_timer.exit_section("Chebyshev solve");
}


//compute the maximum of the residual norm of the highest occupied state among all k points
  template<unsigned int FEOrder>
double dftClass<FEOrder>::computeMaximumHighestOccupiedStateResidualNorm(const std::vector<std::vector<double> > & residualNormWaveFunctionsAllkPoints,
    const std::vector<std::vector<double> > & eigenValuesAllkPoints,
    const double fermiEnergy)
{
  double maxHighestOccupiedStateResNorm=-1e+6;
  for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
  {
    unsigned int highestOccupiedState = 0;

    for(unsigned int i = 0; i < eigenValuesAllkPoints[kPoint].size(); i++)
    {
      const double factor=(eigenValuesAllkPoints[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
      if (factor<0)
        highestOccupiedState=i;
    }

    if(residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState]>maxHighestOccupiedStateResNorm)
    {
      maxHighestOccupiedStateResNorm=residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState];
    }

  }
  maxHighestOccupiedStateResNorm= Utilities::MPI::max(maxHighestOccupiedStateResNorm, interpoolcomm);
  return maxHighestOccupiedStateResNorm;
}
