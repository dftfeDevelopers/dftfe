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


#ifndef kohnShamDFTOperatorCUDAClass_H_
#define kohnShamDFTOperatorCUDAClass_H_
#ifndef USE_COMPLEX
#include <headers.h>
#include <constants.h>
#include <operatorCUDA.h>

namespace dftfe{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	template <unsigned int T> class dftClass;
#endif

	/**
	 * @brief Implementation class for building the Kohn-Sham DFT discrete operator and the action of the discrete operator on a single vector or multiple vectors
	 *
	 * @author Phani Motamarri, Sambit Das
	 */

	//
	//Define kohnShamDFTOperatorCUDAClass class
	//
	template <unsigned int FEOrder>
		class kohnShamDFTOperatorCUDAClass : public operatorDFTCUDAClass
	{
		//template <unsigned int T>
		// friend class dftClass;

		template <unsigned int T>
			friend class symmetryClass;

		public:
		kohnShamDFTOperatorCUDAClass(dftClass<FEOrder>* _dftPtr, const MPI_Comm &mpi_comm_replica);


		void createCublasHandle();

		void destroyCublasHandle();

		cublasHandle_t & getCublasHandle();

		const double * getSqrtMassVec();

		const double * getInvSqrtMassVec();

		thrust::device_vector<unsigned int> & getBoundaryIdToLocalIdMap();

		//distributedGPUVec<double> & getBlockCUDADealiiVector();

		//distributedGPUVec<double> & getBlockCUDADealiiVector2();

		//distributedGPUVec<double> & getBlockCUDADealiiVector3();


		//thrust::device_vector<dataTypes::number> & getBlockCUDADealiiVector();


		//thrust::device_vector<dataTypes::number> & getBlockCUDADealiiVector2(); 

		distributedCPUVec<dataTypes::number> &  getProjectorKetTimesVectorSingle();

		thrust::device_vector<double> & getShapeFunctionGradientIntegral();

		thrust::device_vector<double> & getShapeFunctionValues();

		thrust::device_vector<double> & getShapeFunctionValuesInverted(const bool use2pPlusOneGLQuad=false);

		thrust::device_vector<double> & getShapeFunctionValuesNLPInverted();

		thrust::device_vector<double> & getShapeFunctionGradientValuesX();

		thrust::device_vector<double> & getShapeFunctionGradientValuesY();

		thrust::device_vector<double> & getShapeFunctionGradientValuesZ();

		thrust::device_vector<double> & getShapeFunctionGradientValuesXInverted(const bool use2pPlusOneGLQuad=false);

		thrust::device_vector<double> & getShapeFunctionGradientValuesYInverted(const bool use2pPlusOneGLQuad=false);

		thrust::device_vector<double> & getShapeFunctionGradientValuesZInverted(const bool use2pPlusOneGLQuad=false);

		thrust::device_vector<dealii::types::global_dof_index> & getFlattenedArrayCellLocalProcIndexIdMap();

		thrust::device_vector<dataTypes::number> & getCellWaveFunctionMatrix();

		thrust::device_vector<unsigned int> & getLocallyOwnedProcBoundaryNodesVectorDevice();

		thrust::device_vector<unsigned int> & getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice();

		/**
		 * @brief Compute operator times vector or operator times bunch of vectors
		 * @param X Vector of Vectors containing current values of X
		 * @param Y Vector of Vectors containing operator times vectors product
		 */
		void HX(std::vector<distributedCPUVec<double>> & X,
				std::vector<distributedCPUVec<double>> & Y);


		/**
		 * @brief Compute discretized operator matrix times multi-vectors and add it to the existing dst vector
		 * works for both real and complex data types
		 * @param src Vector containing current values of source array with multi-vector array stored
		 * in a flattened format with all the wavefunction value corresponding to a given node is stored
		 * contiguously (non-const as we scale src and rescale src to avoid creation of temporary vectors)
		 * @param numberComponents Number of multi-fields(vectors)

		 * @param scaleFlag which decides whether dst has to be scaled square root of diagonal mass matrix before evaluating
		 * matrix times src vector
		 * @param scalar which multiplies src before evaluating matrix times src vector
		 * @param dst Vector containing sum of dst vector and operator times given multi-vectors product
		 */
		void HX(distributedGPUVec<double> & src,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				const bool scaleFlag,
				const double scalar,
				distributedGPUVec<double> & dst,
				const bool doUnscalingX=true);

		void HX(distributedGPUVec<double> & src,
				distributedGPUVec<float> & srcFloat,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				const bool scaleFlag,
				const double scalar,
				distributedGPUVec<double> & dst,
				const bool doUnscalingX=true,
				const bool singlePrecCommun=false);

		void HXCheby(distributedGPUVec<double> & X,
				distributedGPUVec<float> & XFloat,  
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				distributedGPUVec<double> & Y,
				bool mixedPrecflag=false,
				bool computePart1=false,
				bool computePart2=false);


		void HXChebyNoCommun(distributedGPUVec<double> & X,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				distributedGPUVec<double> & Y);

		/**
		 * @brief Compute projection of the operator into orthogonal basis
		 *
		 * @param X given orthogonal basis vectors
		 * @return ProjMatrix projected small matrix
		 */
		void XtHX(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				double* projHam,
				const bool isProjHamOnDevice=true);

#ifdef DEAL_II_WITH_SCALAPACK
		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHX(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);

		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis.
		 * This routine also overlaps communication and computation.
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHXOverlapComputeCommun(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);


		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis.
		 * This routine uses a mixed precision algorithm (https://doi.org/10.1016/j.cpc.2019.07.016).
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param floatXb parallel distributed vector datastructure for handling block of wavefunction
		 * vectors in single precision
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param Noc number of fully occupied wavefunction vectors considered in the mixed precision algorithm
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHXMixedPrec(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const unsigned int Noc,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);


		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis.
		 * This routine uses a mixed precision algorithm (https://doi.org/10.1016/j.cpc.2019.07.016).
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param floatXb parallel distributed vector datastructure for handling block of wavefunction
		 * vectors in single precision
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHXOffDiagBlockSinglePrec(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);


		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis.
		 * This routine uses a mixed precision algorithm (https://doi.org/10.1016/j.cpc.2019.07.016)
		 * and further overlaps communication and computation.
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param floatXb parallel distributed vector datastructure for handling block of wavefunction
		 * vectors in single precision
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param Noc number of fully occupied wavefunction vectors considered in the mixed precision algorithm
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHXMixedPrecOverlapComputeCommun(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const unsigned int Noc,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);


		/**
		 * @brief Compute projection of the operator into a subspace spanned by a given basis.
		 * This routine uses a mixed precision algorithm (https://doi.org/10.1016/j.cpc.2019.07.016)
		 * and further overlaps communication and computation.
		 *
		 * @param X Vector of Vectors containing all wavefunction vectors
		 * @param Xb parallel distributed vector datastructure for handling block of wavefunction vectors
		 * @param floatXb parallel distributed vector datastructure for handling block of wavefunction
		 * vectors in single precision
		 * @param HXb parallel distributed vector datastructure for handling H multiplied by block of 
		 * wavefunction vectors
		 * @param projectorKetTimesVector parallel distributed vector datastructure for handling nonlocal 
		 * projector kets times block wavefunction vectors
		 * @param M number of local dofs
		 * @param N total number of wavefunction vectors
		 * @param handle cublasHandle
		 * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
		 * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
		 * of the operation into the given subspace
		 */
		void XtHXOffDiagBlockSinglePrecOverlapComputeCommun(const double *  X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & projHamPar);


#endif



		/**
		 * @brief Computes effective potential involving local-density exchange-correlation functionals
		 *
		 * @param rhoValues electron-density
		 * @param phi electrostatic potential arising both from electron-density and nuclear charge
		 * @param phiExt electrostatic potential arising from nuclear charges
		 * @param pseudoValues quadrature data of pseudopotential values
		 */
		void computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				const distributedCPUVec<double> & phi,
				const distributedCPUVec<double> & phiExt,
				const std::map<dealii::CellId,std::vector<double> > & pseudoValues);


		/**
		 * @brief Computes effective potential involving local spin density exchange-correlation functionals
		 *
		 * @param rhoValues electron-density
		 * @param phi electrostatic potential arising both from electron-density and nuclear charge
		 * @param phiExt electrostatic potential arising from nuclear charges
		 * @param spinIndex flag to toggle spin-up or spin-down
		 * @param pseudoValues quadrature data of pseudopotential values
		 */
		void computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				const distributedCPUVec<double> & phi,
				const distributedCPUVec<double> & phiExt,
				unsigned int spinIndex,
				const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

		/**
		 * @brief Computes effective potential involving gradient density type exchange-correlation functionals
		 *
		 * @param rhoValues electron-density
		 * @param gradRhoValues gradient of electron-density
		 * @param phi electrostatic potential arising both from electron-density and nuclear charge
		 * @param phiExt electrostatic potential arising from nuclear charges
		 * @param pseudoValues quadrature data of pseudopotential values
		 */
		void computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				const distributedCPUVec<double> & phi,
				const distributedCPUVec<double> & phiExt,
				const std::map<dealii::CellId,std::vector<double> > & pseudoValues);


		/**
		 * @brief Computes effective potential for gradient-spin density type exchange-correlation functionals
		 *
		 * @param rhoValues electron-density
		 * @param gradRhoValues gradient of electron-density
		 * @param phi electrostatic potential arising both from electron-density and nuclear charge
		 * @param phiExt electrostatic potential arising from nuclear charges
		 * @param spinIndex flag to toggle spin-up or spin-down
		 * @param pseudoValues quadrature data of pseudopotential values
		 */
		void computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				const distributedCPUVec<double> & phi,
				const distributedCPUVec<double> & phiExt,
				const unsigned int spinIndex,
				const std::map<dealii::CellId,std::vector<double> > & pseudoValues);


		/**
		 * @brief sets the data member to appropriate kPoint Index
		 *
		 * @param kPointIndex  k-point Index to set
		 */
		void reinitkPointSpinIndex(const unsigned int kPointIndex, const unsigned int spinIndex);

		//
		//initialize eigen class
		//
		void init ();

		/**
		 * @brief initializes parallel layouts and index maps required for HX, XtHX and creates a flattened array
		 * format for X
		 *
		 * @param wavefunBlockSize number of wavefunction vectors to which the parallel layouts and
		 * index maps correspond to. The same number of wavefunction vectors must be used
		 * in subsequent calls to HX, XtHX.
		 * @param flag controls the creation of flattened array format and index maps or only index maps
		 *
		 *
		 * @return X format to store a multi-vector array
		 * in a flattened format with all the wavefunction values corresponding to a given node being stored
		 * contiguously
		 *
		 */

		void reinit(const unsigned int wavefunBlockSize);

		void reinit(const unsigned int wavefunBlockSize,
				bool flag);

		void reinitNoRemesh(const unsigned int wavefunBlockSize);

		/**
		 * @brief Computes diagonal mass matrix
		 *
		 * @param dofHandler dofHandler associated with the current mesh
		 * @param constraintMatrix constraints to be used
		 * @param sqrtMassVec output the value of square root of diagonal mass matrix
		 * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
		 */
		void computeMassVector(const dealii::DoFHandler<3> & dofHandler,
				const dealii::AffineConstraints<double> & constraintMatrix,
				distributedCPUVec<double> & sqrtMassVec,
				distributedCPUVec<double> & invSqrtMassVec);

		///precompute shapefunction gradient integral
		void preComputeShapeFunctionGradientIntegrals();


		void computeHamiltonianMatrix(const unsigned int kPointIndex, const unsigned int spinIndex);


		/**
		 * @brief implementation of non-local projector kets times psi product
		 * using non-local discretized projectors at cell-level.
		 * works for both complex and real data type
		 * @param src Vector containing current values of source array with multi-vector array stored
		 * in a flattened format with all the wavefunction value corresponding to a given node is stored
		 * contiguously.
		 * @param numberWaveFunctions Number of wavefunctions at a given node.
		 */
		void computeNonLocalProjectorKetTimesXTimesV(const double *src,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int numberWaveFunctions);

		private:
		/**
		 * @brief implementation of matrix-free based matrix-vector product at cell-level
		 * @param data matrix-free data
		 * @param dst Vector of Vectors containing matrix times vectors product
		 * @param src Vector of Vectors containing input vectors
		 * @param cell_range range of cell-blocks
		 */
		void computeLocalHamiltonianTimesXMF(const dealii::MatrixFree<3,double>  &data,
				std::vector<distributedCPUVec<double>>  &dst,
				const std::vector<distributedCPUVec<double>>  &src,
				const std::pair<unsigned int,unsigned int> &cell_range) const;

		/**
		 * @brief implementation of  matrix-vector product for nonlocal Hamiltonian
		 * @param src Vector of Vectors containing input vectors
		 * @param dst Vector of Vectors containing matrix times vectors product
		 */
		void computeNonLocalHamiltonianTimesX(const std::vector<distributedCPUVec<double>> &src,
				std::vector<distributedCPUVec<double>>       &dst) const;




		/**
		 * @brief finite-element cell level stiffness matrix with first dimension traversing the cell id(in the order of macro-cell and subcell)
		 * and second dimension storing the stiffness matrix of size numberNodesPerElement x numberNodesPerElement in a flattened 1D array
		 * of complex data type
		 */
		std::vector<dataTypes::number>  d_cellHamiltonianMatrixFlattened;
		thrust::device_vector<dataTypes::number> d_cellHamiltonianMatrixFlattenedDevice;
		//thrust::device_vector<dataTypes::number> d_cellWaveFunctionMatrix;
		thrust::device_vector<dataTypes::number> d_cellHamMatrixTimesWaveMatrix;

		/// for non local

		std::vector<dataTypes::number> d_cellHamiltonianMatrixNonLocalFlattened;
		thrust::device_vector<dataTypes::number> d_cellHamiltonianMatrixNonLocalFlattenedDevice;
		std::vector<dataTypes::number> d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
		thrust::device_vector<dataTypes::number> d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice;
		thrust::device_vector<dataTypes::number> d_cellWaveFunctionMatrixNonLocalDevice;
		thrust::device_vector<dataTypes::number> d_cellHamMatrixTimesWaveMatrixNonLocalDevice;
		//distributedGPUVec<double> d_projectorKetTimesVectorDealiiParFlattenedDevice;
		thrust::device_vector<dataTypes::number> d_projectorKetTimesVectorParFlattenedDevice;
		thrust::device_vector<dataTypes::number> d_projectorKetTimesVectorAllCellsDevice;
		thrust::device_vector<dataTypes::number> d_projectorKetTimesVectorDevice;
		std::vector<double> d_nonLocalPseudoPotentialConstants;
		thrust::device_vector<double> d_nonLocalPseudoPotentialConstantsDevice;

		std::vector<double> d_projectorKetTimesVectorAllCellsReduction;
		thrust::device_vector<double>  d_projectorKetTimesVectorAllCellsReductionDevice;
		std::vector<unsigned int> d_pseudoWfcAccumNonlocalAtoms;
		unsigned int d_totalNonlocalAtomsCurrentProc;
		unsigned int d_totalNonlocalElems;
		unsigned int d_totalPseudoWfcNonLocal;
		unsigned int d_maxSingleAtomPseudoWfc;
		std::vector<unsigned int> d_pseduoWfcNonLocalAtoms;
		std::vector<unsigned int> d_numberCellsNonLocalAtoms;
		std::vector<unsigned int> d_numberCellsAccumNonLocalAtoms;
		std::vector<dealii::types::global_dof_index> d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal;
		thrust::device_vector<dealii::types::global_dof_index> d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice;
		std::vector<unsigned int> d_projectorIdsParallelNumberingMap;
		thrust::device_vector<unsigned int> d_projectorIdsParallelNumberingMapDevice;
		std::vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
		thrust::device_vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice;
		std::vector<unsigned int>  d_cellNodeIdMapNonLocalToLocal;
		thrust::device_vector<unsigned int>  d_cellNodeIdMapNonLocalToLocalDevice;
		std::vector<unsigned int> d_normalCellIdToMacroCellIdMap;
		std::vector<unsigned int> d_macroCellIdToNormalCellIdMap;

		thrust::device_vector<unsigned int> d_locallyOwnedProcBoundaryNodesVectorDevice;

		thrust::device_vector<unsigned int> d_locallyOwnedProcProjectorKetBoundaryNodesVectorDevice;


		/**
		 * @brief implementation of matrix-vector product using cell-level stiffness matrices.
		 * works for both real and complex data type
		 * @param src Vector containing current values of source array with multi-vector array stored
		 * in a flattened format with all the wavefunction value corresponding to a given node is stored
		 * contiguously.
		 * @param numberWaveFunctions Number of wavefunctions at a given node.
		 * @param dst Vector containing matrix times given multi-vectors product
		 */
		void computeLocalHamiltonianTimesX(const distributedCPUVec<dataTypes::number> & src,
				const unsigned int numberWaveFunctions,
				distributedCPUVec<dataTypes::number> & dst) const;


		void computeLocalHamiltonianTimesX(const double* src,
				const unsigned int numberWaveFunctions,
				double* dst,
				const bool skipAccumulationBoundaryNodes=false);

		/**
		 * @brief implementation of non-local Hamiltonian matrix-vector product
		 * using non-local discretized projectors at cell-level.
		 * works for both complex and real data type
		 * @param src Vector containing current values of source array with multi-vector array stored
		 * in a flattened format with all the wavefunction value corresponding to a given node is stored
		 * contiguously.
		 * @param numberWaveFunctions Number of wavefunctions at a given node.
		 * @param dst Vector containing matrix times given multi-vectors product
		 */
		void computeNonLocalHamiltonianTimesX(const distributedCPUVec<dataTypes::number> & src,
				const unsigned int numberWaveFunctions,
				distributedCPUVec<dataTypes::number> & dst) const;


		/**
		 * @brief implementation of non-local Hamiltonian matrix-vector product
		 * using non-local discretized projectors at cell-level.
		 * works for both complex and real data type
		 * @param src Vector containing current values of source array with multi-vector array stored
		 * in a flattened format with all the wavefunction value corresponding to a given node is stored
		 * contiguously.
		 * @param numberWaveFunctions Number of wavefunctions at a given node.
		 * @param dst Vector containing matrix times given multi-vectors product
		 */
		void computeNonLocalHamiltonianTimesX(const double *src,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int numberWaveFunctions,
				double* dst,
				const bool skip1=false,
				const bool skip2=false,
				const bool skipAccumulationBoundaryNodes=false);




		///pointer to dft class
		dftClass<FEOrder>* dftPtr;


		///data structures to store diagonal of inverse square root mass matrix and square root of mass matrix
		distributedCPUVec<double> d_invSqrtMassVector,d_sqrtMassVector;
		thrust::device_vector<double> d_invSqrtMassVectorDevice, d_sqrtMassVectorDevice;

		dealii::Table<2, dealii::VectorizedArray<double> > vEff;

		std::vector<double> d_vEff;
		std::vector<double> d_vEffJxW;
		thrust::device_vector<double> d_vEffJxWDevice;

		const unsigned int d_numQuadPoints;
		const unsigned int d_numLocallyOwnedCells;

		dealii::Table<2, dealii::Tensor<1,3,dealii::VectorizedArray<double> > > derExcWithSigmaTimesGradRho;

		std::vector<double> d_derExcWithSigmaTimesGradRho;
		std::vector<double> d_derExcWithSigmaTimesGradRhoJxW;
		thrust::device_vector<double> d_derExcWithSigmaTimesGradRhoJxWDevice;


		/**
		 * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(\nabla N_i \cdot \nabla N_j))
		 * with first dimension traversing the macro cell id
		 * and second dimension storing the matrix of size numberNodesPerElement x numberNodesPerElement in a flattened 1D dealii Vectorized array
		 */
		std::vector<std::vector<dealii::VectorizedArray<double> > > d_cellShapeFunctionGradientIntegral;

		std::vector<double> d_cellShapeFunctionGradientIntegralFlattened;
		//thrust::device_vector<double> d_cellShapeFunctionGradientIntegralFlattenedDevice;

		/// storage for shapefunctions
		std::vector<double> d_shapeFunctionValue;
		std::vector<double> d_shapeFunctionValueInverted;
		//thrust::device_vector<double> d_shapeFunctionValueDevice;
		//thrust::device_vector<double> d_shapeFunctionValueInvertedDevice;

		/// storage for shapefunction gradients
		std::vector<double> d_shapeFunctionGradientValueX;
		std::vector<double> d_shapeFunctionGradientValueXInverted;
		//thrust::device_vector<double> d_shapeFunctionGradientValueXDevice;
		//thrust::device_vector<double> d_shapeFunctionGradientValueXInvertedDevice;

		std::vector<double> d_shapeFunctionGradientValueY;
		std::vector<double> d_shapeFunctionGradientValueYInverted;
		//thrust::device_vector<double> d_shapeFunctionGradientValueYDevice;
		//thrust::device_vector<double> d_shapeFunctionGradientValueYInvertedDevice;

		std::vector<double> d_shapeFunctionGradientValueZ;
		std::vector<double> d_shapeFunctionGradientValueZInverted;
		//thrust::device_vector<double> d_shapeFunctionGradientValueZDevice;
		//thrust::device_vector<double> d_shapeFunctionGradientValueZInvertedDevice;


		std::vector<double> d_cellJxWValues;
		thrust::device_vector<double> d_cellJxWValuesDevice;

		//storage for  matrix-free cell data
		const unsigned int d_numberNodesPerElement;
		const unsigned int d_numberMacroCells;
		std::vector<unsigned int> d_macroCellSubCellMap;

		//parallel objects
		const MPI_Comm mpi_communicator;
		const unsigned int n_mpi_processes;
		const unsigned int this_mpi_process;
		dealii::ConditionalOStream   pcout;

		//compute-time logger
		dealii::TimerOutput computing_timer;

		//mutex thread for managing multi-thread writing to XHXvalue
		//mutable dealii::Threads::Mutex  assembler_lock;

		//d_kpoint index for which Hamiltonian is computed
		unsigned int d_kPointIndex;

		unsigned int d_spinIndex;

		//storage for precomputing index maps
		std::vector<dealii::types::global_dof_index > d_flattenedArrayCellLocalProcIndexIdMap;

		//storage for precomputing index maps
		std::vector<dealii::types::global_dof_index> d_flattenedArrayMacroCellLocalProcIndexIdMapFlattened;
		thrust::device_vector<dealii::types::global_dof_index> d_DeviceFlattenedArrayMacroCellLocalProcIndexIdMapFlattened;
		thrust::device_vector<unsigned int> d_boundaryIdsVecDevice;      

		///storage for magma and cublas handles
		cublasHandle_t  d_cublasHandle;

		///storage for CUDA device dealii array
		// distributedCPUVec<dataTypes::number,dealii::MemorySpace::CUDA> d_cudaFlattenedArrayBlock;
		//distributedGPUVec<double> d_cudaFlattenedArrayBlock;
		//distributedGPUVec<double> d_cudaFlattenedArrayBlock2;
		//distributedGPUVec<double> d_cudaFlattenedArrayBlock3;
		//distributedGPUVec<double> d_cudaFlattenedArrayBlock4;
	};
}
#endif
#endif
