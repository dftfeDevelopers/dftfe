// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju, Phani Motamarri
//

#ifndef eigen_H_
#define eigen_H_
#include <headers.h>
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <operator.h>

namespace dftfe{

  template <unsigned int T> class dftClass;

  //
  //Define eigenClass class
  //
  template <unsigned int FEOrder>
    class eigenClass : public operatorDFTClass
    {
      template <unsigned int T>
	friend class dftClass;

      template <unsigned int T>
	friend class symmetryClass;

    public:
      eigenClass(dftClass<FEOrder>* _dftPtr, const MPI_Comm &mpi_comm_replica);

      /**
       * @brief Compute operator times vector or operator times bunch of vectors
       *
       * @param src Vector of Vectors containing current values of source array (non-const as
         we scale src and rescale src to avoid creation of temporary vectors)
       * @param dst Vector of Vectors containing operator times vectors product
       */
      void HX(std::vector<vectorType> &src,
	      std::vector<vectorType> &dst);



      /**
       * @brief Compute discretized operator matrix times multi-vectors and add it to the existing dst vector
       * works for both real and complex data types
       * @param src Vector containing current values of source array with multi-vector array stored
       * in a flattened format with all the wavefunction value corresponding to a given node is stored
       * contiguously (non-const as we scale src and rescale src to avoid creation of temporary vectors)
       * @param numberComponents Number of multi-fields(vectors)
       * @param macroCellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param cellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of local active cells
       * @param scaleFlag which decides whether dst has to be scaled square root of diagonal mass matrix before evaluating
       * matrix times src vector
       * @param scalar which multiplies src before evaluating matrix times src vector
       * @param dst Vector containing sum of dst vector and operator times given multi-vectors product
       */
      void HX(dealii::parallel::distributed::Vector<dataTypes::number> & src,
	      const unsigned int numberComponents,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
	      const bool scaleFlag,
	      const dataTypes::number scalar,
	      dealii::parallel::distributed::Vector<dataTypes::number> & dst);


      /**
       * @brief Compute projection of the operator into orthogonal basis
       *
       * @param src given orthogonal basis vectors
       * @return ProjMatrix projected small matrix
       */
      void XtHX(dealii::parallel::distributed::Vector<dataTypes::number> & src,
		const unsigned int numberComponents,
		const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
		const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
		std::vector<dataTypes::number> & ProjHam);



      /**
       * @brief Compute projection of the operator into orthogonal basis
       *
       * @param src given orthogonal basis vectors
       * @return ProjMatrix projected small matrix
       */
      void XtHX(std::vector<vectorType> &src,
		std::vector<dataTypes::number> & ProjHam);



       /**
       * @brief Computes effective potential involving local-density exchange-correlation functionals
       *
       * @param rhoValues electron-density
       * @param phi electrostatic potential arising both from electron-density and nuclear charge
       * @param phiExt electrostatic potential arising from nuclear charges
       * @param pseudoValues quadrature data of pseudopotential values
       */
      void computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
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
				    const vectorType & phi,
				    const vectorType & phiExt,
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
		       const vectorType & phi,
		       const vectorType & phiExt,
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
				    const vectorType & phi,
				    const vectorType & phiExt,
				    const unsigned int spinIndex,
				    const std::map<dealii::CellId,std::vector<double> > & pseudoValues);


      /**
       * @brief sets the data member to appropriate kPoint Index
       *
       * @param kPointIndex  k-point Index to set
       */
      void reinitkPointIndex(unsigned int & kPointIndex);

      //
      //initialize eigen class
      //
      void init ();

      /**
       * @brief initializes parallel layouts and index maps required for HX, XtHX.
       *
       * @param wavefunBlockSize number of wavefunction vectors to which the parallel layouts and
       * index maps correspond to. The same number of wavefunction vectors must be used
       * in subsequent calls to HX, XtHX.
       */
      void reinit(const unsigned int wavefunBlockSize);

      /**
       * @brief Computes diagonal mass matrix
       *
       * @param dofHandler dofHandler associated with the current mesh
       * @param constraintMatrix constraints to be used
       * @param sqrtMassVec output the value of square root of diagonal mass matrix
       * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
       */
      void computeMassVector(const dealii::DoFHandler<3> & dofHandler,
	                     const dealii::ConstraintMatrix & constraintMatrix,
			     vectorType & sqrtMassVec,
			     vectorType & invSqrtMassVec);

      ///precompute shapefunction gradient integral
      void preComputeShapeFunctionGradientIntegrals();

      ///compute element Hamiltonian matrix
      void computeHamiltonianMatrix(unsigned int kPointIndex);





    private:
      /**
       * @brief implementation of matrix-free based matrix-vector product at cell-level
       * @param data matrix-free data
       * @param dst Vector of Vectors containing matrix times vectors product
       * @param src Vector of Vectors containing input vectors
       * @param cell_range range of cell-blocks
       */
      void computeLocalHamiltonianTimesXMF(const dealii::MatrixFree<3,double>  &data,
					   std::vector<vectorType>  &dst,
					   const std::vector<vectorType>  &src,
					   const std::pair<unsigned int,unsigned int> &cell_range) const;

      /**
       * @brief implementation of  matrix-vector product for nonlocal Hamiltonian
       * @param src Vector of Vectors containing input vectors
       * @param dst Vector of Vectors containing matrix times vectors product
       */
      void computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
					    std::vector<vectorType>       &dst) const;




      /**
       * @brief finite-element cell level stiffness matrix with first dimension traversing the cell id(in the order of macro-cell and subcell)
       * and second dimension storing the stiffness matrix of size numberNodesPerElement x numberNodesPerElement in a flattened 1D array
       * of complex data type
       */
      std::vector<std::vector<dataTypes::number> > d_cellHamiltonianMatrix;

      /**
       * @brief implementation of matrix-vector product using cell-level stiffness matrices.
       * works for both real and complex data type
       * @param src Vector containing current values of source array with multi-vector array stored
       * in a flattened format with all the wavefunction value corresponding to a given node is stored
       * contiguously.
       * @param numberWaveFunctions Number of wavefunctions at a given node.
       * @param flattenedArrayCellLocalProcIndexIdMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param dst Vector containing matrix times given multi-vectors product
       */
      void computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<dataTypes::number> & src,
					 const unsigned int numberWaveFunctions,
					 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					 dealii::parallel::distributed::Vector<dataTypes::number> & dst) const;






      /**
       * @brief implementation of non-local Hamiltonian matrix-vector product using non-local discretized projectors at cell-level
       * works for both complex and real data type
       * @param src Vector containing current values of source array with multi-vector array stored
       * in a flattened format with all the wavefunction value corresponding to a given node is stored
       * contiguously.
       * @param numberWaveFunctions Number of wavefunctions at a given node.
       * @param flattenedArrayCellLocalProcIndexIdMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param dst Vector containing matrix times given multi-vectors product
       */
      void computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<dataTypes::number> & src,
					    const unsigned int numberWaveFunctions,
					    const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					    dealii::parallel::distributed::Vector<dataTypes::number> & dst) const;



      ///pointer to dft class
      dftClass<FEOrder>* dftPtr;


      ///data structures to store diagonal of inverse square root mass matrix and square root of mass matrix
      vectorType d_invSqrtMassVector,d_sqrtMassVector;

      dealii::Table<2, dealii::VectorizedArray<double> > vEff;
      dealii::Table<3, dealii::VectorizedArray<double> > derExcWithSigmaTimesGradRho;


       /**
       * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(del N_i \dot \del N_j))
       * with first dimension traversing the macro cell id
       * and second dimension storing the matrix of size numberNodesPerElement x numberNodesPerElement in a flattened 1D dealii Vectorized array
       */
      std::vector<std::vector<dealii::VectorizedArray<double> > > d_cellShapeFunctionGradientIntegral;

      ///storage for shapefunctions
      std::vector<double> d_shapeFunctionValue;
      dealii::Table<4, dealii::VectorizedArray<double> > d_cellShapeFunctionGradientValue;



      ///storage for  matrix-free cell data
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
      mutable dealii::Threads::Mutex  assembler_lock;

      //d_kpoint index for which Hamiltonian is computed
      unsigned int d_kPointIndex;
    };


}
#endif
