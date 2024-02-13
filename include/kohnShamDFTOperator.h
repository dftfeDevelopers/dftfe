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


#ifndef kohnShamDFTOperatorClass_H_
#define kohnShamDFTOperatorClass_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <headers.h>
#include <operator.h>
#include <BLASWrapper.h>

namespace dftfe
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  template <unsigned int T1, unsigned int T2>
  class dftClass;
#endif

  /**
   * @brief Implementation class for building the Kohn-Sham DFT discrete operator and the action of the discrete operator on a single vector or multiple vectors
   *
   * @author Phani Motamarri, Sambit Das
   */

  //
  // Define kohnShamDFTOperatorClass class
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class kohnShamDFTOperatorClass : public operatorDFTClass
  {
    // template <unsigned int T1, unsigned int T2>
    friend class dftClass<FEOrder, FEOrderElectro>;

    // template <unsigned int T>
    //	friend class symmetryClass;

  public:
    kohnShamDFTOperatorClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
                             const MPI_Comm &                   mpi_comm_parent,
                             const MPI_Comm &mpi_comm_domain);

    /**
     * @brief Compute discretized operator matrix times multi-vectors and add it to the existing dst vector
     * works for both real and complex data types
     * @param src Vector containing current values of source array with multi-vector array stored
     * in a flattened format with all the wavefunction value corresponding to a
     given node is stored
     * contiguously (non-const as we scale src and rescale src to avoid creation
     of temporary vectors)
     * @param numberComponents Number of multi-fields(vectors)

     * @param scaleFlag which decides whether dst has to be scaled square root of diagonal mass matrix before evaluating
     * matrix times src vector
     * @param scalar which multiplies src before evaluating matrix times src vector
     * @param dst Vector containing sum of dst vector and operator times given multi-vectors product
     */
    void
    HX(distributedCPUMultiVec<dataTypes::number> &src,
       const unsigned int                         numberComponents,
       const bool                                 scaleFlag,
       const double                               scalar,
       distributedCPUMultiVec<dataTypes::number> &dst,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
* @brief Compute discretized operator matrix times multi-vectors and add it to the existing dst vector
* works for real and complex data types (Optimized matrix times multi-vectors
using different datastructures for
           * interior nodes and exterior nodes to reduce memory access costs
during global to cell level vectors and vice versa)
* @param src Vector containing current values of source array with multi-vector array stored
* in a flattened format with all the wavefunction value corresponding to a given
node is stored
* contiguously (non-const as we scale src and rescale src to avoid creation of
temporary vectors)
           * @param cellSrcWaveFunctionMatrix containing current values of source array with multi-vector array stored
* in a flattened format with all the wavefunction value corresponding to a given
node is stored
* contiguously for a given cell
* @param numberComponents Number of multi-fields(vectors)

* @param scaleFlag which decides whether dst has to be scaled square root of diagonal mass matrix before evaluating
* matrix times src vector
* @param scalar which multiplies src before evaluating matrix times src vector
           * @param scalarA which is used for Chebyshev recursive iteration
           * @param scalarB which is used for Chebyshev recursive iteration
* @param dst Vector containing sum of dst vector and operator times given multi-vectors product
           * @param cellDstWaveFunctionMatrix containing sum of cell level dst vector and operator times given multi-vectors product
           */
    void
    HX(distributedCPUMultiVec<dataTypes::number> &src,
       std::vector<dataTypes::number> &           cellSrcWaveFunctionMatrix,
       const unsigned int                         numberComponents,
       const bool                                 scaleFlag,
       const double                               scalar,
       const double                               scalarA,
       const double                               scalarB,
       distributedCPUMultiVec<dataTypes::number> &dst,
       std::vector<dataTypes::number> &           cellDstWaveFunctionMatrix);


    /**
     * @brief Compute projection of the operator into orthogonal basis
     *
     * @param src given orthogonal basis vectors
     * @return ProjMatrix projected small matrix
     */
    void
    XtHX(const dataTypes::number *       src,
         const unsigned int              numberComponents,
         const unsigned int              numberLocalDofs,
         std::vector<dataTypes::number> &ProjHam);

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     *
     * The XtHX and filling into projHamPar is done in a blocked approach
     * which avoids creation of full projected Hamiltonian matrix memory, and
     * also avoids creation of another full X memory.
     */
    void
    XtHX(const dataTypes::number *                        X,
         const unsigned int                               numberComponents,
         const unsigned int                               numberLocalDofs,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param N total number of wavefunctions components
     * @param Ncore number of wavecfuntions starting from the first for
     * which the project Hamiltionian block will be computed in single
     * procession. However the cross blocks will still be computed in double
     * precision.
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHXMixedPrec(
      const dataTypes::number *                        X,
      const unsigned int                               N,
      const unsigned int                               Ncore,
      const unsigned int                               numberLocalDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
     * @brief Computes effective potential involving local-density exchange-correlation functionals
     *
     * @param rhoValues electron-density
     * @param phi electrostatic potential arising both from electron-density and nuclear charge
     * @param phiExt electrostatic potential arising from nuclear charges
     * @param externalPotCorrValues quadrature data of sum{Vext} minus sum{Vnu}
     */
    void
    computeVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiValues,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const unsigned int externalPotCorrQuadratureId);


    /**
     * @brief Computes effective potential involving local spin density exchange-correlation functionals
     *
     * @param rhoValues electron-density
     * @param phi electrostatic potential arising both from electron-density and nuclear charge
     * @param spinIndex flag to toggle spin-up or spin-down
     * @param externalPotCorrValues quadrature data of sum{Vext} minus sum{Vnu}
     */
    void
    computeVEffSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &          phiValues,
      unsigned int spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const unsigned int externalPotCorrQuadratureId);

    /**
     * @brief Computes effective potential involving gradient density type exchange-correlation functionals
     *
     * @param rhoValues electron-density
     * @param gradRhoValues gradient of electron-density
     * @param phi electrostatic potential arising both from electron-density and nuclear charge
     * @param externalPotCorrValues quadrature data of sum{Vext} minus sum{Vnu}
     */
    void
    computeVEff(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiValues,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const unsigned int externalPotCorrQuadratureId);


    /**
     * @brief Computes effective potential for gradient-spin density type exchange-correlation functionals
     *
     * @param rhoValues electron-density
     * @param gradRhoValues gradient of electron-density
     * @param phi electrostatic potential arising both from electron-density and nuclear charge
     * @param spinIndex flag to toggle spin-up or spin-down
     * @param externalPotCorrValues quadrature data of sum{Vext} minus sum{Vnu}
     */
    void
    computeVEffSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                phiValues,
      const unsigned int spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const unsigned int externalPotCorrQuadratureId);

    /**
     * @brief Computes directional derivative of effective potential for local density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrime(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues);

    /**
     * @brief Computes directional derivative of effective potential for local spin-density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues);


    /**
     * @brief Computes directional derivative of effective potential for gradient density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrime(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues);

    /**
     * @brief Computes directional derivative of effective potential for gradient spin-density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                                                  phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues);

    /**
     * @brief sets the data member to appropriate kPoint and spin Index
     *
     * @param kPointIndex  k-point Index to set
     */
    void
    reinitkPointSpinIndex(const unsigned int kPointIndex,
                          const unsigned int spinIndex);


    void
    resetExtPotHamFlag();

    //
    // initialize eigen class
    //
    void
    init();

    /**
     * @brief initializes parallel layouts and index maps required for HX, XtHX and creates a flattened array
     * format for X
     *
     * @param wavefunBlockSize number of wavefunction vectors to which the parallel layouts and
     * index maps correspond to. The same number of wavefunction vectors must be
     * used in subsequent calls to HX, XtHX.
     * @param flag controls the creation of flattened array format and index maps or only index maps
     *
     *
     * @return X format to store a multi-vector array
     * in a flattened format with all the wavefunction values corresponding to a
     * given node being stored contiguously
     *
     */
    void
    reinit(const unsigned int                    wavefunBlockSize,
           distributedCPUVec<dataTypes::number> &X,
           bool                                  flag);

    void
    reinit(const unsigned int                         wavefunBlockSize,
           distributedCPUMultiVec<dataTypes::number> &X,
           bool                                       flag);

    void
    reinit(const unsigned int wavefunBlockSize);


    void
    initCellWaveFunctionMatrix(
      const unsigned int                         numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &X,
      std::vector<dataTypes::number> &           cellWaveFunctionMatrix);


    void
    fillGlobalArrayFromCellWaveFunctionMatrix(
      const unsigned int                         wavefunBlockSize,
      const std::vector<dataTypes::number> &     cellWaveFunctionMatrix,
      distributedCPUMultiVec<dataTypes::number> &X);

    void
    initWithScalar(const unsigned int              numberWaveFunctions,
                   double                          scalarValue,
                   std::vector<dataTypes::number> &cellWaveFunctionMatrix);


    void
    axpby(double                                scalarA,
          double                                scalarB,
          const unsigned int                    numberWaveFunctions,
          const std::vector<dataTypes::number> &cellXWaveFunctionMatrix,
          std::vector<dataTypes::number> &      cellYWaveFunctionMatrix);



    void
    getInteriorSurfaceNodesMapFromGlobalArray(
      std::vector<unsigned int> &globalArrayClassificationMap);

    const std::vector<dealii::types::global_dof_index> &
    getFlattenedArrayCellLocalProcIndexIdMap() const;

    distributedCPUMultiVec<dataTypes::number> &
    getParallelProjectorKetTimesBlockVector();

    const std::vector<double> &
    getShapeFunctionGradValuesDensityGaussQuad() const;

    const std::vector<double> &
    getShapeFunctionValuesDensityGaussLobattoQuad() const;

    const std::vector<double> &
    getShapeFunctionValuesDensityTransposed() const;

    const std::vector<double> &
    getShapeFunctionValuesNLPTransposed() const;

    const std::vector<double> &
    getShapeFunctionGradientValuesNLPTransposed() const;

    const std::vector<double> &
    getInverseJacobiansNLP() const;

    /**
     * @brief Computes diagonal mass matrix
     *
     * @param dofHandler dofHandler associated with the current mesh
     * @param constraintMatrix constraints to be used
     * @param sqrtMassVec output the value of square root of diagonal mass matrix
     * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
     */
    void
    computeMassVector(const dealii::DoFHandler<3> &            dofHandler,
                      const dealii::AffineConstraints<double> &constraintMatrix,
                      distributedCPUVec<double> &              sqrtMassVec,
                      distributedCPUVec<double> &              invSqrtMassVec);

    /// precompute shapefunction gradient integral
    void
    preComputeShapeFunctionGradientIntegrals(
      const unsigned int lpspQuadratureId);

    /// compute element Hamiltonian matrix
    void
    computeHamiltonianMatrix(
      const unsigned int kPointIndex,
      const unsigned int spinIndex,
      const bool         onlyHPrimePartForFirstOrderDensityMatResponse = false);
    void
    computeKineticMatrix();


  private:
    /**
     * @brief Computes effective potential for external potential correction to phiTot
     *
     * @param externalPotCorrValues quadrature data of sum{Vext} minus sum{Vnu}
     */
    void
    computeVEffExternalPotCorr(
      const std::map<dealii::CellId, std::vector<double>>
        &                externalPotCorrValues,
      const unsigned int externalPotCorrQuadratureId);

    /**
     * @brief finite-element cell level stiffness matrix with first dimension traversing the cell id(in the order of macro-cell and subcell)
     * and second dimension storing the stiffness matrix of size
     * numberNodesPerElement x numberNodesPerElement in a flattened 1D array of
     * complex data type
     */
    std::vector<std::vector<std::vector<dataTypes::number>>>
      d_cellHamiltonianMatrix;

    std::vector<double> d_elementHamiltonianMatrixImag;


    std::vector<std::vector<dataTypes::number>> d_cellMassMatrix;
    std::vector<double> d_cellHamiltonianMatrixExternalPotCorr;

    /**
     * @brief implementation of matrix-vector product using cell-level stiffness matrices.
     * works for both real and complex data type
     * @param src Vector containing current values of source array with multi-vector array stored
     * in a flattened format with all the wavefunction value corresponding to a
     * given node is stored contiguously.
     * @param numberWaveFunctions Number of wavefunctions at a given node.
     * @param dst Vector containing matrix times given multi-vectors product
     */
    void
    computeLocalHamiltonianTimesX(
      const distributedCPUMultiVec<dataTypes::number> &src,
      const unsigned int                               numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &      dst,
      const double                                     scalar = 1.0);

    void
    computeLocalHamiltonianTimesX(
      const distributedCPUMultiVec<dataTypes::number> &src,
      std::vector<dataTypes::number> &           cellSrcWaveFunctionMatrix,
      const unsigned int                         numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &dst,
      std::vector<dataTypes::number> &           cellDstWaveFunctionMatrix,
      const double                               scalar = 1.0);


    void
    computeHamiltonianTimesXInternal(
      const distributedCPUMultiVec<dataTypes::number> &src,
      std::vector<dataTypes::number> &           cellSrcWaveFunctionMatrix,
      const unsigned int                         numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &dst,
      std::vector<dataTypes::number> &           cellDstWaveFunctionMatrix,
      const double                               scalar    = 1.0,
      const double                               scalarA   = 1.0,
      const double                               scalarB   = 1.0,
      bool                                       scaleFlag = false);



    /**
     * @brief implementation of non-local Hamiltonian matrix-vector product
     * using non-local discretized projectors at cell-level.
     * works for both complex and real data type
     * @param src Vector containing current values of source array with multi-vector array stored
     * in a flattened format with all the wavefunction value corresponding to a
     * given node is stored contiguously.
     * @param numberWaveFunctions Number of wavefunctions at a given node.
     * @param dst Vector containing matrix times given multi-vectors product
     */
    void
    computeNonLocalHamiltonianTimesX(
      const distributedCPUMultiVec<dataTypes::number> &src,
      const unsigned int                               numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &      dst,
      const double                                     scalar = 1.0) const;

    void
    computeNonLocalProjectorKetTimesXTimesV(
      distributedCPUMultiVec<dataTypes::number> &src,
      distributedCPUMultiVec<dataTypes::number>
        &                projectorKetTimesVectorFlattened,
      const unsigned int numberWaveFunctions);


    /// pointer to dft class
    dftClass<FEOrder, FEOrderElectro> *dftPtr;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperPtrHost;

    /// data structures to store diagonal of inverse square root mass matrix and
    /// square root of mass matrix
    distributedCPUVec<double> d_invSqrtMassVector, d_sqrtMassVector;

    dealii::Table<2, dealii::VectorizedArray<double>> vEff;
    dealii::Table<2, dealii::VectorizedArray<double>> d_vEffExternalPotCorr;
    std::vector<double>                               d_vEffExternalPotCorrJxW;
    std::vector<double>                               d_vEffJxW;
    std::vector<double>              d_invJacderExcWithSigmaTimesGradRhoJxW;
    std::vector<std::vector<double>> d_invJacKPointTimesJxW;
    dealii::Table<2, dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      derExcWithSigmaTimesGradRho;


    /**
     * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(\nabla N_i \cdot \nabla N_j))
     * with first dimension traversing the macro cell id
     * and second dimension storing the matrix of size numberNodesPerElement x
     * numberNodesPerElement in a flattened 1D dealii Vectorized array
     */
    std::vector<double> d_cellShapeFunctionGradientIntegral;

    /// storage for shapefunctions
    std::vector<unsigned int> d_blockiNodeIndex;
    std::vector<unsigned int> d_blockjNodeIndex;
    std::vector<unsigned int> d_blockiNodeIndexNoSym;
    std::vector<unsigned int> d_blockjNodeIndexNoSym;
    std::vector<double>       d_shapeFunctionData;
    std::vector<double>       d_shapeFunctionLpspQuadData;
    std::vector<double>       d_shapeFunctionGradientValueRefX;
    std::vector<double>       d_shapeFunctionGradientValueRefY;
    std::vector<double>       d_shapeFunctionGradientValueRefZ;

    /// storage for shapefunctions
    std::vector<double> d_NiNjIntegral;


    /// storage for  matrix-free cell data
    const unsigned int        d_numberNodesPerElement;
    const unsigned int        d_numberCellsLocallyOwned;
    std::vector<unsigned int> d_nodesPerCellClassificationMap;
    std::vector<unsigned int> d_globalArrayClassificationMap;

    // parallel objects
    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;

    // compute-time logger
    dealii::TimerOutput computing_timer;

    // mutex thread for managing multi-thread writing to XHXvalue
    mutable dealii::Threads::Mutex assembler_lock;

    // d_kpoint index for which Hamiltonian is used in HX
    unsigned int d_kPointIndex;

    // spin index for which Hamiltonian is used in HX
    unsigned int d_spinIndex;

    // storage for precomputing index maps
    std::vector<std::vector<dealii::types::global_dof_index>>
      d_flattenedArrayMacroCellLocalProcIndexIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap;

    std::vector<dealii::types::global_dof_index>
      d_FullflattenedArrayMacroCellLocalProcIndexIdMap;

    std::vector<unsigned int> d_normalCellIdToMacroCellIdMap;
    std::vector<unsigned int> d_macroCellIdToNormalCellIdMap;

    /// flag for precomputing stiffness matrix contribution from
    /// sum{Vext}-sum{Vnuc}
    bool d_isStiffnessMatrixExternalPotCorrComputed;

    /// external potential correction quadrature id
    unsigned int d_externalPotCorrQuadratureId;
  };
} // namespace dftfe
#endif
