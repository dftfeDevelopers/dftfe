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


#ifndef kohnShamDFTOperatorDeviceClass_H_
#define kohnShamDFTOperatorDeviceClass_H_
#include <constants.h>
#include <headers.h>
#include <operatorDevice.h>
#include <FEBasisOperations.h>
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
  // Define kohnShamDFTOperatorDeviceClass class
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class kohnShamDFTOperatorDeviceClass : public operatorDFTDeviceClass
  {
  public:
    kohnShamDFTOperatorDeviceClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
                                   const MPI_Comm &mpi_comm_parent,
                                   const MPI_Comm &mpi_comm_domain);

    /**
     * @brief destructor
     */
    ~kohnShamDFTOperatorDeviceClass();

    void
    createDeviceBlasHandle();

    void
    destroyDeviceBlasHandle();

    dftfe::utils::deviceBlasHandle_t &
    getDeviceBlasHandle();

    const double *
    getSqrtMassVec();

    const double *
    getInvSqrtMassVec();


    distributedCPUVec<dataTypes::number> &
    getProjectorKetTimesVectorSingle();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionGradientIntegral();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &
    getShapeFunctionGradientIntegralElectro();

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &
      getShapeFunctionValues();

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &
      getShapeFunctionValuesTransposed(const bool use2pPlusOneGLQuad = false);

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &
      getShapeFunctionValuesNLPTransposed();

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &
      getShapeFunctionGradientValuesNLPTransposed();

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &
      getInverseJacobiansNLP();

    dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                dftfe::utils::MemorySpace::DEVICE> &
    getFlattenedArrayCellLocalProcIndexIdMap();

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE> &
    getCellWaveFunctionMatrix();

    distributedCPUVec<dataTypes::number> &
    getParallelVecSingleComponent();

    distributedDeviceVec<dataTypes::number> &
    getParallelChebyBlockVectorDevice();

    distributedDeviceVec<dataTypes::number> &
    getParallelChebyBlockVector2Device();

    distributedDeviceVec<dataTypes::number> &
    getParallelProjectorKetTimesBlockVectorDevice();

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      &
      getLocallyOwnedProcBoundaryNodesVectorDevice();


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
    HX(distributedDeviceVec<dataTypes::number> &src,
       distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
       const unsigned int                       localVectorSize,
       const unsigned int                       numberComponents,
       const bool                               scaleFlag,
       const double                             scalar,
       distributedDeviceVec<dataTypes::number> &dst,
       const bool                               doUnscalingX    = true,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    HX(distributedDeviceVec<dataTypes::number> &    src,
       distributedDeviceVec<dataTypes::numberFP32> &srcFloat,
       distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector,
       const unsigned int                           localVectorSize,
       const unsigned int                           numberComponents,
       const bool                                   scaleFlag,
       const double                                 scalar,
       distributedDeviceVec<dataTypes::number> &    dst,
       const bool                                   doUnscalingX     = true,
       const bool                                   singlePrecCommun = false,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse      = false);

    void
    HXCheby(distributedDeviceVec<dataTypes::number> &    X,
            distributedDeviceVec<dataTypes::numberFP32> &XFloat,
            distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
            const unsigned int                       localVectorSize,
            const unsigned int                       numberComponents,
            distributedDeviceVec<dataTypes::number> &Y,
            bool                                     mixedPrecflag = false,
            bool                                     computePart1  = false,
            bool                                     computePart2  = false);


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
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHX(const dataTypes::number *                X,
         distributedDeviceVec<dataTypes::number> &Xb,
         distributedDeviceVec<dataTypes::number> &HXb,
         distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
         const unsigned int                       M,
         const unsigned int                       N,
         dftfe::utils::deviceBlasHandle_t &       handle,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         utils::DeviceCCLWrapper &devicecclMpiCommDomain,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

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
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHXOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given basis.
     * This routine uses a mixed precision algorithm
     * (https://doi.org/10.1016/j.cpc.2019.07.016) and further overlaps
     * communication and computation.
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
     * @param handle deviceBlasHandle
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHXMixedPrecOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::numberFP32> &    floatXb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    XtHXMixedPrecCommunOverlapComputeCommun(
      const dataTypes::number *                        X,
      distributedDeviceVec<dataTypes::number> &        Xb,
      distributedDeviceVec<dataTypes::number> &        HXb,
      distributedDeviceVec<dataTypes::number> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      dftfe::utils::deviceBlasHandle_t &               handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);
#endif



    /**
     * @brief Computes effective potential involving local-density exchange-correlation functionals
     *
     * @param rhoValues electron-density
     * @param phi electrostatic potential arising both from electron-density and nuclear charge
     * @param phiExt electrostatic potential arising from nuclear charges
     * @param pseudoValues quadrature data of pseudopotential values
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
     * @param phiExt electrostatic potential arising from nuclear charges
     * @param spinIndex flag to toggle spin-up or spin-down
     * @param pseudoValues quadrature data of pseudopotential values
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
     * @param phiExt electrostatic potential arising from nuclear charges
     * @param pseudoValues quadrature data of pseudopotential values
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
     * @param phiExt electrostatic potential arising from nuclear charges
     * @param spinIndex flag to toggle spin-up or spin-down
     * @param pseudoValues quadrature data of pseudopotential values
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
        &                                                  rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues);

    /**
     * @brief Computes directional derivative of effective potential for local spin-density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrimeSpinPolarized(
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues);


    /**
     * @brief Computes directional derivative of effective potential for gradient density type exchange-correlation functionals
     *
     */
    void
    computeVEffPrime(
      const std::map<dealii::CellId, std::vector<double>> &rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
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
        &                                                  rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &                                                  gradRhoValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoPrimeValues,
      const std::map<dealii::CellId, std::vector<double>> &phiPrimeValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues);

    /**
     * @brief sets the data member to appropriate kPoint Index
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
    reinit(const unsigned int wavefunBlockSize, bool flag);

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
      const unsigned int lpspQuadratureId,
      const bool         onlyUpdateGradNiNjIntegral = false);


    void
    computeHamiltonianMatricesAllkpt(
      const unsigned int spinIndex,
      const bool         onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
     * @brief implementation of non-local projector kets times psi product
     * using non-local discretized projectors at cell-level.
     * works for both complex and real data type
     * @param src Vector containing current values of source array with multi-vector array stored
     * in a flattened format with all the wavefunction value corresponding to a
     * given node is stored contiguously.
     * @param numberWaveFunctions Number of wavefunctions at a given node.
     */
    void
    computeNonLocalProjectorKetTimesXTimesV(
      const dataTypes::number *                src,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                       numberWaveFunctions);

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
    std::vector<dataTypes::number> d_cellHamiltonianMatrixFlattened;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamMatrixTimesWaveMatrix;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_kpointCoordsVecDevice;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_kSquareTimesHalfVecDevice;

    /// for non local

    std::vector<dataTypes::number>
      d_cellHamiltonianMatrixNonLocalFlattenedConjugate;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedConjugateDevice;
    std::vector<dataTypes::number>
      d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellHamMatrixTimesWaveMatrixNonLocalDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_projectorKetTimesVectorParFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_projectorKetTimesVectorAllCellsDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
                        d_projectorKetTimesVectorDevice;
    std::vector<double> d_nonLocalPseudoPotentialConstants;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_nonLocalPseudoPotentialConstantsDevice;

    std::vector<dataTypes::number> d_projectorKetTimesVectorAllCellsReduction;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
                              d_projectorKetTimesVectorAllCellsReductionDevice;
    std::vector<unsigned int> d_pseudoWfcAccumNonlocalAtoms;
    unsigned int              d_totalNonlocalAtomsCurrentProc;
    unsigned int              d_totalNonlocalElems;
    unsigned int              d_totalPseudoWfcNonLocal;
    unsigned int              d_maxSingleAtomPseudoWfc;
    std::vector<unsigned int> d_nonlocalElemIdToLocalElemIdMap;
    std::vector<unsigned int> d_pseduoWfcNonLocalAtoms;
    std::vector<unsigned int> d_numberCellsNonLocalAtoms;
    std::vector<unsigned int> d_numberCellsAccumNonLocalAtoms;
    std::vector<dealii::types::global_dof_index>
      d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal;
    dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                dftfe::utils::MemorySpace::DEVICE>
                              d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice;
    std::vector<unsigned int> d_projectorIdsParallelNumberingMap;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
                     d_projectorIdsParallelNumberingMapDevice;
    std::vector<int> d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE>
                              d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice;
    std::vector<unsigned int> d_cellNodeIdMapNonLocalToLocal;
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
                              d_cellNodeIdMapNonLocalToLocalDevice;
    std::vector<unsigned int> d_normalCellIdToMacroCellIdMap;
    std::vector<unsigned int> d_macroCellIdToNormalCellIdMap;

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_locallyOwnedProcBoundaryNodesVectorDevice;

    bool                d_isMallocCalled = false;
    dataTypes::number **d_A, **d_B, **d_C;
    dataTypes::number **h_d_A, **h_d_B, **h_d_C;

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
      const dataTypes::number *src,
      const unsigned int       numberWaveFunctions,
      dataTypes::number *      dst,
      bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

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
      const dataTypes::number *                src,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                       numberWaveFunctions,
      dataTypes::number *                      dst,
      const bool                               skip1 = false,
      const bool                               skip2 = false);



    /// pointer to dft class
    dftClass<FEOrder, FEOrderElectro> *dftPtr;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsPtrDevice;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrHost;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperPtrHost;


    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperPtr;


    /// data structures to store diagonal of inverse square root mass matrix and
    /// square root of mass matrix
    distributedCPUVec<double> d_invSqrtMassVector, d_sqrtMassVector;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_invSqrtMassVectorDevice, d_sqrtMassVectorDevice;

    std::vector<double> d_vEff;
    std::vector<double> d_vEffExternalPotCorrJxW;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                        d_vEffExternalPotCorrJxWDevice;
    std::vector<double> d_vEffJxW;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_vEffJxWDevice;

    const unsigned int d_numQuadPoints;
    unsigned int       d_numQuadPointsLpsp;
    const unsigned int d_numLocallyOwnedCells;

    std::vector<double> d_derExcWithSigmaTimesGradRhoJxW;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_derExcWithSigmaTimesGradRhoJxWDevice;


    /**
     * @brief finite-element cell level matrix to store dot product between shapeFunction gradients (\int(\nabla N_i \cdot \nabla N_j))
     */

    // storage for  matrix-free cell data
    const unsigned int        d_numberNodesPerElement;
    const unsigned int        d_numberMacroCells;
    std::vector<unsigned int> d_macroCellSubCellMap;

    // parallel objects
    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;

    // compute-time logger
    dealii::TimerOutput computing_timer;

    // mutex thread for managing multi-thread writing to XHXvalue
    // mutable dealii::Threads::Mutex  assembler_lock;

    // d_kpoint index for which Hamiltonian is computed
    unsigned int d_kPointIndex;

    unsigned int d_spinIndex;

    // storage for precomputing index maps
    std::vector<dealii::types::global_dof_index>
      d_flattenedArrayCellLocalProcIndexIdMap;

    // storage for precomputing index maps
    std::vector<dealii::types::global_dof_index>
      d_flattenedArrayMacroCellLocalProcIndexIdMapFlattened;
    dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                dftfe::utils::MemorySpace::DEVICE>
      d_DeviceFlattenedArrayMacroCellLocalProcIndexIdMapFlattened;

    /// storage for deviceblas handle
    dftfe::utils::deviceBlasHandle_t d_deviceBlasHandle;

    /// flag for precomputing stiffness matrix contribution from
    /// sum{Vext}-sum{Vnuc}
    bool d_isStiffnessMatrixExternalPotCorrComputed;

    /// external potential correction quadrature id
    unsigned int d_externalPotCorrQuadratureId;

    /// Temporary storage for real and imaginary portions of the complex
    /// wavefunction vectors
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_tempRealVec;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      d_tempImagVec;
  };
} // namespace dftfe
#endif
