//
// Created by Arghadwip Paul, Sambit Das
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIXATOMICBASIS_H
#define DFTFE_AUXDM_AUXDENSITYMATRIXATOMICBASIS_H

#include "AuxDensityMatrix.h"
#include "AtomicBasis.h"
#include "AtomicBasisData.h"
#include <vector>
#include <utility>
#include <map>
#include <algorithm>


namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class AuxDensityMatrixAtomicBasis : public AuxDensityMatrix<memorySpace>
  {
  public:
    void
    reinit(
      const AtomicBasis::BasisType basisType,
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                                                 atomCoords,
      const std::unordered_map<std::string, std::string> &atomBasisFileNames,
      const int                                           nSpin,
      const int                                           maxDerOrder);

    void
    applyLocalOperations(
      const std::vector<double> &quadpts,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) override;

    void
    evalOverlapMatrixStart(const std::vector<double> &quadpts,
                           const std::vector<double> &quadWt) override;

    void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) override;

    /**
     *
     * @param projectionInputs is a map from string to inputs needed
     *                          for projection.
     *      eg - projectionInputsReal["quadpts"],
     *          projectionInputsReal["quadWt"],
     *          projectionInputsDataType["psiFunc"],
     *          projectionInputsReal["fValues"]
     *
     *      psiFunc The SCF wave function or eigen function in FE Basis.
     *                psiFunc(quad_index, wfc_index),
     *                quad_index is fastest.
     *      fValues are the occupancies.
     *
     * @param iSpin indicates up (iSpin = 0) or down (iSpin = 0) spin.
     *
     */
    virtual void
    projectDensityMatrixStart(
      const std::unordered_map<std::string, std::vector<dataTypes::number>>
        &projectionInputsDataType,
      const std::unordered_map<std::string, std::vector<double>>
        &       projectionInputsReal,
      const int iSpin) override;

    void
    projectDensityMatrixEnd(const MPI_Comm &mpiComm) override;

    void
    projectDensityStart(
      const std::unordered_map<std::string, std::vector<double>>
        &projectionInputs) override;

    void
    projectDensityEnd(const MPI_Comm &mpiComm) override;


  private:
    int                          d_nQuad;
    int                          d_nSpin;
    std::unique_ptr<AtomicBasis> d_atomicBasisPtr;
    AtomicBasisData              d_atomicBasisData;

    int d_nBasis;
    int d_maxDerOrder;

    std::vector<double> d_DM;
    std::vector<double> d_SMatrix;
    std::vector<double> d_SMatrixInv;
    std::vector<double> d_basisWFCInnerProducts;
    std::vector<double> d_fValues;

    int d_nWFC;
    int d_iSpin;


    void
    evalOverlapMatrixInv();
    std::vector<double> &
    getOverlapMatrixInv();
  };
} // namespace dftfe
#endif // DFTFE_AUXDM_AUXDENSITYMATRIXATOMICBASIS_H
