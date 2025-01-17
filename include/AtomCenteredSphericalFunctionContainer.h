// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionBase.h"
#include <memory>
#include <MemorySpaceType.h>
#include "FEBasisOperations.h"
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>


namespace dftfe
{
  class AtomCenteredSphericalFunctionContainer
  {
  public:
    /**
     * @brief Initialises the class with the atomicNumbers of various atoms and the AtomCenteredSphericalFn of various spherical functions. This function is only called once per run.
     * @param[in] atomicNumbers vector of size Natoms storing the Znumbers of
     * various atoms present
     * @param[in] listOfSphericalFunctions map of std::pain (Znum, l) to the
     * sphericalFUnction class shared pointer.
     */
    void
    init(const std::vector<unsigned int> &atomicNumbers,
         const std::map<std::pair<unsigned int, unsigned int>,
                        std::shared_ptr<AtomCenteredSphericalFunctionBase>>
           &listOfSphericalFunctions);
    /**
     * @brief Initialises the position of atoms, the image posisiton and image ids after every update of atom positions.
     * @param[in] atomCoords vector of size 3*Natoms storing the X,Y,Z
     * coordiantes of atom in cell.
     * @param[in] periodicCoords vector of vector storing the image coordinates
     * @param[in] imageIds the image Id of image atoms present in periodicCoords
     * input
     */
    void
    initaliseCoordinates(const std::vector<double> &             atomCoords,
                         const std::vector<std::vector<double>> &periodicCoords,
                         const std::vector<int> &                imageIds);
    /**
     * @brief Returns the number of atoms present in domain
     * @return  Returns size of atomicNumbers vector
     */
    unsigned int
    getNumAtomCentersSize();


    /**
     * @brief Returns the cooridnates of atom present in domain
     * @return  Returns atomCoords vector
     */
    const std::vector<double> &
    getAtomCoordinates() const;
    /**
     * @brief Returns the map of atomId vs vector of image coordinates
     * @return  Returns d_periodicImageCoord
     */
    const std::map<unsigned int, std::vector<double>> &
    getPeriodicImageCoordinatesList() const;

    // This functions returns the number of spherical functions associated with
    // an atomic number.
    // If the atomic number does not exist, it returns a zero.
    /**
     * @brief Returns the he number of total spherical functions indexed by {ilm} associated with  an atomic number. If the atomic number does not exist, it returns a zero.
     * @return d_numSphericalFunctions.find(atomicNumber)->size()
     */
    unsigned int
    getTotalNumberOfSphericalFunctionsPerAtom(unsigned int atomicNumber);

    /**
     * @brief Returns the he number of radial spherical functions indexed by {i} associated with  an atomic number. If the atomic number does not exist, it returns a zero.
     * @return d_numRadialSphericalFunctions.find(atomicNumber)->size()
     */
    unsigned int
    getTotalNumberOfRadialSphericalFunctionsPerAtom(unsigned int atomicNumber);
    /**
     * @brief Returns the total number of total spherical functions indexed by {ilm} present in the current processor. If the atomic number does not exist, it returns a zero.
     */
    unsigned int
    getTotalNumberOfSphericalFunctionsInCurrentProcessor();
    /**
     * @brief Returns the maximum number of total spherical functions indexed by {ilm} across all atom Types present in atomNumbers vector
     */
    unsigned int
    getMaximumNumberOfSphericalFunctions();
    /**
     * @brief
     * @param[out] totalAtomsInCurrentProcessor number of atoms in current
     * processor based on compact support
     * @param[out] totalNonLocalElements number of nonLocal elements in current
     * processor
     * @param[out] numberCellsForEachAtom number of cells associated which each
     * atom in the current processor. vecot of size totalAtomsInCurrentProcessor
     * @param[out] numberCellsAccumNonLocalAtoms number of cells accumulated
     * till iatom in current processor. vector of size
     * totalAtomsInCurrentProcessor
     */
    void
    getTotalAtomsAndNonLocalElementsInCurrentProcessor(
      unsigned int &             totalAtomsInCurrentProcessor,
      unsigned int &             totalNonLocalElements,
      std::vector<unsigned int> &numberCellsForEachAtom,
      std::vector<unsigned int> &numberCellsAccumNonLocalAtoms);

    /**
     * @brief Returns the total number of total radial-spherical functions indexed by {i} present in atomicNumbers list.
     */
    unsigned int
    getTotalNumberOfRadialSphericalFunctions();

    /**
     * @brief Returns the shared_ptr of AtomCenteredSphericalFunctionBase associated with std::pair(atomic Number and lQuantumNo)
     */
    const std::map<std::pair<unsigned int, unsigned int>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>> &
    getSphericalFunctions() const;
    /**
     * @brief Returns the vector of size Natoms of all atoms in system
     */
    const std::vector<unsigned int> &
    getAtomicNumbers() const;
    /**
     * @brief Returns the atomIds of atoms present in current processor
     */
    const std::vector<unsigned int> &
    getAtomIdsInCurrentProcess() const;
    /**
     * @brief Returns the startIndex of spherical Function alpha associated with atomic number Znum
     */
    const unsigned int
    getTotalSphericalFunctionIndexStart(unsigned int Znum, unsigned int alpha);
    // COmputes the sparsity Pattern for the compact support Fn
    // cutOffVal the max/min value to consider to be part of copact support
    // cutOffType = 0 based on Fn Value, cutOffType = 1 based on Distance from
    // atom
    template <typename NumberType>
    void
    computeSparseStructure(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<NumberType,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureIndex,
      const double       cutOffVal  = 1.0E-8,
      const unsigned int cutOffType = 0);


    std::vector<std::vector<unsigned int>> d_elementIndexesInAtomCompactSupport;
    void
    setImageCoordinates(const std::vector<int> &                imageIds,
                        const std::vector<std::vector<double>> &periodicCoords);



    const std::vector<int> &
    getAtomIdsInElement(unsigned int iElem);

    const std::map<unsigned int, std::vector<int>> &
    getSparsityPattern();

    bool
    atomSupportInElement(unsigned int iElem);

    void
    getDataForSparseStructure(
      const std::map<unsigned int, std::vector<int>> &sparsityPattern,
      const std::vector<std::vector<dealii::CellId>>
        &elementIdsInAtomCompactSupport,
      const std::vector<std::vector<unsigned int>>
        &                              elementIndexesInAtomCompactSupport,
      const std::vector<unsigned int> &atomIdsInCurrentProcess,
      unsigned int                     numberElements);

    const unsigned int
    getOffsetLocation(const unsigned int iAtom);

  private:
    // A flattened vector that stores the coordinates of the atoms of interest
    // in the unit cell
    // Coord of atom I is stored at 3*I +0 ( x-coord),3*I+1 ( y-coord),3*I+2 (
    // z-coord)
    std::vector<double> d_atomCoords;

    // A vector of size = number of atoms of interest
    // the Ith atom in d_atomicNumbers has its coordinates
    // in d_atomCoords[3*I+0], d_atomCoords[3*I+1], d_atomCoords[3*I+2]
    std::vector<unsigned int> d_atomicNumbers;

    // This maps the atom I in the unit cell to all its image atoms.
    // number of image atoms of Ith atom = d_periodicImageCoord[I].size()/ dim
    // with dim=3 The coordinates are stored as a flattened vector
    std::map<unsigned int, std::vector<double>> d_periodicImageCoord;


    // This maps, from std::pair<atomic number, \alpha> to S_{z,\alpha},
    // where \alpha is the index for unique radial function
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_sphericalFunctionsContainer;
    // Stores the number of  distinct Radial Functions for a particular AtomType
    std::map<unsigned int, unsigned int> d_numRadialSphericalFunctions;
    // Stores the number of  distinct  Functions include m for a particular
    // AtomType
    std::map<unsigned int, unsigned int> d_numSphericalFunctions;
    // This maps is between atomId in unit cell and the sparsity pattern of the
    // atom and its images in the unitcell domain.
    std::map<unsigned int, std::vector<int>> d_sparsityPattern;
    //
    std::vector<std::vector<dealii::CellId>> d_elementIdsInAtomCompactSupport;
    // std::vector<std::vector<unsigned int>>
    // d_elementIndexesInAtomCompactSupport;
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
                                  d_elementOneFieldIteratorsInAtomCompactSupport;
    std::vector<unsigned int>     d_AtomIdsInCurrentProcess;
    std::vector<unsigned int>     d_offsetLocation;
    std::vector<std::vector<int>> d_AtomIdsInElement;
    std::map<unsigned int, std::vector<unsigned int>>
      d_totalSphericalFunctionIndexStart;

  }; // end of class AtomCenteredSphericalFunctionContainerBase
} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONCONTAINERBASE_H
