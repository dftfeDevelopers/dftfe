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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Sambit Das (2018)
//


template<unsigned int FEOrder>
const std::vector<unsigned int> & dftClass<FEOrder>::getLocalDofIndicesReal()
{
  return  local_dof_indicesReal;
}

template<unsigned int FEOrder>
const std::vector<unsigned int> & dftClass<FEOrder>::getLocalDofIndicesImag()
{
  return  local_dof_indicesImag;
}

template<unsigned int FEOrder>
const std::vector<unsigned int> & dftClass<FEOrder>::getLocalProcDofIndicesReal()
{
  return  localProc_dof_indicesReal;
}

template<unsigned int FEOrder>
const std::vector<unsigned int> & dftClass<FEOrder>::getLocalProcDofIndicesImag()
{
  return  localProc_dof_indicesImag;
}

template<unsigned int FEOrder>
const ConstraintMatrix & dftClass<FEOrder>::getConstraintMatrixEigen()
{
  return  constraintsNoneEigen;
}

template<unsigned int FEOrder>
const dftUtils::constraintMatrixInfo & dftClass<FEOrder>::getConstraintMatrixEigenDataInfo()
{
  return constraintsNoneEigenDataInfo;
}
