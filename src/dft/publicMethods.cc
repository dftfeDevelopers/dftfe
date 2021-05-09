// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
const std::vector<dealii::types::global_dof_index> &
dftClass<FEOrder, FEOrderElectro>::getLocalDofIndicesReal() const
{
  return local_dof_indicesReal;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
const std::vector<dealii::types::global_dof_index> &
dftClass<FEOrder, FEOrderElectro>::getLocalDofIndicesImag() const
{
  return local_dof_indicesImag;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
const std::vector<dealii::types::global_dof_index> &
dftClass<FEOrder, FEOrderElectro>::getLocalProcDofIndicesReal() const
{
  return localProc_dof_indicesReal;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
const std::vector<dealii::types::global_dof_index> &
dftClass<FEOrder, FEOrderElectro>::getLocalProcDofIndicesImag() const
{
  return localProc_dof_indicesImag;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
const dealii::AffineConstraints<double> &
dftClass<FEOrder, FEOrderElectro>::getConstraintMatrixEigen() const
{
  return constraintsNoneEigen;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
const dftUtils::constraintMatrixInfo &
dftClass<FEOrder, FEOrderElectro>::getConstraintMatrixEigenDataInfo() const
{
  return constraintsNoneEigenDataInfo;
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
const MatrixFree<3, double> &
dftClass<FEOrder, FEOrderElectro>::getMatrixFreeData() const
{
  return matrix_free_data;
}
