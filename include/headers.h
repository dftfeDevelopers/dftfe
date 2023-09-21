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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Sambit Das (2018)
//

#ifndef headers_H_
#define headers_H_

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// Include all deal.II header file
#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/base/function.h>
#  include <deal.II/base/logstream.h>
#  include <deal.II/base/point.h>
#  include <deal.II/base/quadrature.h>
//#  include <deal.II/base/quadrature_point_data.h>
#  include <deal.II/base/table.h>
#  include <deal.II/base/tensor_function.h>
#  include <deal.II/base/timer.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/distributed/grid_refinement.h>
#  include <deal.II/distributed/solution_transfer.h>
#  include <deal.II/distributed/tria.h>

#  include <deal.II/dofs/dof_accessor.h>
#  include <deal.II/dofs/dof_handler.h>
#  include <deal.II/dofs/dof_renumbering.h>
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_system.h>
#  include <deal.II/fe/fe_values.h>
#  include <deal.II/fe/mapping_q1.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/grid_in.h>
#  include <deal.II/grid/grid_out.h>
#  include <deal.II/grid/grid_refinement.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/tria_accessor.h>
#  include <deal.II/grid/tria_iterator.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/exceptions.h>
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/lapack_full_matrix.h>
#  include <deal.II/lac/precondition.h>
#  include <deal.II/lac/solver_cg.h>
#  include <deal.II/lac/solver_gmres.h>
#  include <deal.II/lac/vector.h>

#  include <deal.II/matrix_free/fe_evaluation.h>
#  include <deal.II/matrix_free/matrix_free.h>

#  include <deal.II/numerics/data_out.h>
#  include <deal.II/numerics/error_estimator.h>
#  include <deal.II/numerics/matrix_tools.h>
#  include <deal.II/numerics/vector_tools.h>
#  ifdef USE_PETSC
#    include <deal.II/lac/slepc_solver.h>
#  endif
#  include <deal.II/base/config.h>

#  include <deal.II/base/smartpointer.h>
#  include <deal.II/base/types.h>

#  include <dftfeDataTypes.h>
#  include <MultiVector.h>

// Include generic C++ headers
#  include <fstream>
#  include <iostream>
#  include <fenv.h>

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
// commonly used  typedefs used in dftfe go here
namespace dftfe
{
  template <typename elem_type>
  using distributedCPUVec =
    dealii::LinearAlgebra::distributed::Vector<elem_type,
                                               dealii::MemorySpace::Host>;

  template <typename NumberType>
  using distributedCPUMultiVec =
    dftfe::linearAlgebra::MultiVector<NumberType,
                                      dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE
  template <typename NumberType>
  using distributedDeviceVec =
    dftfe::linearAlgebra::MultiVector<NumberType,
                                      dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe

#endif
