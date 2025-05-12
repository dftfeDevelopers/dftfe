// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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


#ifndef FEEvaluationWrapper_H_
#define FEEvaluationWrapper_H_
#include <variant>
#include <memory>
#include <headers.h>
namespace dftfe
{
  constexpr dftfe::Int
  encodeFEEvaluation(dftfe::Int a, dftfe::Int b)
  {
    return a * 100 + b;
  }

  template <dftfe::Int components>
  struct FEEvalTrait
  {
    using type = std::variant<
#define FEEvaluationWrapperTemplates(T1, T2) \
  std::shared_ptr<dealii::FEEvaluation<3, T1, T2, components>>,
#include "feevaluationWrapper.def"
#undef FEEvaluationWrapperTemplates
      std::shared_ptr<dealii::FEEvaluation<3, -1, 1, components>>>;
  };

  template <>
  struct FEEvalTrait<3>
  {
    using type = std::variant<
#define FEEvaluationWrapperTemplates(T1, T2) \
  std::shared_ptr<dealii::FEEvaluation<3, T1, T2, 3>>,
#include "feevaluationWrapper3Comp.def"
#undef FEEvaluationWrapperTemplates
      std::shared_ptr<dealii::FEEvaluation<3, -1, 1, 3>>>;
  };


  template <dftfe::Int components>
  using FEEvaluationObject = typename FEEvalTrait<components>::type;


  template <dftfe::Int components, class... Args>
  inline FEEvaluationObject<components>
  createFEEvaluationObject(dftfe::Int feOrder,
                           dftfe::Int quadrature,
                           Args &&...args)
  {
    const dftfe::Int key = encodeFEEvaluation(feOrder, quadrature);
    if constexpr (components == 1)
      {
        switch (key)
          {
#define FEEvaluationWrapperTemplates(T1, T2)                         \
  case encodeFEEvaluation(T1, T2):                                   \
    return FEEvaluationObject<components>(                           \
      std::make_shared<dealii::FEEvaluation<3, T1, T2, components>>( \
        std::forward<Args>(args)...));
#include "feevaluationWrapper.def"
#undef FEEvaluationWrapperTemplates
            default:
              return FEEvaluationObject<components>(
                std::make_shared<dealii::FEEvaluation<3, -1, 1, components>>(
                  std::forward<Args>(args)...));
          };
      }
    if constexpr (components == 3)
      {
        switch (key)
          {
#define FEEvaluationWrapperTemplates(T1, T2)                         \
  case encodeFEEvaluation(T1, T2):                                   \
    return FEEvaluationObject<components>(                           \
      std::make_shared<dealii::FEEvaluation<3, T1, T2, components>>( \
        std::forward<Args>(args)...));
#include "feevaluationWrapper3Comp.def"
#undef FEEvaluationWrapperTemplates
            default:
              return FEEvaluationObject<components>(
                std::make_shared<dealii::FEEvaluation<3, -1, 1, components>>(
                  std::forward<Args>(args)...));
          };
      }
  }

  template <dftfe::Int components>
  class FEEvaluationWrapperClass
  {
  public:
    /// Constructor
    FEEvaluationWrapperClass(
      const dftfe::Int                     feOrder,
      const dftfe::Int                     quadrature,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const dftfe::uInt                    matrixFreeVectorComponent,
      const dftfe::uInt                    matrixFreeQuadratureComponent)
      : d_FEEvaluationObject(
          createFEEvaluationObject<components>(feOrder,
                                               quadrature,
                                               matrixFreeData,
                                               matrixFreeVectorComponent,
                                               matrixFreeQuadratureComponent))
      , n_q_points(matrixFreeData.get_n_q_points(matrixFreeQuadratureComponent))
    {}

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    read_dof_values(Args &&...args)
    {
      std::visit(
        [&](auto &t) { t->read_dof_values(std::forward<Args>(args)...); },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    read_dof_values_plain(Args &&...args)
    {
      std::visit(
        [&](auto &t) { t->read_dof_values_plain(std::forward<Args>(args)...); },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    evaluate(Args &&...args)
    {
      std::visit([&](auto &t) { t->evaluate(std::forward<Args>(args)...); },
                 d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    submit_gradient(Args &&...args)
    {
      std::visit(
        [&](auto &t) { t->submit_gradient(std::forward<Args>(args)...); },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    submit_value(Args &&...args)
    {
      std::visit([&](auto &t) { t->submit_value(std::forward<Args>(args)...); },
                 d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    get_value(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->get_value(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    get_gradient(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->get_gradient(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    integrate_value(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->integrate_value(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    get_hessian(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->get_hessian(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    integrate(Args &&...args)
    {
      std::visit([&](auto &t) { t->integrate(std::forward<Args>(args)...); },
                 d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    quadrature_point(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->quadrature_point(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    decltype(auto)
    JxW(Args &&...args)
    {
      return std::visit(
        [&](auto &t) -> decltype(auto) {
          return t->JxW(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }

    template <typename... Args>
    void
    distribute_local_to_global(Args &&...args)
    {
      std::visit(
        [&](auto &t) {
          t->distribute_local_to_global(std::forward<Args>(args)...);
        },
        d_FEEvaluationObject);
    }


    const dftfe::uInt n_q_points;

  private:
    FEEvaluationObject<components> d_FEEvaluationObject;
  };

} // namespace dftfe

#endif
