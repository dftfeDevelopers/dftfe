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


#ifndef poissonSolverProblemWrapper_H_
#define poissonSolverProblemWrapper_H_
#include <variant>
#include <memory>
#include <headers.h>
#include <poissonSolverProblem.h>
#ifdef DFTFE_WITH_DEVICE
#  include <poissonSolverProblemDevice.h>
#endif
namespace dftfe
{
  using poissonSolverProblemObject = std::variant<
#define poissonSolverProblemWrapperTemplates(T1) \
  std::shared_ptr<poissonSolverProblem<T1>>,
#define poissonSolverProblemWrapperTemplatesL(T1) \
  std::shared_ptr<poissonSolverProblem<T1>>
#include "poissonSolverProblemWrapper.def"
#undef poissonSolverProblemWrapperTemplates
#undef poissonSolverProblemWrapperTemplatesL
    >;
  template <class... Args>
  inline poissonSolverProblemObject
  createPoissonSolverProblemObject(dftfe::Int feOrderElectro, Args &&...args)
  {
    switch (feOrderElectro)
      {
#define poissonSolverProblemWrapperTemplates(T1)  \
  case T1:                                        \
    return poissonSolverProblemObject(            \
      std::make_shared<poissonSolverProblem<T1>>( \
        std::forward<Args>(args)...));
#define poissonSolverProblemWrapperTemplatesL(T1) \
  case T1:                                        \
    return poissonSolverProblemObject(            \
      std::make_shared<poissonSolverProblem<T1>>( \
        std::forward<Args>(args)...));
#include "poissonSolverProblemWrapper.def"
#undef poissonSolverProblemWrapperTemplates
#undef poissonSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createPoissonSolverProblemObject dispatch failed"};
      }
  }


  class poissonSolverProblemWrapperClass : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    poissonSolverProblemWrapperClass(const dftfe::Int feOrderElectro,
                                     const MPI_Comm  &mpi_comm)
      : d_poissonSolverProblemObject(
          createPoissonSolverProblemObject(feOrderElectro, mpi_comm))
    {}

    distributedCPUVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedCPUVec<double> & { return t->getX(); },
        d_poissonSolverProblemObject);
    }

    void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x)
    {
      std::visit([&](auto &t) { t->vmult(Ax, x); },
                 d_poissonSolverProblemObject);
    }

    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_poissonSolverProblemObject);
    }

    void
    precondition_Jacobi(distributedCPUVec<double>       &dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const
    {
      std::visit([&](
                   auto const &t) { t->precondition_Jacobi(dst, src, omega); },
                 d_poissonSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); },
                 d_poissonSolverProblemObject);
    }

    void
    subscribe(std::atomic<bool> *const validity,
              const std::string       &identifier = "") const
    {}

    void
    unsubscribe(std::atomic<bool> *const validity,
                const std::string       &identifier = "") const
    {}

    bool
    operator!=(double val) const
    {
      return true;
    }

    void
    clear()
    {
      std::visit([](auto &t) { t->clear(); }, d_poissonSolverProblemObject);
    }

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_poissonSolverProblemObject);
    }

  private:
    poissonSolverProblemObject d_poissonSolverProblemObject;
  };

#ifdef DFTFE_WITH_DEVICE
  using poissonSolverProblemDeviceObject = std::variant<
#  define poissonSolverProblemWrapperTemplates(T1) \
    std::shared_ptr<poissonSolverProblemDevice<T1>>,
#  define poissonSolverProblemWrapperTemplatesL(T1) \
    std::shared_ptr<poissonSolverProblemDevice<T1>>
#  include "poissonSolverProblemWrapper.def"
#  undef poissonSolverProblemWrapperTemplates
#  undef poissonSolverProblemWrapperTemplatesL
    >;


  template <class... Args>
  inline poissonSolverProblemDeviceObject
  createPoissonSolverProblemDeviceObject(dftfe::Int feOrderElectro,
                                         Args &&...args)
  {
    switch (feOrderElectro)
      {
#  define poissonSolverProblemWrapperTemplates(T1)        \
    case T1:                                              \
      return poissonSolverProblemDeviceObject(            \
        std::make_shared<poissonSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...));
#  define poissonSolverProblemWrapperTemplatesL(T1)       \
    case T1:                                              \
      return poissonSolverProblemDeviceObject(            \
        std::make_shared<poissonSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...));
#  include "poissonSolverProblemWrapper.def"
#  undef poissonSolverProblemWrapperTemplates
#  undef poissonSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createPoissonSolverProblemDeviceObject dispatch failed"};
      }
  }


  class poissonSolverProblemDeviceWrapperClass
    : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    poissonSolverProblemDeviceWrapperClass(const dftfe::Int feOrderElectro,
                                           const MPI_Comm  &mpi_comm)
      : d_poissonSolverProblemObject(
          createPoissonSolverProblemDeviceObject(feOrderElectro, mpi_comm))
    {}

    distributedDeviceVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & { return t->getX(); },
        d_poissonSolverProblemObject);
    }

    distributedDeviceVec<double> &
    getPreconditioner()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & {
          return t->getPreconditioner();
        },
        d_poissonSolverProblemObject);
    }

    void
    computeAX(distributedDeviceVec<double> &dst,
              distributedDeviceVec<double> &src)
    {
      std::visit([&](auto &t) { t->computeAX(dst, src); },
                 d_poissonSolverProblemObject);
    }


    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_poissonSolverProblemObject);
    }

    void
    setX()
    {
      std::visit([](auto &t) { t->setX(); }, d_poissonSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); },
                 d_poissonSolverProblemObject);
    }

    void
    copyXfromDeviceToHost()
    {
      std::visit([](auto &t) { t->copyXfromDeviceToHost(); },
                 d_poissonSolverProblemObject);
    }

    void
    clear()
    {
      std::visit([](auto &t) { t->clear(); }, d_poissonSolverProblemObject);
    }

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_poissonSolverProblemObject);
    }

  private:
    poissonSolverProblemDeviceObject d_poissonSolverProblemObject;
  };

#endif

} // namespace dftfe

#endif
