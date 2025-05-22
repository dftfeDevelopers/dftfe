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


#ifndef kerkerSolverProblemWrapper_H_
#define kerkerSolverProblemWrapper_H_
#include <variant>
#include <memory>
#include <headers.h>
#include <kerkerSolverProblem.h>
#ifdef DFTFE_WITH_DEVICE
#  include <kerkerSolverProblemDevice.h>
#endif
namespace dftfe
{
  using kerkerSolverProblemObject = std::variant<
#define kerkerSolverProblemWrapperTemplates(T1) \
  std::shared_ptr<kerkerSolverProblem<T1>>,
#define kerkerSolverProblemWrapperTemplatesL(T1) \
  std::shared_ptr<kerkerSolverProblem<T1>>
#include "kerkerSolverProblemWrapper.def"
#undef kerkerSolverProblemWrapperTemplates
#undef kerkerSolverProblemWrapperTemplatesL
    >;
  template <class... Args>
  inline kerkerSolverProblemObject
  createKerkerSolverProblemObject(dftfe::Int feOrder, Args &&...args)
  {
    switch (feOrder)
      {
#define kerkerSolverProblemWrapperTemplates(T1) \
  case T1:                                      \
    return kerkerSolverProblemObject(           \
      std::make_shared<kerkerSolverProblem<T1>>(std::forward<Args>(args)...));
#define kerkerSolverProblemWrapperTemplatesL(T1) \
  case T1:                                       \
    return kerkerSolverProblemObject(            \
      std::make_shared<kerkerSolverProblem<T1>>(std::forward<Args>(args)...));
#include "kerkerSolverProblemWrapper.def"
#undef kerkerSolverProblemWrapperTemplates
#undef kerkerSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createKerkerSolverProblemObject dispatch failed"};
      }
  }


  class kerkerSolverProblemWrapperClass : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    kerkerSolverProblemWrapperClass(const dftfe::Int feOrder,
                                    const MPI_Comm  &mpi_comm_parent,
                                    const MPI_Comm  &mpi_comm_domain)
      : d_kerkerSolverProblemObject(
          createKerkerSolverProblemObject(feOrder,
                                          mpi_comm_parent,
                                          mpi_comm_domain))
    {}

    distributedCPUVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedCPUVec<double> & { return t->getX(); },
        d_kerkerSolverProblemObject);
    }

    void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x)
    {
      std::visit([&](auto &t) { t->vmult(Ax, x); },
                 d_kerkerSolverProblemObject);
    }

    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_kerkerSolverProblemObject);
    }

    void
    precondition_Jacobi(distributedCPUVec<double>       &dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const
    {
      std::visit([&](
                   auto const &t) { t->precondition_Jacobi(dst, src, omega); },
                 d_kerkerSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); },
                 d_kerkerSolverProblemObject);
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

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_kerkerSolverProblemObject);
    }

    template <typename... Args>
    void
    init(Args &&...args)
    {
      std::visit([&](auto &t) { t->init(std::forward<Args>(args)...); },
                 d_kerkerSolverProblemObject);
    }

  private:
    kerkerSolverProblemObject d_kerkerSolverProblemObject;
  };

#ifdef DFTFE_WITH_DEVICE
  using kerkerSolverProblemDeviceObject = std::variant<
#  define kerkerSolverProblemWrapperTemplates(T1) \
    std::shared_ptr<kerkerSolverProblemDevice<T1>>,
#  define kerkerSolverProblemWrapperTemplatesL(T1) \
    std::shared_ptr<kerkerSolverProblemDevice<T1>>
#  include "kerkerSolverProblemWrapper.def"
#  undef kerkerSolverProblemWrapperTemplates
#  undef kerkerSolverProblemWrapperTemplatesL
    >;


  template <class... Args>
  inline kerkerSolverProblemDeviceObject
  createKerkerSolverProblemDeviceObject(dftfe::Int feOrder, Args &&...args)
  {
    switch (feOrder)
      {
#  define kerkerSolverProblemWrapperTemplates(T1)        \
    case T1:                                             \
      return kerkerSolverProblemDeviceObject{            \
        std::make_shared<kerkerSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...)};
#  define kerkerSolverProblemWrapperTemplatesL(T1)       \
    case T1:                                             \
      return kerkerSolverProblemDeviceObject{            \
        std::make_shared<kerkerSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...)};
#  include "kerkerSolverProblemWrapper.def"
#  undef kerkerSolverProblemWrapperTemplates
#  undef kerkerSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createKerkerSolverProblemDeviceObject dispatch failed"};
      }
  }


  class kerkerSolverProblemDeviceWrapperClass : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    kerkerSolverProblemDeviceWrapperClass(const dftfe::Int feOrder,
                                          const MPI_Comm  &mpi_comm_parent,
                                          const MPI_Comm  &mpi_comm_domain)
      : d_kerkerSolverProblemObject(
          createKerkerSolverProblemDeviceObject(feOrder,
                                                mpi_comm_parent,
                                                mpi_comm_domain))
    {}

    distributedDeviceVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & { return t->getX(); },
        d_kerkerSolverProblemObject);
    }

    distributedDeviceVec<double> &
    getPreconditioner()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & {
          return t->getPreconditioner();
        },
        d_kerkerSolverProblemObject);
    }

    void
    computeAX(distributedDeviceVec<double> &dst,
              distributedDeviceVec<double> &src)
    {
      std::visit([&](auto &t) { t->computeAX(dst, src); },
                 d_kerkerSolverProblemObject);
    }


    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_kerkerSolverProblemObject);
    }

    void
    setX()
    {
      std::visit([](auto &t) { t->setX(); }, d_kerkerSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); },
                 d_kerkerSolverProblemObject);
    }

    void
    copyXfromDeviceToHost()
    {
      std::visit([](auto &t) { t->copyXfromDeviceToHost(); },
                 d_kerkerSolverProblemObject);
    }

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_kerkerSolverProblemObject);
    }

    template <typename... Args>
    void
    init(Args &&...args)
    {
      std::visit([&](auto &t) { t->init(std::forward<Args>(args)...); },
                 d_kerkerSolverProblemObject);
    }

  private:
    kerkerSolverProblemDeviceObject d_kerkerSolverProblemObject;
  };

#endif

} // namespace dftfe

#endif
