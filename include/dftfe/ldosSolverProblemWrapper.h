#ifndef ldosSolverProblemWrapper_H_
#define ldosSolverProblemWrapper_H_
#include <variant>
#include <memory>
#include <dftfe/headers.h>
#include <dftfe/ldosSolverProblem.h>
#ifdef DFTFE_WITH_DEVICE
#  include <dftfe/ldosSolverProblemDevice.h>
#endif

namespace dftfe
{
  using ldosSolverProblemObject = std::variant<
#define ldosSolverProblemWrapperTemplates(T1) \
  std::shared_ptr<ldosSolverProblem<T1>>,
#define ldosSolverProblemWrapperTemplatesL(T1) \
  std::shared_ptr<ldosSolverProblem<T1>>
#include "ldosSolverProblemWrapper.def"
#undef ldosSolverProblemWrapperTemplates
#undef ldosSolverProblemWrapperTemplatesL
    >;

  template <class... Args>
  inline ldosSolverProblemObject
  createLdosSolverProblemObject(dftfe::Int feOrder, Args &&...args)
  {
    switch (feOrder)
      {
#define ldosSolverProblemWrapperTemplates(T1) \
  case T1:                                    \
    return ldosSolverProblemObject(           \
      std::make_shared<ldosSolverProblem<T1>>(std::forward<Args>(args)...));
#define ldosSolverProblemWrapperTemplatesL(T1) \
  case T1:                                     \
    return ldosSolverProblemObject(            \
      std::make_shared<ldosSolverProblem<T1>>(std::forward<Args>(args)...));
#include "ldosSolverProblemWrapper.def"
#undef ldosSolverProblemWrapperTemplates
#undef ldosSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createLdosSolverProblemObject dispatch failed"};
      }
  }

  class ldosSolverProblemWrapperClass : public dealiiLinearSolverProblem
  {
  public:
    /// Constructor
    ldosSolverProblemWrapperClass(const dftfe::Int feOrder,
                                  const MPI_Comm  &mpi_comm_parent,
                                  const MPI_Comm  &mpi_comm_domain)
      : d_ldosSolverProblemObject(
          createLdosSolverProblemObject(feOrder,
                                        mpi_comm_parent,
                                        mpi_comm_domain))
    {}

    distributedCPUVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedCPUVec<double> & { return t->getX(); },
        d_ldosSolverProblemObject);
    }

    void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x)
    {
      std::visit([&](auto &t) { t->vmult(Ax, x); }, d_ldosSolverProblemObject);
    }

    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_ldosSolverProblemObject);
    }

    void
    precondition_Jacobi(distributedCPUVec<double>       &dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const
    {
      std::visit([&](
                   auto const &t) { t->precondition_Jacobi(dst, src, omega); },
                 d_ldosSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); }, d_ldosSolverProblemObject);
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
                 d_ldosSolverProblemObject);
    }

    template <typename... Args>
    void
    init(Args &&...args)
    {
      std::visit([&](auto &t) { t->init(std::forward<Args>(args)...); },
                 d_ldosSolverProblemObject);
    }

    const distributedCPUVec<double> &
    getDlocMassVector() const
    {
      return std::visit(
        [](auto const &t) -> const distributedCPUVec<double> & {
          return t->getDlocMassVector();
        },
        d_ldosSolverProblemObject);
    }

    double
    computeDlocIntegral(const distributedCPUVec<double> &phi) const
    {
      return std::visit(
        [&](auto const &t) { return t->computeDlocIntegral(phi); },
        d_ldosSolverProblemObject);
    }

    double
    getTotalDOS() const
    {
      return std::visit([](auto const &t) { return t->getTotalDOS(); },
                        d_ldosSolverProblemObject);
    }

  private:
    ldosSolverProblemObject d_ldosSolverProblemObject;
  };

#ifdef DFTFE_WITH_DEVICE
  using ldosSolverProblemDeviceObject = std::variant<
#  define ldosSolverProblemWrapperTemplates(T1) \
    std::shared_ptr<ldosSolverProblemDevice<T1>>,
#  define ldosSolverProblemWrapperTemplatesL(T1) \
    std::shared_ptr<ldosSolverProblemDevice<T1>>
#  include "ldosSolverProblemWrapper.def"
#  undef ldosSolverProblemWrapperTemplates
#  undef ldosSolverProblemWrapperTemplatesL
    >;

  template <class... Args>
  inline ldosSolverProblemDeviceObject
  createLdosSolverProblemDeviceObject(dftfe::Int feOrder, Args &&...args)
  {
    switch (feOrder)
      {
#  define ldosSolverProblemWrapperTemplates(T1)        \
    case T1:                                           \
      return ldosSolverProblemDeviceObject{            \
        std::make_shared<ldosSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...)};
#  define ldosSolverProblemWrapperTemplatesL(T1)       \
    case T1:                                           \
      return ldosSolverProblemDeviceObject{            \
        std::make_shared<ldosSolverProblemDevice<T1>>( \
          std::forward<Args>(args)...)};
#  include "ldosSolverProblemWrapper.def"
#  undef ldosSolverProblemWrapperTemplates
#  undef ldosSolverProblemWrapperTemplatesL
        default:
          throw std::logic_error{
            "createLdosSolverProblemDeviceObject dispatch failed"};
      }
  }

  class ldosSolverProblemDeviceWrapperClass : public linearSolverProblemDevice
  {
  public:
    /// Constructor
    ldosSolverProblemDeviceWrapperClass(const dftfe::Int feOrder,
                                        const MPI_Comm  &mpi_comm_parent,
                                        const MPI_Comm  &mpi_comm_domain)
      : d_ldosSolverProblemObject(
          createLdosSolverProblemDeviceObject(feOrder,
                                              mpi_comm_parent,
                                              mpi_comm_domain))
    {}

    distributedDeviceVec<double> &
    getX()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & { return t->getX(); },
        d_ldosSolverProblemObject);
    }

    distributedDeviceVec<double> &
    getPreconditioner()
    {
      return std::visit(
        [](auto &t) -> distributedDeviceVec<double> & {
          return t->getPreconditioner();
        },
        d_ldosSolverProblemObject);
    }

    void
    computeAX(distributedDeviceVec<double> &dst,
              distributedDeviceVec<double> &src)
    {
      std::visit([&](auto &t) { t->computeAX(dst, src); },
                 d_ldosSolverProblemObject);
    }

    void
    computeRhs(distributedCPUVec<double> &rhs)
    {
      std::visit([&](auto &t) { t->computeRhs(rhs); },
                 d_ldosSolverProblemObject);
    }

    void
    setX()
    {
      std::visit([](auto &t) { t->setX(); }, d_ldosSolverProblemObject);
    }

    void
    distributeX()
    {
      std::visit([](auto &t) { t->distributeX(); }, d_ldosSolverProblemObject);
    }

    void
    copyXfromDeviceToHost()
    {
      std::visit([](auto &t) { t->copyXfromDeviceToHost(); },
                 d_ldosSolverProblemObject);
    }

    template <typename... Args>
    void
    reinit(Args &&...args)
    {
      std::visit([&](auto &t) { t->reinit(std::forward<Args>(args)...); },
                 d_ldosSolverProblemObject);
    }

    template <typename... Args>
    void
    init(Args &&...args)
    {
      std::visit([&](auto &t) { t->init(std::forward<Args>(args)...); },
                 d_ldosSolverProblemObject);
    }

    void
    setBLASWrapperPtr(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                        dftfe::utils::MemorySpace::DEVICE>> blasWrapperPtr)
    {
      std::visit([&](auto &t) { t->setBLASWrapperPtr(blasWrapperPtr); },
                 d_ldosSolverProblemObject);
    }

    double
    computeDlocIntegral(const distributedCPUVec<double> &phi) const
    {
      return std::visit(
        [&](auto const &t) { return t->computeDlocIntegral(phi); },
        d_ldosSolverProblemObject);
    }

    double
    getTotalDOS() const
    {
      return std::visit([](auto const &t) { return t->getTotalDOS(); },
                        d_ldosSolverProblemObject);
    }

  private:
    ldosSolverProblemDeviceObject d_ldosSolverProblemObject;
  };

#endif // DFTFE_WITH_DEVICE

} // namespace dftfe
#endif
