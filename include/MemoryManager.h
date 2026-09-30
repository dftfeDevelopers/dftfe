#ifndef dftfeMemoryManager_h
#define dftfeMemoryManager_h

#include <TypeConfig.h>
#include <MemorySpaceType.h>
#include <vector>

namespace dftfe
{
  namespace utils
  {
    //
    // MemoryManager
    //
    template <typename ValueType, MemorySpace memorySpace>
    class MemoryManager
    {
    public:
      static void
      allocate(std::size_t size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(std::size_t size, ValueType *ptr, ValueType val);
    };

    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::HOST>
    {
    public:
      static void
      allocate(std::size_t size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(std::size_t size, ValueType *ptr, ValueType val);
    };

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::HOST_PINNED>
    {
    public:
      static void
      allocate(std::size_t size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(std::size_t size, ValueType *ptr, ValueType val);
    };


    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::DEVICE>
    {
    public:
      static void
      allocate(std::size_t size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(std::size_t size, ValueType *ptr, ValueType val);
    };
#endif // DFTFE_WITH_DEVICE
  }    // namespace utils

} // namespace dftfe

#endif
