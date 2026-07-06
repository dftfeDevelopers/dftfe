#ifndef dftfeScratchMemoryStorage_h
#define dftfeScratchMemoryStorage_h

#include <dftfe/MemoryStorage.h>

#include <cstddef>
#include <deque>
#include <memory>
#include <utility>

namespace dftfe
{
  namespace utils
  {
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    class ScratchMemoryStorage
    {
    private:
      struct PoolEntry;
      struct PoolState;

    public:
      class Handle;

      class View
      {
      public:
        typedef ValueType        value_type;
        typedef ValueType       *pointer;
        typedef ValueType       &reference;
        typedef const ValueType &const_reference;
        typedef ValueType       *iterator;
        typedef const ValueType *const_iterator;

        View() = default;

        std::size_t
        size() const;

        std::size_t
        capacity() const;

        pointer
        data();

        const ValueType *
        data() const;

        iterator
        begin();

        const_iterator
        begin() const;

        iterator
        end();

        const_iterator
        end() const;

        reference
        operator[](std::size_t i);

        const_reference
        operator[](std::size_t i) const;

        void
        setValue(ValueType val);

        void
        swap(View &rhs);

        explicit operator bool() const noexcept;

      private:
        friend class Handle;

        explicit View(Handle *owner) noexcept;

        void
        rebind(Handle *owner) noexcept;

        Handle &
        activeOwner();

        const Handle &
        activeOwner() const;

        Handle *d_owner = nullptr;
      };

      class Handle
      {
      public:
        Handle() noexcept;

        ~Handle();

        Handle(const Handle &) = delete;

        Handle &
        operator=(const Handle &) = delete;

        Handle(Handle &&other) noexcept;

        Handle &
        operator=(Handle &&other) noexcept;

        View &
        get() &;

        const View &
        get() const &;

        View &
        get() && = delete;

        const View &
        get() const && = delete;

        View *
        operator->() &;

        const View *
        operator->() const &;

        View *
        operator->() && = delete;

        const View *
        operator->() const && = delete;

        View &
        operator*() &;

        const View &
        operator*() const &;

        View &
        operator*() && = delete;

        const View &
        operator*() const && = delete;

        explicit operator bool() const noexcept;

        void
        release();

      private:
        friend class ScratchMemoryStorage<ValueType, memorySpace>;
        friend class View;

        Handle(std::shared_ptr<PoolState>             state,
               MemoryStorage<ValueType, memorySpace> *buffer,
               std::size_t                            size) noexcept;

        void
        validateInUse() const;

        MemoryStorage<ValueType, memorySpace> &
        storage() &;

        const MemoryStorage<ValueType, memorySpace> &
        storage() const &;

        void
        releaseNoThrow() noexcept;

        std::shared_ptr<PoolState>             d_state;
        MemoryStorage<ValueType, memorySpace> *d_buffer = nullptr;
        std::size_t                            d_size   = 0;
        View                                   d_view;
      };

      ScratchMemoryStorage();

      ~ScratchMemoryStorage() = default;

      ScratchMemoryStorage(const ScratchMemoryStorage &) = delete;

      ScratchMemoryStorage &
      operator=(const ScratchMemoryStorage &) = delete;

      ScratchMemoryStorage(ScratchMemoryStorage &&) noexcept = default;

      ScratchMemoryStorage &
      operator=(ScratchMemoryStorage &&) noexcept = default;

      /**
       * @brief Acquire a scratch buffer with at least the requested number of
       * entries.
       *
       * A reused buffer can be larger than the requested size so that the pool
       * can avoid reallocating memory. The returned handle exposes a logical
       * view whose size matches the requested size even when the underlying
       * reused buffer is larger. Regardless of whether the buffer is reused or
       * resized, its contents are initialized to @p initVal. The returned
       * handle releases the buffer automatically on destruction.
       */
      [[nodiscard]] Handle
      acquire(std::size_t size, ValueType initVal = ValueType());

      /**
       * @brief Release all pooled buffers.
       */
      void
      clear();

    private:
      struct PoolEntry
      {
        MemoryStorage<ValueType, memorySpace> storage;
        bool                                  inUse = false;
      };

      struct PoolState
      {
        bool
        release(const MemoryStorage<ValueType, memorySpace> *buffer) noexcept;

        bool
        hasBuffersInUse() const noexcept;

        std::deque<PoolEntry> pool;
      };

      std::shared_ptr<PoolState> d_state;
    };
  } // namespace utils
} // namespace dftfe

#include <dftfe/ScratchMemoryStorage.t.cc>

#endif // dftfeScratchMemoryStorage_h
