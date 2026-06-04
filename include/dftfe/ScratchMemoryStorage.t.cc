#include <Exceptions.h>

namespace dftfe
{
  namespace utils
  {
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::View::View(
      Handle *owner) noexcept
      : d_owner(owner)
    {}

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::View::rebind(
      Handle *owner) noexcept
    {
      d_owner = owner;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::Handle &
    ScratchMemoryStorage<ValueType, memorySpace>::View::activeOwner()
    {
      throwException<InvalidArgument>(
        d_owner != nullptr,
        "Attempted to access an unbound scratch memory view.");
      d_owner->validateInUse();
      return *d_owner;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const typename ScratchMemoryStorage<ValueType, memorySpace>::Handle &
    ScratchMemoryStorage<ValueType, memorySpace>::View::activeOwner() const
    {
      throwException<InvalidArgument>(
        d_owner != nullptr,
        "Attempted to access an unbound scratch memory view.");
      d_owner->validateInUse();
      return *d_owner;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    std::size_t
    ScratchMemoryStorage<ValueType, memorySpace>::View::size() const
    {
      return activeOwner().d_size;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    std::size_t
    ScratchMemoryStorage<ValueType, memorySpace>::View::capacity() const
    {
      return activeOwner().storage().size();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::pointer
    ScratchMemoryStorage<ValueType, memorySpace>::View::data()
    {
      return activeOwner().storage().data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const ValueType *
    ScratchMemoryStorage<ValueType, memorySpace>::View::data() const
    {
      return activeOwner().storage().data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::iterator
    ScratchMemoryStorage<ValueType, memorySpace>::View::begin()
    {
      return data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::const_iterator
    ScratchMemoryStorage<ValueType, memorySpace>::View::begin() const
    {
      return data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::iterator
    ScratchMemoryStorage<ValueType, memorySpace>::View::end()
    {
      return begin() + size();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::const_iterator
    ScratchMemoryStorage<ValueType, memorySpace>::View::end() const
    {
      return begin() + size();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::reference
    ScratchMemoryStorage<ValueType, memorySpace>::View::operator[](
      const std::size_t i)
    {
      throwException<InvalidArgument>(
        i < size(), "Attempted to access a scratch memory view out of bounds.");
      return data()[i];
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View::const_reference
    ScratchMemoryStorage<ValueType, memorySpace>::View::operator[](
      const std::size_t i) const
    {
      throwException<InvalidArgument>(
        i < size(), "Attempted to access a scratch memory view out of bounds.");
      return data()[i];
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::View::setValue(
      const ValueType val)
    {
      Handle &owner = activeOwner();
      dftfe::utils::MemoryManager<ValueType, memorySpace>::set(
        owner.d_size, owner.storage().data(), val);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::View::swap(View &rhs)
    {
      if (this == &rhs)
        return;

      Handle &lhsOwner = activeOwner();
      Handle &rhsOwner = rhs.activeOwner();

      if (&lhsOwner == &rhsOwner)
        return;

      std::swap(lhsOwner.d_state, rhsOwner.d_state);
      std::swap(lhsOwner.d_buffer, rhsOwner.d_buffer);
      std::swap(lhsOwner.d_size, rhsOwner.d_size);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::View::operator bool()
      const noexcept
    {
      return d_owner != nullptr && d_owner->d_buffer != nullptr;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::Handle() noexcept
      : d_view(this)
    {}

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::Handle(
      std::shared_ptr<PoolState>             state,
      MemoryStorage<ValueType, memorySpace> *buffer,
      const std::size_t                      size) noexcept
      : d_state(std::move(state))
      , d_buffer(buffer)
      , d_size(size)
      , d_view(this)
    {}

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::~Handle()
    {
      releaseNoThrow();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::Handle(
      Handle &&other) noexcept
      : d_state(std::move(other.d_state))
      , d_buffer(other.d_buffer)
      , d_size(other.d_size)
      , d_view(this)
    {
      other.d_buffer = nullptr;
      other.d_size   = 0;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::Handle &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator=(
      Handle &&other) noexcept
    {
      if (&other != this)
        {
          releaseNoThrow();
          d_state  = std::move(other.d_state);
          d_buffer = other.d_buffer;
          d_size   = other.d_size;
          d_view.rebind(this);
          other.d_buffer = nullptr;
          other.d_size   = 0;
        }

      return *this;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::get() &
    {
      validateInUse();
      return d_view;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const typename ScratchMemoryStorage<ValueType, memorySpace>::View &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::get() const &
    {
      validateInUse();
      return d_view;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View *
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator->() &
    {
      return &get();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const typename ScratchMemoryStorage<ValueType, memorySpace>::View *
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator->() const &
    {
      return &get();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::View &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator*() &
    {
      return get();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const typename ScratchMemoryStorage<ValueType, memorySpace>::View &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator*() const &
    {
      return get();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::validateInUse() const
    {
      throwException<InvalidArgument>(
        d_buffer != nullptr,
        "Attempted to access an empty scratch memory handle.");
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace> &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::storage() &
    {
      validateInUse();
      return *d_buffer;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const MemoryStorage<ValueType, memorySpace> &
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::storage() const &
    {
      validateInUse();
      return *d_buffer;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::operator bool()
      const noexcept
    {
      return d_buffer != nullptr;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::Handle::release()
    {
      throwException<InvalidArgument>(
        d_buffer != nullptr,
        "Attempted to release an empty scratch memory handle.");
      throwException<InvalidArgument>(
        d_state != nullptr && d_state->release(d_buffer),
        "Attempted to release a scratch buffer that is not in use or is not "
        "owned by this ScratchMemoryStorage.");
      d_state.reset();
      d_buffer = nullptr;
      d_size   = 0;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType,
                         memorySpace>::Handle::releaseNoThrow() noexcept
    {
      if (d_state != nullptr && d_buffer != nullptr)
        d_state->release(d_buffer);

      d_state.reset();
      d_buffer = nullptr;
      d_size   = 0;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ScratchMemoryStorage<ValueType, memorySpace>::ScratchMemoryStorage()
      : d_state(std::make_shared<PoolState>())
    {}

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename ScratchMemoryStorage<ValueType, memorySpace>::Handle
    ScratchMemoryStorage<ValueType, memorySpace>::acquire(
      const std::size_t size,
      const ValueType   initVal)
    {
      const std::size_t invalidIndex     = d_state->pool.size();
      std::size_t       bestFitIndex     = invalidIndex;
      std::size_t       largestFreeIndex = invalidIndex;

      for (std::size_t i = 0; i < d_state->pool.size(); ++i)
        {
          PoolEntry &entry = d_state->pool[i];

          if (entry.inUse)
            continue;

          const std::size_t entrySize = entry.storage.size();

          if (largestFreeIndex == invalidIndex ||
              d_state->pool[largestFreeIndex].storage.size() < entrySize)
            largestFreeIndex = i;

          if (entrySize >= size &&
              (bestFitIndex == invalidIndex ||
               entrySize < d_state->pool[bestFitIndex].storage.size()))
            bestFitIndex = i;
        }

      if (bestFitIndex != invalidIndex)
        {
          PoolEntry &entry = d_state->pool[bestFitIndex];
          entry.storage.setValue(initVal);
          entry.inUse = true;
          return Handle(d_state, &entry.storage, size);
        }

      if (largestFreeIndex != invalidIndex)
        {
          PoolEntry &entry = d_state->pool[largestFreeIndex];
          entry.storage.resize(size, initVal);
          entry.inUse = true;
          return Handle(d_state, &entry.storage, size);
        }

      d_state->pool.emplace_back();
      PoolEntry &entry = d_state->pool.back();
      entry.storage.resize(size, initVal);
      entry.inUse = true;
      return Handle(d_state, &entry.storage, size);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    bool
    ScratchMemoryStorage<ValueType, memorySpace>::PoolState::release(
      const MemoryStorage<ValueType, memorySpace> *buffer) noexcept
    {
      for (PoolEntry &entry : pool)
        {
          if (&entry.storage == buffer)
            {
              if (!entry.inUse)
                return false;

              entry.inUse = false;
              return true;
            }
        }

      return false;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    bool
    ScratchMemoryStorage<ValueType, memorySpace>::PoolState::hasBuffersInUse()
      const noexcept
    {
      for (const PoolEntry &entry : pool)
        if (entry.inUse)
          return true;

      return false;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    ScratchMemoryStorage<ValueType, memorySpace>::clear()
    {
      throwException<InvalidArgument>(
        !d_state->hasBuffersInUse(),
        "Attempted to clear ScratchMemoryStorage while scratch buffers are "
        "still in use.");
      std::deque<PoolEntry>().swap(d_state->pool);
    }
  } // namespace utils
} // namespace dftfe
