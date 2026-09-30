
namespace dftfe
{
  namespace utils
  {
    template <class T>
    void
    throwException(bool condition, std::string msg)
    {
      if (!condition)
        throw T(msg);
    }
  } // end of namespace utils

} // end of namespace dftfe
