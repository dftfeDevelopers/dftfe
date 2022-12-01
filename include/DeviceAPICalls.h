#ifdef DFTFE_WITH_DEVICE

#  ifndef dftfeDeviceAPICalls_H
#    define dftfeDeviceAPICalls_H

#    include <TypeConfig.h>
#  include <DeviceTypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    void
    deviceGetDeviceCount(int *count);

    void
    deviceGetDevice(int *deviceId);

    void
    deviceSetDevice(int deviceId);

    void
    deviceMalloc(void **devPtr, size_type size);

    void
    deviceMemset(void *devPtr,int value, size_type count);

    /**
     * @brief
     * @param devPtr
     * @param value
     * @param size
     */
    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, size_type size);

    void
    deviceFree(void *devPtr);

    void
    hostPinnedMalloc(void **hostPtr, size_type size);

    void
    hostPinnedFree(void *hostPtr);

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2H(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2D(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyH2D(void *dst, const void *src, size_type count);

    /**
     * @brief HOST-DEVICE synchronization
     */
    void
    deviceSynchronize();

    /**
     * @brief This function sets the attributes of a function specified via func. 
     * The parameter func must be a pointer to a function that executes on the device. 
     */
    void
    deviceFuncSetAttribute(const void* func, deviceFuncAttribute attr, int  value);


  } // namespace utils
} // namespace dftfe

#  endif // dftfeDeviceAPICalls_H
#endif   // DFTFE_WITH_DEVICE
