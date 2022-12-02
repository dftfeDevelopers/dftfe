#ifdef DFTFE_WITH_DEVICE

#  ifndef dftfeDeviceAPICalls_H
#    define dftfeDeviceAPICalls_H

#    include <TypeConfig.h>
#  include <DeviceTypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    deviceError_t
    deviceGetDeviceCount(int *count);

    deviceError_t
    deviceGetDevice(int *deviceId);

    deviceError_t
    deviceSetDevice(int deviceId);

    deviceError_t
    deviceMalloc(void **devPtr, size_type size);

    deviceError_t
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

    deviceError_t
    deviceFree(void *devPtr);

    deviceError_t
    hostPinnedMalloc(void **hostPtr, size_type size);

    deviceError_t
    hostPinnedFree(void *hostPtr);

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, size_type count);

    /**
     * @brief Copy 2D array from device to host
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2H_2D(void* dst, size_type dpitch, const void* src, size_type spitch, size_type width, size_type height);

    /**
     * @brief Copy 2D array from device to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2D_2D(void* dst, size_type dpitch, const void* src, size_type spitch, size_type width, size_type height);

    /**
     * @brief Copy 2D array from host to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyH2D_2D(void* dst, size_type dpitch, const void* src, size_type spitch, size_type width, size_type height);    

    /**
     * @brief HOST-DEVICE synchronization
     */
    deviceError_t
    deviceSynchronize();
  } // namespace utils
} // namespace dftfe

#  endif // dftfeDeviceAPICalls_H
#endif   // DFTFE_WITH_DEVICE
