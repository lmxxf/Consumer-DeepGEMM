#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"

#include <string>

bool cutlass_sm120_probe_compiled() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return true;
#else
  return false;
#endif
}

std::string cutlass_sm120_probe_arch() {
#if defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return "sm_121a";
#elif defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  return "sm_120a";
#else
  return "unsupported";
#endif
}
