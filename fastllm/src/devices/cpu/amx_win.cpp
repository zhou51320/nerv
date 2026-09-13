#include "fastllm.h"
#include "utils.h"

// nerv: Intel AMX (tile) requires Linux syscall plumbing to enable the
// XFEATURE permissions and is a no-go on Windows.  Keep the two externally
// referenced symbols linkable; the linear kernel falls back to AVX512/AVX2.
namespace fastllm {
    void InitAMX() {
    }

    bool LinearBFloat16BFloat16_AMX_Kernel(uint16_t *, uint16_t *, float *,
                                           float *, int, int, int, int, int) {
        return false;
    }
}