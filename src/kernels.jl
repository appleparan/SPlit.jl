"""
Kernel types selecting the discrepancy that support points minimize.
"""

"""
    SplitKernel

Abstract supertype of kernels usable with [`SupportPointSplitter`](@ref).
Each kernel defines the discrepancy between the support-point set and the
data distribution that the optimizer minimizes.
"""
abstract type SplitKernel end

"""
    EnergyKernel()

The kernel `k(x, y) = −‖x − y‖`, whose maximum mean discrepancy between two
samples is the energy distance. Support points under this kernel are those of
Mak & Joseph (2018), optimized by their closed-form
majorization–minimization update.
"""
struct EnergyKernel <: SplitKernel end
