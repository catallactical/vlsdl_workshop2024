/**
 * @author Alberto M. Esmoris Pena
 * @brief Useful macros for the VL3D++ software.
 */

// Start if to prevent multiple inclusions
#ifndef VL3DPP_MACROS_HPP_
#define VL3DPP_MACROS_HPP_




// Macro to flag methods as used (even if they are not) with GCC and CLang
#if defined(__GNUC__) || defined(__clang__)
#define VL3DPP_USED_ __attribute__((used))
#else
#define VL3DPP_USED_
#endif




// End if to prevent multiple inclusions
#endif