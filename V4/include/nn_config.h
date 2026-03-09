#ifndef NN_CONFIG_H // include guard
#define NN_CONFIG_H

#include "nn_types.h" // import nn_status_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

#ifndef NN_CAP_MAX_LAYERS // default max layers if not defined at compile time
#define NN_CAP_MAX_LAYERS 8
#endif

#ifndef NN_CAP_MAX_WIDTH // default max width if not defined at compile time
#define NN_CAP_MAX_WIDTH 64
#endif

extern int nn_g_max_layers; // global max layers constraint (default NN_CAP_MAX_LAYERS)
extern int nn_g_max_width; // global max width constraint (default NN_CAP_MAX_WIDTH)

// PROTOTYPE:

// sets global constraints for max layers and max width. Must be called before creating any networks or layers.
// Returns NN_ERR_BADARG if constraints are invalid.
nn_status_t nn_set_global_constraints(int max_layers, int max_width);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard