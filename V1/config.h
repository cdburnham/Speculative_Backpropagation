#include "fx16.h"

#ifndef CONFIG_H
#define CONFIG_H

// Global compile‑time limits (tune for FPGA resources)
#define MAX_LAYERS        32
#define MAX_NEURONS       1024
#define MAX_BATCH_SIZE    1
#define MAX_PARAMS        (MAX_LAYERS * MAX_NEURONS * 16)

#endif