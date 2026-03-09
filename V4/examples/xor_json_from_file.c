#include <stdio.h>
#include "xor_json_common.h"

int main(int argc, char** argv) {
    int ok = 0;
    const char* path = "data/models/xor_small_tanh.json";

    if (argc > 1) path = argv[1];
    if (xor_run_from_json(path, 1, &ok) != 0) return 1;

    if (!ok) {
        printf("Model converged poorly for %s\n", path);
        return 2;
    }

    return 0;
}
