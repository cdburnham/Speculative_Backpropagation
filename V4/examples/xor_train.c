#include <stdio.h>
#include "xor_json_common.h"

int main(void) {
    int ok = 0;
    if (xor_run_from_json("data/models/xor_small_tanh.json", 1, &ok) != 0) {
        return 1;
    }

    if (!ok) {
        printf("XOR training quality gate failed (loss did not reduce enough).\n");
        return 2;
    }

    return 0;
}
