#include <stdio.h>
#include "xor_json_common.h"

int main(void) {
    const char* models[] = {
        "data/models/xor_small_tanh.json",
        "data/models/xor_wide_tanh.json",
        "data/models/xor_deep_tanh.json"
    };
    int pass_count = 0;
    int n_models = (int)(sizeof(models) / sizeof(models[0]));

    for (int i = 0; i < n_models; i++) {
        int ok = 0;
        int rc;
        printf("\n--- Running model %d/%d ---\n", i + 1, n_models);
        rc = xor_run_from_json(models[i], 0, &ok);
        if (rc == 0 && ok) {
            pass_count++;
            printf("Result: PASS (%s)\n", models[i]);
        } else {
            printf("Result: FAIL (%s)\n", models[i]);
        }
    }

    printf("\nSuite result: %d/%d passed\n", pass_count, n_models);
    return (pass_count == n_models) ? 0 : 2;
}
