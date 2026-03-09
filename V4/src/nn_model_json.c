#include "nn_model_json.h"
#include "nn_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NN_JSON_MAX_FILE_BYTES 16384

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static const char* find_key(const char* text, const char* key) {
    return strstr(text, key);
}

static int parse_int_after_key(const char* text, const char* key, int* out) {
    const char* p = find_key(text, key);
    char* endptr = NULL;
    long v;
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p = skip_ws(p + 1);
    v = strtol(p, &endptr, 10);
    if (endptr == p) return 0;
    *out = (int)v;
    return 1;
}

static int parse_float_after_key(const char* text, const char* key, nn_scalar_t* out) {
    const char* p = find_key(text, key);
    char* endptr = NULL;
    float v;
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p = skip_ws(p + 1);
    v = strtof(p, &endptr);
    if (endptr == p) return 0;
    *out = (nn_scalar_t)v;
    return 1;
}

static int parse_string_after_key(const char* text, const char* key, char* out, int out_sz) {
    const char* p = find_key(text, key);
    const char* q;
    int n;
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p = skip_ws(p + 1);
    if (*p != '"') return 0;
    p++;
    q = strchr(p, '"');
    if (!q) return 0;
    n = (int)(q - p);
    if (n <= 0 || n >= out_sz) return 0;
    memcpy(out, p, (size_t)n);
    out[n] = '\0';
    return 1;
}

static nn_activation_t parse_activation(const char* s) {
    if (strcmp(s, "linear") == 0) return NN_ACT_LINEAR;
    if (strcmp(s, "tanh") == 0) return NN_ACT_TANH;
    if (strcmp(s, "sigmoid") == 0) return NN_ACT_SIGMOID;
    if (strcmp(s, "relu") == 0) return NN_ACT_RELU;
    return NN_ACT_LINEAR;
}

static nn_init_t parse_init(const char* s) {
    if (strcmp(s, "uniform_sym") == 0) return NN_INIT_UNIFORM_SYM;
    if (strcmp(s, "xavier") == 0) return NN_INIT_XAVIER;
    if (strcmp(s, "he") == 0) return NN_INIT_HE;
    return NN_INIT_XAVIER;
}

static nn_loss_t parse_loss(const char* s) {
    if (strcmp(s, "mse") == 0) return NN_LOSS_MSE;
    return NN_LOSS_MSE;
}

static int parse_layers(const char* text, nn_model_config_t* cfg) {
    const char* p = find_key(text, "\"layers\"");
    const char* arr_end;
    int layer_count = 0;
    if (!p) return 0;

    p = strchr(p, '[');
    if (!p) return 0;
    arr_end = strchr(p, ']');
    if (!arr_end) return 0;
    p++;

    while (p < arr_end) {
        const char* obj_s = strchr(p, '{');
        const char* obj_e;
        const char* key_pos;
        const char* colon;
        int out_dim = 0;
        char act_s[32];
        int n;

        if (!obj_s || obj_s >= arr_end) break;
        obj_e = strchr(obj_s, '}');
        if (!obj_e || obj_e > arr_end) return 0;
        if (layer_count >= NN_CAP_MAX_LAYERS) return 0;

        key_pos = strstr(obj_s, "\"out_dim\"");
        if (!key_pos || key_pos > obj_e) return 0;
        colon = strchr(key_pos, ':');
        if (!colon || colon > obj_e) return 0;
        out_dim = (int)strtol(skip_ws(colon + 1), NULL, 10);
        if (out_dim <= 0) return 0;

        key_pos = strstr(obj_s, "\"activation\"");
        if (!key_pos || key_pos > obj_e) return 0;
        colon = strchr(key_pos, ':');
        if (!colon || colon > obj_e) return 0;
        colon = skip_ws(colon + 1);
        if (*colon != '"') return 0;
        colon++;
        n = 0;
        while (colon + n < obj_e && colon[n] != '"' && n < (int)sizeof(act_s) - 1) n++;
        if (colon + n >= obj_e) return 0;
        memcpy(act_s, colon, (size_t)n);
        act_s[n] = '\0';

        cfg->layers[layer_count].out_dim = out_dim;
        cfg->layers[layer_count].act = parse_activation(act_s);
        layer_count++;

        p = obj_e + 1;
    }

    cfg->num_layers = layer_count;
    return layer_count > 0;
}

void nn_model_config_default(nn_model_config_t* cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    cfg->max_layers = 4;
    cfg->max_width = 16;
    cfg->input_dim = 2;

    cfg->num_layers = 2;
    cfg->layers[0].out_dim = 4;
    cfg->layers[0].act = NN_ACT_TANH;
    cfg->layers[1].out_dim = 1;
    cfg->layers[1].act = NN_ACT_SIGMOID;

    cfg->init = NN_INIT_XAVIER;
    cfg->seed = 123u;

    cfg->loss = NN_LOSS_MSE;
    cfg->opt.lr = (nn_scalar_t)0.1f;
    cfg->epochs = 3000;
    cfg->print_every = 300;
    cfg->use_spec1 = 1;
}

nn_status_t nn_model_config_load_json(const char* path, nn_model_config_t* cfg) {
    FILE* f;
    char text[NN_JSON_MAX_FILE_BYTES + 1];
    size_t nread;
    char s_buf[64];
    int i_buf;
    if (!path || !cfg) return NN_ERR_BADARG;

    nn_model_config_default(cfg);

    f = fopen(path, "rb");
    if (!f) return NN_ERR_BADARG;

    nread = fread(text, 1, NN_JSON_MAX_FILE_BYTES, f);
    if (ferror(f)) {
        fclose(f);
        return NN_ERR_BADARG;
    }
    if (!feof(f)) {
        fclose(f);
        return NN_ERR_UNSUPPORTED;
    }
    fclose(f);
    text[nread] = '\0';

    if (parse_int_after_key(text, "\"max_layers\"", &i_buf)) cfg->max_layers = i_buf;
    if (parse_int_after_key(text, "\"max_width\"", &i_buf)) cfg->max_width = i_buf;
    if (parse_int_after_key(text, "\"input_dim\"", &i_buf)) cfg->input_dim = i_buf;
    if (parse_int_after_key(text, "\"seed\"", &i_buf)) cfg->seed = (uint32_t)i_buf;
    if (parse_int_after_key(text, "\"epochs\"", &i_buf)) cfg->epochs = i_buf;
    if (parse_int_after_key(text, "\"print_every\"", &i_buf)) cfg->print_every = i_buf;
    if (parse_float_after_key(text, "\"lr\"", &cfg->opt.lr)) {}

    if (parse_string_after_key(text, "\"init\"", s_buf, (int)sizeof(s_buf))) cfg->init = parse_init(s_buf);
    if (parse_string_after_key(text, "\"loss\"", s_buf, (int)sizeof(s_buf))) cfg->loss = parse_loss(s_buf);
    if (parse_string_after_key(text, "\"mode\"", s_buf, (int)sizeof(s_buf))) {
        cfg->use_spec1 = (strcmp(s_buf, "spec1") == 0) ? 1 : 0;
    }

    if (!parse_layers(text, cfg)) return NN_ERR_BADARG;

    if (cfg->max_layers <= 0 || cfg->max_layers > NN_CAP_MAX_LAYERS) return NN_ERR_BADARG;
    if (cfg->max_width <= 0 || cfg->max_width > NN_CAP_MAX_WIDTH) return NN_ERR_BADARG;
    if (cfg->input_dim <= 0 || cfg->input_dim > cfg->max_width) return NN_ERR_BADARG;
    if (cfg->num_layers <= 0 || cfg->num_layers > cfg->max_layers) return NN_ERR_BADARG;
    if (cfg->epochs <= 0) return NN_ERR_BADARG;

    for (int i = 0; i < cfg->num_layers; i++) {
        if (cfg->layers[i].out_dim <= 0 || cfg->layers[i].out_dim > cfg->max_width) return NN_ERR_BADARG;
    }

    return NN_OK;
}

nn_status_t nn_model_build_from_config(const nn_model_config_t* cfg, nn_net_t* net) {
    int in_dim;
    if (!cfg || !net) return NN_ERR_BADARG;

    if (nn_set_global_constraints(cfg->max_layers, cfg->max_width) != NN_OK) return NN_ERR_BADARG;
    if (nn_net_create(net) != NN_OK) return NN_ERR_BADARG;

    in_dim = cfg->input_dim;
    for (int i = 0; i < cfg->num_layers; i++) {
        nn_status_t st = nn_net_add_dense(net, in_dim, cfg->layers[i].out_dim, cfg->layers[i].act);
        if (st != NN_OK) return st;
        in_dim = cfg->layers[i].out_dim;
    }

    nn_net_init(net, cfg->init, cfg->seed);
    return NN_OK;
}
