CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=

LIB_SRC=src/nn_config.c src/nn_math.c src/nn_rng.c src/nn_activation.c src/nn_loss.c src/nn_layer_dense.c src/nn_network.c src/nn_train.c src/nn_memory.c src/nn_model_json.c
LIB_OBJ=$(LIB_SRC:.c=.o)
EX_COMMON_OBJ=examples/xor_json_common.o

ALL_BINS=xor_train xor_json_from_file xor_json_suite xor_compare_csv

all: $(ALL_BINS)

xor_train: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_train.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

xor_json_from_file: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_json_from_file.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

xor_json_suite: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_json_suite.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


xor_compare_csv: $(LIB_OBJ) examples/xor_compare_csv.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

examples/xor_compare_csv.o: examples/xor_compare_csv.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_train.o examples/xor_json_from_file.o examples/xor_json_suite.o examples/xor_compare_csv.o $(ALL_BINS)
