# C compiler used for all compile/link steps.
CC=gcc

# Compiler flags:
# -O2        : optimization level 2
# -Wall      : enable common warnings
# -Wextra    : enable extra warnings
# -Iinclude  : add ./include to header search path
CFLAGS=-O2 -Wall -Wextra -Iinclude

# All core library source files.
LIB_SRC=src/nn_config.c src/nn_math.c src/nn_rng.c src/nn_activation.c src/nn_loss.c src/nn_layer_dense.c src/nn_network.c src/nn_train.c src/nn_memory.c src/nn_model_json.c

# Convert each entry in LIB_SRC from .c to .o.
LIB_OBJ=$(LIB_SRC:.c=.o)

# Object file shared by multiple example binaries.
EX_COMMON_OBJ=examples/xor_json_common.o

# Executables produced by this Makefile.
ALL_BINS=xor_train xor_json_from_file xor_json_suite xor_compare_csv

# Default target: build all binaries.
all: $(ALL_BINS)

# Build xor_train from core objects, shared example object and its main object.
xor_train: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_train.o
	# $@ = target name (xor_train), $^ = all prerequisites
	$(CC) $(CFLAGS) -o $@ $^

# Build xor_json_from_file with its main object.
xor_json_from_file: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_json_from_file.o
	# Link all needed objects into the final executable.
	$(CC) $(CFLAGS) -o $@ $^

# Build xor_json_suite with its main object.
xor_json_suite: $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_json_suite.o
	# Link all needed objects into the final executable.
	$(CC) $(CFLAGS) -o $@ $^

# Build xor_compare_csv from core objects and its main object.
xor_compare_csv: $(LIB_OBJ) examples/xor_compare_csv.o
	# Link all needed objects into the final executable.
	$(CC) $(CFLAGS) -o $@ $^

# Explicit compile rule for xor_compare_csv source.
examples/xor_compare_csv.o: examples/xor_compare_csv.c
	# -c compiles to object file only; $< = first prerequisite
	$(CC) $(CFLAGS) -c $< -o $@

# Generic pattern rule: compile any .c file into matching .o file.
%.o: %.c
	# Reused for library and example object compilation.
	$(CC) $(CFLAGS) -c $< -o $@

# Remove generated object files and binaries.
clean:
	# -f ignores missing files; clean and idempotent.
	rm -f $(LIB_OBJ) $(EX_COMMON_OBJ) examples/xor_train.o examples/xor_json_from_file.o examples/xor_json_suite.o examples/xor_compare_csv.o $(ALL_BINS)
