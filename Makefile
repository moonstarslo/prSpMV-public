.PHONY: prep spmv run_batch run all

SRC_DIR = src
MAT_DIR = matrix_py
RUN_DIR = run_py

all: prep spmv run_batch

prep:
	$(MAKE) -C $(MAT_DIR) prepro

spmv: 
	$(MAKE) -C $(SRC_DIR) all

run_batch:
	$(MAKE) -C $(RUN_DIR) run_batch
