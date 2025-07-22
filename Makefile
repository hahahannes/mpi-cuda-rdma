compile:
	cd cpu && make
	cd cuda/cuda-aware && make
	cd cuda/staged && make
