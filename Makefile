CMAKE ?= cmake
BUILD_DIR ?= build-standalone
JOBS ?= 2

.PHONY: all clean test cmake-build
all:
	$(CMAKE) -S . -B $(BUILD_DIR) -DPMI_WITH_LLAMA=OFF -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)
	cp $(BUILD_DIR)/pmi pmi

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

cmake-build:
	$(CMAKE) -S . -B build -DPMI_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build build -j$(JOBS)

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean
	rm -f pmi
