CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS ?=

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:.cpp=.o)

.PHONY: all clean test
all: pmi

.PHONY: cmake-build
cmake-build:
	cmake -S . -B build -DPMI_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build build --target pmi -j2

pmi: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: pmi
	python3 tests/make_fixture.py
	./pmi self-test tests/fixtures/toy.gguf

clean:
	rm -f $(OBJ) pmi
