CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all clean

all: build/sus2-sh

build:
	mkdir -p build

build/sus2-sh: src/sus2_sh.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf build
