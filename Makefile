CC = g++
CC_FLAGS = -std=c++11 -O2 -Wall `pkg-config --cflags --libs opencv4` 

all: convex

convex: Convex.cpp
	$(CC) $(CC_FLAGS) Convex.cpp -o convex

clean: 
	rm convex
