# lang.02 reference Makefile for the two-file program main.c + greet.c.
CFLAGS ?= -std=c11 -Wall -Wextra -O2
OBJS := main.o greet.o

.PHONY: all clean
all: greet

# Link: the program depends on every object.
greet: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Compile: one object per source. Each object also depends on the header it
# includes, so editing greet.h rebuilds both.
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
main.o: greet.h
greet.o: greet.h

clean:
	rm -f greet $(OBJS)
