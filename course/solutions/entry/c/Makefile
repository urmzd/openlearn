# Optional standalone C archive for local exploration. Course checks compile
# each module's C suite as its own process; no runtime loads this archive.
# SANITIZE=1 adds ASan and UBSan.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra
CPPFLAGS += -I../contracts/c/include
ifeq ($(SANITIZE),1)
CFLAGS  += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address,undefined
endif

SRCS := $(sort $(wildcard src/*/*.c))
OBJS := $(patsubst src/%.c,build/obj/%.o,$(SRCS))

.PHONY: all clean
all: build/libtinyllm.a

build/obj/%.o: src/%.c $(wildcard ../contracts/c/include/*.h ../contracts/c/include/tinyllm/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/libtinyllm.a: $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -rf build
