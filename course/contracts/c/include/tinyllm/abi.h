/* tinyllm/abi.h (rt.02): shared status codes, error reporting, and allocator
 * hooks for standalone C modules. Rules in c/ABI.md.
 *
 * tl_set_last_error, tl_alloc, and tl_free are the seams other units use:
 * a unit reaches abi.c only through the names in this header.
 *
 * module: rt.02
 * chapter: content/courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md */
#ifndef TINYLLM_ABI_H
#define TINYLLM_ABI_H

#include <stddef.h>
#include <stdint.h>


/* Version of this C support interface. */
#define TL_ABI_VERSION 1
uint32_t tl_abi_version(void); /* returns TL_ABI_VERSION */

/* Status codes. tl_status is an int32_t rather than a C enum type, so callers
 * can safely inspect unknown values from newer modules. Codes are positive;
 * 0 is success. */
typedef int32_t tl_status;
enum {
    TL_OK = 0,
    TL_EINVAL = 1,       /* bad argument: NULL pointer, negative dim, bad stride */
    TL_ENOMEM = 2,       /* the allocator hook returned NULL */
    TL_ESHAPE = 3,       /* dims inconsistent with each other */
    TL_EDTYPE = 4,       /* dtype not accepted by this function */
    TL_EFULL = 5,        /* a fixed-capacity structure is full */
    TL_ENOTFOUND = 6,    /* lookup miss */
    TL_EFORMAT = 7,      /* bad bytes: magic, CRC, version */
    TL_EBUSY = 8,        /* object in use */
    TL_EUNSUPPORTED = 9, /* not built yet (stub unit) or not supported */
    TL_EIO = 10          /* the OS reported an I/O error */
};

/* A static, human-readable name for s ("TL_OK", "TL_EINVAL", ...).
 * Unknown values give "TL_UNKNOWN". Never NULL. */
const char *tl_status_str(tl_status s);

/* The error slot. One per thread (_Thread_local in abi.c).
 * Every function that returns a status other than TL_OK, or NULL where a
 * pointer was expected, first sets the slot to a message naming the
 * function and the reason. tl_last_error never returns NULL; before any
 * error it returns "". The text stays valid until the next tl_ call on the
 * same thread. tl_set_last_error copies msg (NULL means "") and truncates it
 * to TL_LAST_ERROR_CAP - 1 bytes. */
#define TL_LAST_ERROR_CAP 256
const char *tl_last_error(void);
void tl_set_last_error(const char *msg);

/* Element types. Like tl_status, an int32_t with named constants. */
typedef int32_t tl_dtype;
enum {
    TL_F32 = 0,
    TL_F16 = 1,
    TL_BF16 = 2,
    TL_F8_E4M3 = 3,
    TL_F8_E5M2 = 4,
    TL_I8 = 5,
    TL_U8 = 6,
    TL_I32 = 7,
    TL_Q4_G = 8
};

/* The allocator hook. Every allocation inside the library goes through it,
 * which is how tests count allocations and inject failures (fail after n).
 * alloc returns memory aligned to `align` (a power of two) or NULL; free
 * accepts NULL. `user` is passed back unchanged. */
typedef struct {
    void *(*alloc)(void *user, size_t n, size_t align);
    void (*free)(void *user, void *p);
    void *user;
} tl_allocator;

/* Installs a copy of *a. NULL restores the default (malloc-based) hook.
 * TL_EINVAL when a->alloc or a->free is NULL. Change the hook only while no
 * object created through the old one is alive. */
tl_status tl_set_allocator(const tl_allocator *a);

/* Allocate and free through the installed hook. align is a power of two.
 * tl_alloc returns NULL and sets the error slot when the hook fails, when
 * n is 0, or when align is not a power of two; the caller then returns
 * TL_ENOMEM (or TL_EINVAL for a bad align). tl_free(NULL) does nothing. */
void *tl_alloc(size_t n, size_t align);
void tl_free(void *p);


#endif /* TINYLLM_ABI_H */
