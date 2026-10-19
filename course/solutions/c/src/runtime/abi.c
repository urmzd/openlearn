/* Shared C test support: ABI version, status names, error slot, and
 * allocator hooks. Contract: tinyllm/abi.h. Rules: c/ABI.md.
 *
 * Every other C unit reaches this file only through the names in abi.h:
 * tl_set_last_error to report a failure, tl_alloc and tl_free to get memory.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm/abi.h"

/* ---- The error slot -----------------------------------------------------
 * Every function in the runtime reports failures through tl_set_last_error,
 * so the setter and its getter are the first thing the ABI needs.
 *
 * _Thread_local gives each thread its own copy: an error on one thread
 * never overwrites the message another thread is about to read. */
static _Thread_local char err_slot[TL_LAST_ERROR_CAP];

void tl_set_last_error(const char *msg) {
    if (msg == NULL) msg = "";
    size_t n = strlen(msg);
    if (n > TL_LAST_ERROR_CAP - 1) n = TL_LAST_ERROR_CAP - 1;
    memcpy(err_slot, msg, n); /* copy: the caller's buffer may die right after */
    err_slot[n] = '\0';
}

const char *tl_last_error(void) { return err_slot; }
/* ---- End of the given block ------------------------------------------- */

/* The default hook: the C library allocator, through aligned_alloc. */
static void *default_alloc(void *user, size_t n, size_t align) {
    (void)user;
    /* aligned_alloc wants align >= sizeof(void *) on some libcs and a size
     * that is a multiple of align (C11 7.22.3.1), so round both up. */
    if (align < sizeof(void *)) align = sizeof(void *);
    if (n > SIZE_MAX - (align - 1)) return NULL;
    size_t size = (n + align - 1) & ~(align - 1);
    return aligned_alloc(align, size);
}

static void default_free(void *user, void *p) {
    (void)user;
    free(p);
}

/* The installed hook: a copy, never a pointer to the caller's struct. */
static tl_allocator hook = {default_alloc, default_free, NULL};

uint32_t tl_abi_version(void) {
    return TL_ABI_VERSION;
}

const char *tl_status_str(tl_status s) {
    /* A switch, not an array indexed by s: s comes from the caller and may be
     * any int32_t, including negative values and codes from a newer ABI. */
    switch (s) {
    case TL_OK: return "TL_OK";
    case TL_EINVAL: return "TL_EINVAL";
    case TL_ENOMEM: return "TL_ENOMEM";
    case TL_ESHAPE: return "TL_ESHAPE";
    case TL_EDTYPE: return "TL_EDTYPE";
    case TL_EFULL: return "TL_EFULL";
    case TL_ENOTFOUND: return "TL_ENOTFOUND";
    case TL_EFORMAT: return "TL_EFORMAT";
    case TL_EBUSY: return "TL_EBUSY";
    case TL_EUNSUPPORTED: return "TL_EUNSUPPORTED";
    case TL_EIO: return "TL_EIO";
    default: return "TL_UNKNOWN";
    }
}

tl_status tl_set_allocator(const tl_allocator *a) {
    if (a == NULL) {
        hook.alloc = default_alloc;
        hook.free = default_free;
        hook.user = NULL;
        return TL_OK;
    }
    if (a->alloc == NULL || a->free == NULL) {
        tl_set_last_error("tl_set_allocator: alloc and free must both be set");
        return TL_EINVAL;
    }
    hook = *a; /* copy the struct: the caller's may be a stack local */
    return TL_OK;
}

void *tl_alloc(size_t n, size_t align) {
    if (n == 0) {
        tl_set_last_error("tl_alloc: size 0");
        return NULL;
    }
    if (align == 0 || (align & (align - 1)) != 0) {
        tl_set_last_error("tl_alloc: align must be a power of two");
        return NULL;
    }
    void *p = hook.alloc(hook.user, n, align);
    if (p == NULL) tl_set_last_error("tl_alloc: the allocator hook returned NULL");
    return p;
}

void tl_free(void *p) {
    if (p == NULL) return;
    hook.free(hook.user, p);
}
