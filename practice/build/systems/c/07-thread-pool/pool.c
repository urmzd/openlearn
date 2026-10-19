/* pool.c - the part you write. */
#include "pool.h"

#include <pthread.h>
#include <stdlib.h>

typedef struct {
  TaskFn fn;
  void *arg;
} Task;

struct ThreadPool {
  pthread_t *threads;
  size_t n_threads;

  Task *queue;      /* ring buffer */
  size_t cap;
  size_t head, tail, count;

  size_t active;    /* tasks currently being run by a worker */
  size_t completed;
  int shutting_down;

  pthread_mutex_t lock;
  pthread_cond_t not_empty; /* a worker waits here for work */
  pthread_cond_t not_full;  /* a producer waits here for space */
  pthread_cond_t idle;      /* pool_wait waits here for quiet */
};

static void *worker_main(void *arg) {
  ThreadPool *p = arg;

  for (;;) {
    pthread_mutex_lock(&p->lock);

    /* while, not if. A condition variable may wake spuriously, and even
     * without that, several workers can be woken for one signal and only the
     * first to reacquire the lock gets the task. Re-test the predicate. */
    while (p->count == 0 && !p->shutting_down) {
      pthread_cond_wait(&p->not_empty, &p->lock);
    }

    /* Drain before exiting: shutdown means "no new work", not "abandon what
     * was accepted". A producer whose submit returned 0 is entitled to have
     * that task run. */
    if (p->count == 0 && p->shutting_down) {
      pthread_mutex_unlock(&p->lock);
      return NULL;
    }

    Task t = p->queue[p->head];
    p->head = (p->head + 1) % p->cap;
    p->count--;
    p->active++;

    /* A slot just freed up. */
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->lock);

    /* Run the task OUTSIDE the lock. Holding it here would serialise the whole
     * pool down to one effective worker, which is the single most common way
     * to write a thread pool that is slower than a plain loop. */
    t.fn(t.arg);

    pthread_mutex_lock(&p->lock);
    p->active--;
    p->completed++;
    /* Quiet means nothing queued AND nothing running. Signalling on an empty
     * queue alone would let pool_wait return while tasks are still executing. */
    if (p->count == 0 && p->active == 0) pthread_cond_broadcast(&p->idle);
    pthread_mutex_unlock(&p->lock);
  }
}

ThreadPool *pool_new(size_t n_threads, size_t queue_cap) {
  if (n_threads == 0 || queue_cap == 0) return NULL;

  ThreadPool *p = calloc(1, sizeof *p);
  if (!p) return NULL;

  p->queue = malloc(queue_cap * sizeof(Task));
  p->threads = malloc(n_threads * sizeof(pthread_t));
  if (!p->queue || !p->threads) {
    free(p->queue);
    free(p->threads);
    free(p);
    return NULL;
  }
  p->cap = queue_cap;

  pthread_mutex_init(&p->lock, NULL);
  pthread_cond_init(&p->not_empty, NULL);
  pthread_cond_init(&p->not_full, NULL);
  pthread_cond_init(&p->idle, NULL);

  for (size_t i = 0; i < n_threads; i++) {
    if (pthread_create(&p->threads[i], NULL, worker_main, p) != 0) {
      /* Start what we can; tear down cleanly if even one thread started. */
      p->n_threads = i;
      if (i == 0) {
        pthread_mutex_destroy(&p->lock);
        pthread_cond_destroy(&p->not_empty);
        pthread_cond_destroy(&p->not_full);
        pthread_cond_destroy(&p->idle);
        free(p->queue);
        free(p->threads);
        free(p);
        return NULL;
      }
      return p;
    }
  }
  p->n_threads = n_threads;
  return p;
}

int pool_submit(ThreadPool *p, TaskFn fn, void *arg) {
  pthread_mutex_lock(&p->lock);

  while (p->count == p->cap && !p->shutting_down) {
    pthread_cond_wait(&p->not_full, &p->lock);
  }

  if (p->shutting_down) {
    pthread_mutex_unlock(&p->lock);
    return -1;
  }

  p->queue[p->tail] = (Task){fn, arg};
  p->tail = (p->tail + 1) % p->cap;
  p->count++;

  pthread_cond_signal(&p->not_empty);
  pthread_mutex_unlock(&p->lock);
  return 0;
}

void pool_wait(ThreadPool *p) {
  pthread_mutex_lock(&p->lock);
  while (p->count > 0 || p->active > 0) {
    pthread_cond_wait(&p->idle, &p->lock);
  }
  pthread_mutex_unlock(&p->lock);
}

void pool_destroy(ThreadPool *p) {
  if (!p) return;

  pthread_mutex_lock(&p->lock);
  p->shutting_down = 1;
  /* Broadcast, not signal: every worker is parked on not_empty and every one
   * of them has to wake up to notice the shutdown and exit. Signalling would
   * wake one and leave the rest blocked forever, so the join below would
   * never return. */
  pthread_cond_broadcast(&p->not_empty);
  pthread_cond_broadcast(&p->not_full);
  pthread_mutex_unlock(&p->lock);

  for (size_t i = 0; i < p->n_threads; i++) pthread_join(p->threads[i], NULL);

  pthread_mutex_destroy(&p->lock);
  pthread_cond_destroy(&p->not_empty);
  pthread_cond_destroy(&p->not_full);
  pthread_cond_destroy(&p->idle);
  free(p->queue);
  free(p->threads);
  free(p);
}

size_t pool_completed(ThreadPool *p) {
  size_t n;
  pthread_mutex_lock(&p->lock);
  n = p->completed;
  pthread_mutex_unlock(&p->lock);
  return n;
}
