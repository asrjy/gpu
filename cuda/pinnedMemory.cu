#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct timer {
  struct timespec start_time, end_time;
};

void start_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->start_time);
}

void stop_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->end_time);
}

double time_diff(struct timer *t) {
  double diff = (t->end_time.tv_sec - t->start_time.tv_sec) +
                (t->end_time.tv_nsec - t->start_time.tv_nsec) / 1000000000.0;
  return diff;
}

struct timer t;

int32_t main() {
  int32_t N = 100000000;

  cudaDeviceSynchronize();

  int32_t *x = (int32_t *)malloc(N * sizeof(int32_t));
  for (int32_t i = 0; i < N; ++i) {
    x[i] = rand();
  }

  start_timer(&t);
  int32_t *x_h = (int32_t *)malloc(N * sizeof(int32_t));
  memcpy(x_h, x, N * sizeof(int32_t));
  stop_timer(&t);
  printf("CPU to CPU memcpy time: %f\n", time_diff(&t));

  int32_t *x_d;
  start_timer(&t);
  cudaMalloc(&x_d, N);
  cudaMemcpy(x_d, x, N, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("CPU to GPU copy time: %f\n", time_diff(&t));

  int32_t *x_pinned;
  cudaHostAlloc((void**)&x_pinned, N * sizeof(int32_t), cudaHostAllocDefault);
  memcpy(x_pinned, x, N * sizeof(int32_t));  // Copy to pinned memorywhat 
  start_timer(&t);
  cudaMemcpy(x_d, x_pinned, N * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("CPU to GPU (pinned) copy time: %f\n", time_diff(&t));
}