#pragma once
#include <omp.h>

#include <thread>

#include "cinn/runtime/cinn_runtime.h"

extern "C" {

int max_concurrency();

/**
 * @brief The callback function to execute a parallel lambda
 * @param task_id the task id of the function.
 * @param num_task The Number of tasks to launch. If 0, it means to launch
 *           with all available threads.
 * @param datas The closure datas.
 */
typedef int (*FCINNParallelLambda)(int task_id, int num_task, void* datas);

/**
 * @brief Backend function for running parallel jobs.
 *
 * @param flambda The parallel function to be launched.
 * @param datas The closure datas.
 * @param num_task The Number of tasks to launch. If 0, it means to launch
 *           with all available threads.
 *
 * @return 0 when no error is thrown, -1 when failure happens
 */
int cinn_backend_parallel_launch(FCINNParallelLambda flambda, void* datas, int num_task);

}  // extern "C"
