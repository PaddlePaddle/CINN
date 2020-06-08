#pragma once
//! \file This file defines some C APIs to trigger CBLAS methods.
#include "cinn/runtime/cinn_runtime.h"

#ifdef CINN_WITH_MKL_CBLAS
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

// define some C APIs
extern "C" {

/**
 * \brief Do GEMM on buffer A and B and write result to buffer C.
 * We pass the \param M, \param N, \param K although the shape can retrieve from cinn_buffer_t because the size of a
 * matrix not equals the shape of a buffer it is stored.
 * @param alpha The scaling factor of the product of A and B
 * @param M Number of the rows of A
 * @param N the number of the columns in both B and C
 * @param K the number of columns of A
 * @param ta whether to transpose A
 * @param tb whether to transpose B
 * @param lda The size of the first dimension of A
 * @param ldb The size of the first dimension of B
 * @param ldc The size of the first dimension of C
 * @param beta The scaling factor of C
 * @param A The matrix A
 * @param B The matrix B
 * @param C The output matrix
 */
void cinn_cpu_mkl_gemm_fp32(float alpha,
                            int M,
                            int N,
                            int K,
                            bool ta,
                            bool tb,
                            int lda,
                            int ldb,
                            int ldc,
                            float beta,
                            cinn_buffer_t* A,
                            cinn_buffer_t* B,
                            cinn_buffer_t* C);
}  // extern "C"
