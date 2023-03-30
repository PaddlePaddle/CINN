
#include "cinn/ir/ir_verify.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir.h"
#include <iostream>
#include "cinn/ir/tensor.h"
#include "cinn/lang/placeholder.h"

#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_util.h"

#include "cinn/backends/nvrtc/nvrtc_util.h"

#include "cinn/runtime/cuda/cuda_module.h"

#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>


#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

namespace cinn {
namespace ir {

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}


TEST(IrManul, basic) {
    int reduce_block = 128;
    int flatten_block = 8;
    
    std::vector<int> reduce_range;
    std::vector<int> flatten_range;

    for( int i = 0; i < reduce_block; ++i )
    {
        reduce_range.push_back(i);
    }

    for( int i = 0; i < flatten_block; ++i )
    {
        flatten_range.push_back(i);
    }

    std::string name_blockx = "blockIdx.x";
    std::string name_threadx = "xid";
    std::string index_name = "index";
    Var block_x_var( name_blockx, type_of<int>() );
    Var thread_x_var( name_threadx, type_of<int>() );
    // Var index_var( index_name, type_of<int>()); 

    Var block_id( "blockIdx.x", type_of<int>() );
    Var flatten_id( "xid", type_of<int>() );
    Var r_id( "rid", type_of<int>() );
    Expr expr_flatten( flatten_block);    
    Expr expr_reduce( reduce_block);

    auto index_var =( block_id * expr_flatten + flatten_id ) * expr_reduce + r_id;
        
    auto load_index = LoadIndex::Make( index_var, reduce_range, flatten_range, reduce_block, flatten_block);    

    Var input_var("input", type_of<float>() );

    auto block_load = BlockLoad::Make( input_var, load_index );

    auto reduce_max = ReduceMax::Make( block_load, 1);

    Var output_var("output", type_of<float>() );

    auto store_idx = block_id * expr_flatten + flatten_id;

    auto store_index = LoadIndex::Make( store_idx, reduce_range, flatten_range, reduce_block, flatten_block);
    auto block_store = BlockStore::Make( output_var, store_index, reduce_max );

    cinn::ir::IrPrinter printer(std::cout);
    // std::cerr << std::endl;
    // printer.Print(index_var);
    // std::cerr << std::endl;
    
    // printer.Print( block_load );
    // std::cerr << "======= block " << std::endl;
    // printer.Print( reduce_max );

    // std::cerr << "=== reduce max" << std::endl;

    // printer.Print( block_store );



    // split the range

    int num_warp = 8;
    int num_thread_per_warp = 32;
    int element_per_thread = 4;
    
    Var threadidx("threadIdx.x", type_of<int>());
    Var index_i("i", type_of<int>() );
    Var index_j("j", type_of<int>() );
    Expr expr_warp( num_warp);
    Expr expr_thread_per_warp( num_thread_per_warp );
    Expr expr_element_per_thread( element_per_thread );

    auto warp_id = threadidx / expr_thread_per_warp;

    auto xid = warp_id + index_i * expr_warp;
    auto inner_id = threadidx % expr_thread_per_warp;
    auto inner_index = xid * expr_element_per_thread * expr_thread_per_warp + inner_id + index_j * expr_thread_per_warp;

    // printer.Print( inner_index );
    
    // warp reduce
    auto warp_round = flatten_block / num_warp;
    auto thread_round = reduce_block / num_thread_per_warp;

    std::string temp_name = "tmp";
    Var temp_var( temp_name, type_of<float>() );
    auto temp_out = LocalTemp::Make( temp_var, {warp_round, thread_round});

    Var loop_var("i");

    cinn::lang::Placeholder<float> C("d_in", std::vector<int>{{10, 10}});
    cinn::lang::Placeholder<float> T("tmp", std::vector<int>{{1,4}});
    //Placeholder<float> A("A", std::vector<int>{{10}});
    //Var input( "input", type_of<float>( ));
    Var loop_var_j("j");

    auto t_load = ir::Load::Make( ir::Tensor(C), { inner_index });

    Expr body = Store::Make( ir::Tensor(T), t_load, {Expr(loop_var), Expr(loop_var_j)}); 
                            
    body      = ir::Block::Make({body});
    
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    //printer.Print( load_inner_for );

    std::cerr << "------------------------------\n";

    body =  ir::Block::Make( {load_inner_for});

    auto load_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    
    printer.Print( load_outer_for );

    std::cerr << std::endl;

    Expr inf(-100000.0);
    std::string temp_max_name = "tmp_max";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round});

    std::string name1 = "max1";
    cinn::ir::Var max_t(name1, type_of<float>());

    auto max_var = cinn::ir::Let::Make( max_t, inf);

    t_load = ir::Load::Make( ir::Tensor(T), { Expr(loop_var), Expr(loop_var_j) });


    auto new_max = ir::Max::Make( max_t, t_load);
    auto out_max = ir::Let::Make( max_t, new_max, false);
  


    body =  ir::Block::Make( {out_max });
    load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

    auto warp_call = Call::Make( Float(32), "warpReduceMax<128>", {max_t}, {}, ir::CallType::Extern );

    //auto warp_res = ir::Let::Make( max_t, warp_call, false);

    cinn::lang::Placeholder<float> T_MAX("tmp_max", std::vector<int>{{1, 4}});
    auto max_store = Store::Make( ir::Tensor(T_MAX), warp_call, {Expr(loop_var)}); 


    body =  ir::Block::Make( {max_var,  load_inner_for, max_store});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    // printer.Print( load_outer_for );

    std::cerr << "\n";


    t_load = ir::Load::Make( ir::Tensor(T_MAX), { Expr(loop_var) });
    cinn::lang::Placeholder<float> OUT("d_in", std::vector<int>{{10}});
    auto out_store = Store::Make( ir::Tensor(OUT), t_load, { Expr( block_id * Expr(8) + warp_id ) });

    
    body =  ir::Block::Make( {out_store });
    auto out_store_for = ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    printer.Print( out_store_for );

    std::cerr << std::endl;

    auto cond = ir::EQ::Make( threadidx % Expr(32), Expr(0) );
    auto filter = ir::IfThenElse::Make( cond, out_store_for, Expr());

    //printer.Print( filter );

    
    auto target = common::DefaultNVGPUTarget();
    backends::CodeGenCUDA_Dev cu_dev(target);

    std::string head = R"ROC( 

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceMax(float sum) {
    if (blockSize >= 32)sum = max(sum, __shfl_down_sync(0xffffffff, sum, 16) ); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum = max(sum, __shfl_down_sync(0xffffffff, sum, 8) );// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum = max(sum, __shfl_down_sync(0xffffffff, sum, 4) );// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum = max(sum, __shfl_down_sync(0xffffffff, sum, 2) );// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum = max(sum, __shfl_down_sync(0xffffffff, sum, 1) );// 0-1, 2-3, 4-5, etc.
    return sum;
}

extern "C" {

__global__ void softmax_test(float *d_in,float *d_out ) {
)ROC";

    cu_dev.ss_ << head << std::endl;
    cu_dev.Print( temp_out );
    cu_dev.ss_  << std::endl;  
    cu_dev.Print( load_outer_for );
    cu_dev.ss_ << std::endl;
    cu_dev.Print( temp_max_out );
    cu_dev.ss_ << "\n";
    cu_dev.Print( max_outer_for );
    cu_dev.ss_ << ";" << std::endl;
    cu_dev.Print( filter );
    cu_dev.ss_  << ";" << std::endl;

    cu_dev.ss_ << "} \n }" << std::endl;

    std::cerr << cu_dev.ss_.str() << std::endl;


    auto source_code = cu_dev.ss_.str();

  backends::nvrtc::Compiler compiler;

  auto ptx = compiler(source_code);

  std::cerr << "source code" << source_code << std::endl;

  const int N= 128 * 12 * 128 *128;
  float *a=(float *)malloc(N*sizeof(float));
  float *d_a;
  cudaMalloc((void **)&d_a,N*sizeof(float));

  const int num_warps = 8;
  const int block_num = 128 * 12 * 128 / num_warps;
  const int NUM_PER_BLOCK = N / block_num;
  const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
  float *out=(float *)malloc(N *sizeof(float));
  float *d_out;

  int M = 128 * 12 * 128;
  cudaMalloc((void **)&d_out, M *sizeof(float));
  float *res=(float *)malloc( M *sizeof(float));
  
  srand(0);
  for(int i=0;i<N;i++){
      a[i]= rand() % 100 / 100;
  }

  for(int i=0;i< 128 * 12 * 128;i++){
      float cur=-100000000;
      for(int j=0;j<128;j++){
          if( cur < a[ i * 128 + j ] )
          {
              cur = a[i*128 + j ];
          }
      }
    //   float sum = 0;
    //   float temp[128];
    //   for( int j = 0; j < 128; ++j )
    //   {
    //       temp[j] = exp(a[i*128 + j ] - cur);
    //       sum += temp[j];
    //   }
      
      res[i] = cur;
    //   for( int j = 0; j < 128; ++j )
    //   {
    //       res[ i * 128 + j ] = temp[j] / sum;
    //   }
  }
  std::cerr << "before copy" << std::endl;
  cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

  dim3 Grid( block_num, 1, 1);
  dim3 Block( THREAD_PER_BLOCK, 1, 1);

  void* args[] = {&d_a, &d_out };

  cinn::runtime::cuda::CUDAModule cuda_module(ptx, cinn::runtime::cuda::CUDAModule::Kind::CUBIN);
  
  for ( int i = 0; i < 1000; ++i)
  {
    cuda_module.LaunchKernel(0, "softmax_test", Grid, Block, args);
  }

  std::cerr << "before copy" << std::endl;
  cudaMemcpy(out,d_out, M *sizeof(float),cudaMemcpyDeviceToHost);

  if(check(out,res,M))printf("the ans is right\n");
  else{
      printf("the ans is wrong\n");
      for(int i=0;i< M;i++){
          // printf("%lf ",out[i]);
        if( (out[i] - res[i] ) > 1e-5 ){
              std::cout << i << "\t" << out[i] << "\t" << res[i] << std::endl;
              break;
          }
      }
      printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_out);

}

}
}  // namespace cinn::ir


