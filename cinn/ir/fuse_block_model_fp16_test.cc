
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
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/hlir/framework/pass.h"

#include "cinn/frontend/net_builder.h"

#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>

#include "cinn/lang/placeholder.h"

#include <cmath>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

namespace cinn {
namespace ir {

bool check( common::float16 *out, common::float16 *res,int n){
    for(int i=0;i<n;i++){
        if( abs( static_cast<float>(out[i]) -  static_cast<float>(res[i]) ) > 1e-5 )
            return false;
    }
    return true;
}

int reduce_axis( const std::vector<int>& first, const std::vector<int>& second)
{     

    if( first[0] == 1 && second[0] != 1)
    {
        return 0;
    }
    if( first[1] != 1 && second[1] == 1)
    {
        return 1;
    }
    throw std::runtime_error("reduce_axis: error");
} 


struct InputNode
{
    InputNode() {}
    InputNode( std::string n, cinn::lang::Placeholder<float> *p, std::vector<int> dim)
        : name(n), in_ptr(p), in_dim(dim) {}
    std::string name;
    cinn::lang::Placeholder<float>* in_ptr;
    std::vector<int> in_dim;
};

void process_reduce_max( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, backends::CodeGenCUDA_Dev* code_dev)
{
    std::string in_name;
    for (auto& inlink : node->inlinks_in_order(true)) {
        auto* innode = inlink->source()->safe_as<hlir::framework::NodeData>();
        if (innode) {
            in_name =  innode->id();
        }
    }

    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    InputNode& input_node = input_map->at(in_name);


    Var loop_var("i");
    Var loop_var_j("j");
    Expr inf(-100000.0);
    std::string temp_max_name = "tmp_max";
    Var temp_max_var( temp_max_name, type_of<float>() );
    
    int warp_round = 4;
    int thread_round = 4;
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round});

    std::string name1 = "max1";
    cinn::ir::Var max_t(name1, type_of<float>());

    auto max_var = cinn::ir::Let::Make( max_t, inf);

    auto t_load = ir::Load::Make( ir::Tensor( *(input_node.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });


    auto new_max = ir::Max::Make( max_t, t_load);
    auto out_max = ir::Let::Make( max_t, new_max, false);
  
    cinn::ir::IrPrinter printer(std::cout);
    printer.Print( new_max );

    auto body =  ir::Block::Make( {out_max });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

    auto warp_call = Call::Make( Float(32), "warpReduceMax<128>", {max_t}, {}, ir::CallType::Extern );

    //auto warp_res = ir::Let::Make( max_t, warp_call, false);

    // for test here, memory leak
    cinn::lang::Placeholder<float>* T_MAX = new cinn::lang::Placeholder<float>("tmp_max", std::vector<int>{{1, 4}});
    auto max_store = Store::Make( ir::Tensor(*T_MAX), warp_call, {Expr(loop_var)}); 


    body =  ir::Block::Make( {max_var,  load_inner_for, max_store});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);  
   
    code_dev->Print( temp_max_out);
    code_dev->Print( max_outer_for);
    code_dev->ss_ << std::endl;



    (*input_map)[out_name] = InputNode( "reduce_max", T_MAX, {1, 1});

}


void process_sub( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, backends::CodeGenCUDA_Dev* code_dev)
{
    std::vector<std::string> vec_in_names;
    for (auto& inlink : node->inlinks_in_order(true)) {
        auto* innode = inlink->source()->safe_as<hlir::framework::NodeData>();
        if (innode) {
            vec_in_names.push_back( innode->id() );
        }
    }

    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    int broadcast_axis = reduce_axis( first_input.in_dim, second_input.in_dim);

    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = 4;
    int thread_round = 4;
    Expr inf(-100000.0);
    std::string temp_max_name = "sub_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    std::string name1 = "sub_tmp";
    cinn::ir::Var max_t(name1, type_of<float>());

    auto max_var = cinn::ir::Let::Make( max_t, inf);

    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    
    auto t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var) });
    

    auto out = ir::Sub::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* sub = new cinn::lang::Placeholder<float>("sub_tmp", std::vector<int>{{1, 4}});
    auto sub_store = Store::Make( ir::Tensor(*sub), out, {Expr(loop_var), Expr(loop_var_j)}); 


    auto body =  ir::Block::Make( { sub_store });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

 


    body =  ir::Block::Make( {load_inner_for});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

    code_dev->Print( temp_max_out);
    code_dev->Print( max_outer_for);
    code_dev->ss_ << std::endl;


    (*input_map)[out_name] = InputNode( "sub", sub, {1, 4});

}


void process_exp( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, backends::CodeGenCUDA_Dev* code_dev)
{
    std::string in_name;
    for (auto& inlink : node->inlinks_in_order(true)) {
        auto* innode = inlink->source()->safe_as<hlir::framework::NodeData>();
        if (innode) {
            in_name = innode->id();
        }
    }

    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    InputNode& first_input = input_map->at( in_name);

    Var loop_var("i");
    Var loop_var_j("j");
    Expr inf(-100000.0);
    int warp_round = 4;
    int thread_round = 4;
    std::string temp_max_name = "exp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    std::string name1 = "max1";
    cinn::ir::Var max_t(name1, type_of<float>());


    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    auto out = ir::Minus::Make( t_load);

    cinn::lang::Placeholder<float>* exp = new cinn::lang::Placeholder<float>("exp", std::vector<int>{{1, 4}});
    auto exp_store = Store::Make( ir::Tensor(*exp), out, {Expr(loop_var), Expr(loop_var_j)}); 


    auto body =  ir::Block::Make( { exp_store });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    body =  ir::Block::Make( {load_inner_for});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    
    
    code_dev->Print( temp_max_out);
    code_dev->Print( max_outer_for);
    code_dev->ss_ << std::endl;

    (*input_map)[out_name] = InputNode( "exp", exp, {1, 4});

}


void process_reduce_sum( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, backends::CodeGenCUDA_Dev* code_dev)
{
    std::string in_name;
    for (auto& inlink : node->inlinks_in_order(true)) {
        auto* innode = inlink->source()->safe_as<hlir::framework::NodeData>();
        if (innode) {
            in_name =  innode->id();
        }
    }

    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    InputNode& input_node = input_map->at(in_name);


    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = 4;
    int thread_round = 4;
    Expr zero(0.0);
    std::string temp_max_name = "tmp_sum";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round});

    std::string name1 = "sum1";
    cinn::ir::Var sum1(name1, type_of<float>());

    auto max_var = cinn::ir::Let::Make( sum1, zero);

    auto t_load = ir::Load::Make( ir::Tensor( *(input_node.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });


    auto new_sum = ir::Add::Make( sum1, t_load);
    auto out_sum = ir::Let::Make( sum1, new_sum, false);
  


    auto body =  ir::Block::Make( {out_sum });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

    auto warp_call = Call::Make( Float(32), "warpReduceSum<128>", {sum1}, {}, ir::CallType::Extern );

    //auto warp_res = ir::Let::Make( max_t, warp_call, false);

    cinn::lang::Placeholder<float>* T_SUM = new cinn::lang::Placeholder<float>("tmp_sum", std::vector<int>{{1, 4}});
    auto max_store = Store::Make( ir::Tensor(*T_SUM), warp_call, {Expr(loop_var)}); 


    body =  ir::Block::Make( {max_var,  load_inner_for, max_store});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    code_dev->Print( temp_max_out);
    code_dev->Print( max_outer_for);
    code_dev->ss_ << std::endl;

    (*input_map)[out_name] = InputNode( "reduce_sum", T_SUM, {1, 1});

}


void process_divide( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, backends::CodeGenCUDA_Dev* code_dev)
{
    std::vector<std::string> vec_in_names;
    for (auto& inlink : node->inlinks_in_order(true)) {
        auto* innode = inlink->source()->safe_as<hlir::framework::NodeData>();
        if (innode) {
            vec_in_names.push_back( innode->id() );
        }
    }

    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    int broadcast_axis = reduce_axis( first_input.in_dim, second_input.in_dim);

    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = 4;
    int thread_round = 4;
    Expr inf(-100000.0);
    std::string temp_max_name = "div_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    
    auto t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var) });
    

    auto out = ir::Div::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* div = new cinn::lang::Placeholder<float>("div_tmp", std::vector<int>{{1, 4}});
    auto sub_store = Store::Make( ir::Tensor(*div), out, {Expr(loop_var), Expr(loop_var_j)}); 


    auto body =  ir::Block::Make( { sub_store });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    

 


    body =  ir::Block::Make( {load_inner_for});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    
    code_dev->Print( temp_max_out);
    code_dev->Print( max_outer_for);
    code_dev->ss_ << std::endl;


    (*input_map)[out_name] = InputNode( "divide", div, {1, 4});

}

TEST(IrManul, basic) {

    frontend::NetBuilder net_builder("softmax");
    {
    auto A = net_builder.CreateInput(Float(32), {128, 12, 128, 128}, "A");    
    auto Max = net_builder.ReduceMax(A, {3}, true);    
    auto sub = net_builder.Subtract(A, Max);
    auto exp = net_builder.Exp( sub );
    auto sum = net_builder.ReduceSum( exp, {3}, true);    
    auto out = net_builder.Divide( exp, sum);    
    }

    auto program = net_builder.Build();
    auto target  = common::DefaultTarget();    

    auto graph = std::make_shared<hlir::framework::Graph>(program, target);    
   
    std::cerr << "len " << graph->fusion_groups.size() << std::endl;

    std::cerr << graph->DebugGroupedGraph() << std::endl;

    //auto group0 = graph->FusionGroupsToGroups()[0];

    auto topo_order = graph->topological_order();
    auto& nodes     = std::get<0>(topo_order);

    // add input data
    int reduce_block = 128;
    int flatten_block = 32;
    
    std::vector<int> reduce_range;
    std::vector<int> flatten_range;
    
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
    int element_per_thread = 8;
    
    Var threadidx("threadIdx.x", type_of<int>());
    Var index_i("i", type_of<int>() );
    Var index_j("j", type_of<int>() );
    Expr expr_warp( num_warp);
    Expr expr_thread_per_warp( num_thread_per_warp );
    Expr expr_element_per_thread( element_per_thread );

    auto warp_id = threadidx / expr_thread_per_warp;

    auto xid = warp_id * Expr(4) + index_i;
    auto inner_id = threadidx % expr_thread_per_warp;
    auto inner_index = block_id *Expr(4096) +  xid * Expr(4) * expr_thread_per_warp + inner_id + index_j * expr_thread_per_warp;

    // auto inner_index =   blockIdx.x * 1024 + (threadIdx.x / 32) * 128 + threadIdx.x % 32;
    // auto inner_index = block_id * Expr(1024) +  (threadidx / Expr(32) ) * Expr(128) +  threadidx % Expr(32) + index_j * expr_thread_per_warp;
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

    
    //printer.Print( load_outer_for );


    std::map<std::string, InputNode> map_input;
        
    backends::CodeGenCUDA_Dev cu_dev(target);

    
    std::string head = R"ROC( 

#include <cuda_fp16.h>


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

__global__ void softmax_test(half  *d_in, half  *d_out ) {
)ROC";

    cu_dev.ss_ << head << std::endl;
    cu_dev.Print( temp_out );
    cu_dev.ss_ << "\n";
    cu_dev.Print( load_outer_for);

    std::cerr << cu_dev.ss_.str() << std::endl;
    std::cerr << "=======" << std::endl;
    map_input["A"] = InputNode( "A", &T, {1, 4});

    for (auto* n : nodes) {
        
        auto node = n->safe_as<hlir::framework::Node>();
        if (!node || node->op() == nullptr) {
            continue;
        }
        std::cerr << node->op()->name << std::endl;

        if( node->op()->name == "reduce_max")
        {
            process_reduce_max( &map_input, node, &cu_dev);
        }else if ( node->op()->name == "subtract" )
        {
            process_sub( &map_input, node, &cu_dev);
        }else if ( node->op()->name == "exp" )
        {
            process_exp( &map_input, node, &cu_dev);
        }else if ( node->op()->name == "reduce_sum" )
        {
            process_reduce_sum( &map_input, node, &cu_dev);
        }else if ( node->op()->name == "divide" )
        {
            process_divide( &map_input, node, &cu_dev);
        }

    }



    // name var_4 is output
    auto var_out = map_input.at( "var_4");

    t_load = ir::Load::Make( ir::Tensor( *(var_out.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    //t_load = ir::Load::Make( ir::Tensor( T), { Expr(loop_var), Expr(loop_var_j) });
    cinn::lang::Placeholder<float> OUT("d_out", std::vector<int>{{10}});

    // Expr num1(128);
    // Expr num2( 32 );
    // Expr block_step( 1024);
    // Expr parallel_size(4);
    // auto index_var2 = block_x_var * block_step + thread_x_var / num2 * num1 + thread_x_var % num2;
    auto out_store = Store::Make( ir::Tensor(OUT), t_load, { Expr( inner_index ) });


    //auto out_store = Store::Make( ir::Tensor(OUT), t_load, { Expr( index_var2 + loop_var_j * num2 ) });

    body =  ir::Block::Make( {out_store });
    load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    
    
    body =  ir::Block::Make( {load_inner_for});

    
    auto out_store_for = ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    //printer.Print( out_store_for );

    cu_dev.Print( out_store_for );



    // std::cerr << std::endl;

    // auto cond = ir::EQ::Make( threadidx % Expr(32), Expr(0) );
    // auto filter = ir::IfThenElse::Make( cond, out_store_for, Expr());

    cu_dev.ss_ << "} \n }" << std::endl;

    std::cerr << cu_dev.ss_.str() << std::endl;
    

     auto source_code = cu_dev.ss_.str();

  backends::nvrtc::Compiler compiler;

  auto ptx = compiler(source_code);

  std::cerr << "source code" << source_code << std::endl;

  const int N= 128 * 12 * 128 *128;
  common::float16 *a=(common::float16 *)malloc(N*sizeof(common::float16));
  common::float16 *d_a;
  cudaMalloc((void **)&d_a,N*sizeof(common::float16));

  const int num_warps = 8;
  const int block_num = 128 * 12 * 128 / num_warps / 4;
  const int NUM_PER_BLOCK = N / block_num;
  const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
  common::float16 *out=( common::float16 *)malloc(N *sizeof(common::float16));
  float *d_out;

  int M = N;
  cudaMalloc((void **)&d_out, M *sizeof(common::float16));
  common::float16 *res=(common::float16 *)malloc( M *sizeof(common::float16));
  
  srand(0);
  for(int i=0;i<N;i++){
      a[i]= static_cast<common::float16>( rand() % 100 / 100 );
  }

 for(int i=0;i< 128 * 12 * 128;i++){
    
      float cur=-100000000;
      for(int j=0;j<128;j++){
          if( cur < static_cast<float>( a[ i * 128 + j ] ) )
          {
              cur = static_cast<float>( a[i*128 + j ] );
          }
      }
      float sum = 0;
      float temp[128];
      for( int j = 0; j < 128; ++j )
      {
          temp[j] = exp( static_cast<float>(a[i*128 + j ]) - cur);
          sum += temp[j];
      }
      
      for( int j = 0; j < 128; ++j )
      {
          res[ i * 128 + j ] = static_cast<common::float16>(temp[j] / sum);
      }
  }
  std::cerr << "before copy" << std::endl;
  cudaMemcpy(d_a,a,N*sizeof(common::float16),cudaMemcpyHostToDevice);

  dim3 Grid( block_num, 1, 1);
  dim3 Block( THREAD_PER_BLOCK, 1, 1);

  void* args[] = {&d_a, &d_out };

  cinn::runtime::cuda::CUDAModule cuda_module(ptx, cinn::runtime::cuda::CUDAModule::Kind::CUBIN);
  
  for ( int i = 0; i < 1000; ++i)
  {
    cuda_module.LaunchKernel(0, "softmax_test", Grid, Block, args);
  }

  std::cerr << "before copy" << std::endl;
  cudaMemcpy(out,d_out, M *sizeof( common::float16),cudaMemcpyDeviceToHost);

  if(check(out,res,M))printf("the ans is right\n");
  else{
      printf("the ans is wrong\n");
      for(int i=0;i< M;i++){
          // printf("%lf ",out[i]);
        if( abs( static_cast<float>( out[i] ) - static_cast<float>( res[i] ) ) > 1e-5 ){
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


