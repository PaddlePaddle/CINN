

#include "cinn/ir/ir_verify.h"

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

#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>

#include "cinn/lang/placeholder.h"

#include "cinn/hlir/framework/visualize_helper.h"

#include <cmath>

#include "cinn/ir/thread_model.h"
#include "cinn/hlir/framework/op_lowering_util.h"
#include "cinn/common/type.h"

namespace cinn {
namespace ir {

struct InputNode
{
    InputNode() {}
    InputNode( std::string n, cinn::lang::Placeholder<float> *p, std::vector<int> dim)
        : name(n), in_ptr(p), in_dim(dim) {}
    std::string name;
    cinn::lang::Placeholder<float>* in_ptr;
    std::vector<int> in_dim;
};

struct ThreadConfig
{
    int warp_round;
    int thread_round;
};

void process_reduce_max( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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
    
    std::cerr << in_name << "\t" << input_map->count( in_name ) << std::endl;
    InputNode& input_node = input_map->at(in_name);
    std::cerr << in_name << " found" << std::endl;

    Var loop_var("i");
    Var loop_var_j("j");
    Expr inf(-100000.0);
    std::string temp_max_name = "tmp_max";
    Var temp_max_var( temp_max_name, type_of<float>() );
    
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round});
    std::cerr << in_name << "found  #1" << std::endl;

    std::string name1 = "max1";
    cinn::ir::Var max_t(name1, type_of<float>());

    std::cerr << in_name << "found  #3" << std::endl;
    auto max_var = cinn::ir::Let::Make( max_t, inf);
    std::cerr << in_name << "found  #5" << std::endl;
    auto t_load = ir::Load::Make( ir::Tensor( *(input_node.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    std::cerr << in_name << "found  1" << std::endl;
    auto new_max = ir::Max::Make( max_t, t_load);
    auto out_max = ir::Let::Make( max_t, new_max, false);
  


    auto body =  ir::Block::Make( {out_max });
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);
    
    std::cerr << in_name << "found 2" << std::endl;
    auto warp_call = Call::Make( Float(32), "warpReduceMax", {max_t}, {}, ir::CallType::Extern );

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
   
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );    

    std::cerr << "inert max " << std::endl;
    (*input_map)[out_name] = InputNode( "reduce_max", T_MAX, {1});

    std::cerr << "inert max fin" << std::endl;
}


void process_sub( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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

    std::cerr << "name " << vec_in_names[0] << "\t" << vec_in_names[1] << std::endl;

    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    bool is_scalar = false;
    if( second_input.in_dim.size() == 0)
    {
        is_scalar = true;
    }
    //int broadcast_axis = reduce_axis( first_input.in_dim, second_input.in_dim);
    
    int broadcast_first = -1;
    int broadcast_second = -1;
    if( first_input.in_dim.size() == 1)
    {
        broadcast_first = 1;
    }

    if( second_input.in_dim.size() == 1 )
    {
        broadcast_second = 1;
    }

    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    Expr inf(-100000.0);
    
    std::string temp_max_name = vec_in_names[0] + "_" + vec_in_names[1] +  "_mul_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    std::string name1 = "sub_tmp";
    cinn::ir::Var max_t(name1, type_of<float>());

    auto max_var = cinn::ir::Let::Make( max_t, inf);

    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    Expr t2_load;

    if( is_scalar )
    {
        t2_load = Var( second_input.name, type_of<float>());
    }else if( broadcast_second != -1)    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var)});
    } 
     else
    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    } 
    

    auto out = ir::Sub::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* sub = new cinn::lang::Placeholder<float>( temp_max_name, std::vector<int>{{1, 8}});
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
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );


    (*input_map)[out_name] = InputNode( temp_max_name, sub, {1, 8});

}


void process_exp( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
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
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( "exp", exp, {1, 8});

}

void process_sqrt( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    std::string temp_max_name = "sqrt";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    std::string name1 = "sqrt";
    cinn::ir::Var max_t(name1, type_of<float>());


    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    auto out = ir::Sqrt::Make( t_load);

    cinn::ir::IrPrinter printer(std::cout);

    cinn::lang::Placeholder<float>* exp = new cinn::lang::Placeholder<float>("sqrt", std::vector<int>{{1, 4}});
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
    
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( "sqrt", exp, {1, 8});

}

void process_reduce_sum( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config, 
            cosnt CodeGenOption& gen_opt )
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
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    Expr zero(0.0);
    std::string temp_max_name = in_name + "_tmp_sum";
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
    

    string reduce_func_name; 
    if( gen_opt.reduce_block <= 128 )
    {
        reduce_func_name = "warpReduceSum";
    }
    else
    {
        reduce_func_name = "BlockReduceSum";
    }

    auto warp_call = Call::Make( Float(32), reduce_func_name, {sum1}, {}, ir::CallType::Extern );

    //auto warp_res = ir::Let::Make( max_t, warp_call, false);

    cinn::lang::Placeholder<float>* T_SUM = new cinn::lang::Placeholder<float>(temp_max_name, std::vector<int>{{1, 4}});
    auto max_store = Store::Make( ir::Tensor(*T_SUM), warp_call, {Expr(loop_var)}); 


    body =  ir::Block::Make( {max_var,  load_inner_for, max_store});

    auto max_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( "reduce_sum", T_SUM, {1});

}


void process_divide( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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

    std::cerr << vec_in_names[0] << "\t" << vec_in_names[1] << std::endl;
    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    bool is_scalar = false;
    if( second_input.in_dim.size() == 0)
    {
        is_scalar = true;
    }
    //int broadcast_axis = reduce_axis( first_input.in_dim, second_input.in_dim);
    
    int broadcast_first = -1;
    int broadcast_second = -1;
    if( first_input.in_dim.size() == 1)
    {
        broadcast_first = 1;
    }

    if( second_input.in_dim.size() == 1 )
    {
        broadcast_second = 1;
    }
    
    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    Expr inf(-100000.0);
    std::string temp_max_name = vec_in_names[0] + "_" + vec_in_names[1] +  "_div_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    Expr t_load; 
    
    if( broadcast_first == -1){
        t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    }
    else{
        t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var) });
    }

    
    Expr t2_load;

    if( is_scalar )
    {
        t2_load = Var( second_input.name, type_of<float>());
    }else if( broadcast_second != -1)    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var)});
    } 
     else
    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    }  
    std::cerr << "t2 load " << t2_load << std::endl;

    auto out = ir::Div::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* div = new cinn::lang::Placeholder<float>(temp_max_name, std::vector<int>{{1, 4}});
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
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( "divide", div, {1, 8});

}

void process_add( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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

    std::cerr << vec_in_names[0] << "\t" << vec_in_names[1] << std::endl;
    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    bool is_scalar = false;
    if( second_input.in_dim.size() == 0)
    {
        is_scalar = true;
    }

    
    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    Expr inf(-100000.0);
    std::string temp_max_name = vec_in_names[0] + "_" + vec_in_names[1] +  "_div_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    
    Expr t2_load;

    if( is_scalar )
    {
        t2_load = Var( second_input.name, type_of<float>());
    }  else
    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    }  
    //std::cerr << "t2 load " << t2_load << std::endl;

    auto out = ir::Add::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* div = new cinn::lang::Placeholder<float>(temp_max_name, std::vector<int>{{1, 4}});
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
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( "add", div, {1, 8});

}

void process_mul( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
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

    //std::cerr << vec_in_names[0] << "\t" << vec_in_names[1] << std::endl;
    InputNode& first_input = input_map->at( vec_in_names[0]);
    InputNode& second_input = input_map->at( vec_in_names[1]);

    bool is_scalar = false;
    if( second_input.in_dim.size() == 0)
    {
        is_scalar = true;
    }
    //int broadcast_axis = reduce_axis( first_input.in_dim, second_input.in_dim);

    
    Var loop_var("i");
    Var loop_var_j("j");
    int warp_round = thread_config.warp_round;
    int thread_round = thread_config.thread_round;
    Expr inf(-100000.0);

    std::string temp_max_name = vec_in_names[0] + "_" + vec_in_names[1] +  "_mul_tmp";
    Var temp_max_var( temp_max_name, type_of<float>() );
    auto temp_max_out = LocalTemp::Make( temp_max_var, {warp_round, thread_round});

    auto t_load = ir::Load::Make( ir::Tensor( *(first_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });

    
    Expr t2_load;

    if( is_scalar )
    {
        t2_load = Var( second_input.name, type_of<float>());
    }  else
    {
        t2_load = ir::Load::Make( ir::Tensor( *(second_input.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    }  
    // std::cerr << "t2 load " << t2_load << std::endl;

    auto out = ir::Mul::Make( t_load, t2_load);

    cinn::lang::Placeholder<float>* div = new cinn::lang::Placeholder<float>(temp_max_name, std::vector<int>{{1, 4}});
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
    
    vec_out.push_back( temp_max_out );
    vec_out.push_back( max_outer_for );

    (*input_map)[out_name] = InputNode( temp_max_name, div, {1, 8});

}


void process_fillconstant( std::map<std::string, InputNode>* input_map, hlir::framework::Node* node, std::vector<Expr>& vec_out, ThreadConfig& thread_config)
{
    std::string out_name;
    for (auto& outlink : node->outlinks_in_order(true)) {
        auto* outnode = outlink->sink()->safe_as<hlir::framework::NodeData>();
        if (outnode) {
            out_name = outnode->id();
        }
    }

    //std::cerr << hlir::framework::DebugString(node) << std::endl;
    auto* op =  node->op();

    auto value = absl::get<float>(node->attrs.attr_store.at("value"));

    //std::cerr << value << std::endl;

    auto dtype = absl::get<float>(node->attrs.attr_store.at("value"));

    // std::cerr << out_name << "\t" << dtype << std::endl;


    
    cinn::ir::Var out(out_name, type_of<float>());

    auto max_var = cinn::ir::Let::Make( out, Expr(value));

    vec_out.push_back( max_var );
    
    (*input_map)[out_name] = InputNode( out_name, nullptr, {});
}

ir::Expr generate_index( CodeGenOption gen_opt, bool last_dim)
{
    int reduce_block = gen_opt.reduce_block;
    int flatten_block = gen_opt.flatten_block;    
    
    std::vector<int> reduce_range;
    std::vector<int> flatten_range;
    
    std::string name_blockx = "blockIdx.x";
    std::string name_threadx = "xid";
    std::string index_name = "index";
    Var block_x_var( name_blockx, type_of<int>() );
    Var thread_x_var( name_threadx, type_of<int>() );    

    Var block_id( "blockIdx.x", type_of<int>() );
    Var flatten_id( "xid", type_of<int>() );
    Var r_id( "rid", type_of<int>() );
    Expr expr_flatten( flatten_block);    
    Expr expr_reduce( reduce_block);

    int num_warp = gen_opt.num_warp;
    int num_thread_per_warp = gen_opt.num_thread_per_warp;
    
    Var threadidx("threadIdx.x", type_of<int>());
    Var index_i("i", type_of<int>() );
    Var index_j("j", type_of<int>() );
    Expr expr_warp( num_warp);
    Expr expr_thread_per_warp( num_thread_per_warp );

    auto warp_id = threadidx / expr_thread_per_warp;

    // warp reduce
    auto warp_round = flatten_block / num_warp;
    auto thread_round = reduce_block / num_thread_per_warp;

    auto xid = warp_id * Expr( warp_round ) + index_i;
    auto inner_id = threadidx % expr_thread_per_warp;
    auto inner_index =  xid * Expr( thread_round ) * expr_thread_per_warp + inner_id + index_j * expr_thread_per_warp;
    if( ! last_dim )
    {
        inner_index = block_id *Expr( gen_opt.reduce_dim * flatten_block ) + inner_index;
    }
    
    return inner_index;  
}

void build_load(  std::map<std::string, InputNode>* input_map, CodeGenOption gen_opt, const std::vector<std::string>& vec_input, std::vector<Expr>& vec_out)
{   
    int reduce_block = gen_opt.reduce_block;
    int flatten_block = gen_opt.flatten_block;
    int num_warp = gen_opt.num_warp;
    int num_thread_per_warp = gen_opt.num_thread_per_warp;

    auto warp_round = flatten_block / num_warp;
    auto thread_round = reduce_block / num_thread_per_warp;

    auto inner_index = generate_index( gen_opt, false);

 

    Var loop_var("i");

    if( vec_input.size() != 1)
    {
        std::cerr << "not support input size not equal 1" << std::endl;
        throw std::runtime_error("input not equal 1");
    }
   
    std::string temp_name = "tmp";
    Var temp_var( temp_name, type_of<float>() );
    auto temp_out = LocalTemp::Make( temp_var, {warp_round, thread_round});
    cinn::lang::Placeholder<float> *C = new cinn::lang::Placeholder<float>( vec_input[0], std::vector<int>{{10, 10}});
    cinn::lang::Placeholder<float> *T = new cinn::lang::Placeholder<float>("tmp", std::vector<int>{{1,4}});
    //Placeholder<float> A("A", std::vector<int>{{10}});
    //Var input( "input", type_of<float>( ));
    Var loop_var_j("j");

    auto t_load = ir::Load::Make( ir::Tensor(*C), { inner_index });

    Expr body = Store::Make( ir::Tensor(*T), t_load, {Expr(loop_var), Expr(loop_var_j)}); 
                            
    body      = ir::Block::Make({body});
    
    auto load_inner_for = ir::For::Make(loop_var_j,
                               common::make_const(0),
                               common::make_const(thread_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);

    //printer.Print( load_inner_for );


    body =  ir::Block::Make( {load_inner_for});

    auto load_outer_for =  ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(warp_round),
                               ir::ForType::Unrolled,
                               ir::DeviceAPI::CUDA,
                               body);


    vec_out.push_back( temp_out );
    vec_out.push_back( load_outer_for );



    (*input_map)[ vec_input[0] ] = InputNode( vec_input[0], T, {1, 4});    
    
}

void  build_store( std::map<std::string, InputNode>* input_map,  CodeGenOption gen_opt, const std::vector<std::string>& vec_output_name, std::vector<Expr>& vec_out)
{
    int reduce_block = gen_opt.reduce_block;
    int flatten_block = gen_opt.flatten_block;
    int num_warp = gen_opt.num_warp;
    int num_thread_per_warp = gen_opt.num_thread_per_warp;

    auto warp_round = flatten_block / num_warp;
    auto thread_round = reduce_block / num_thread_per_warp;
    
    auto inner_index = generate_index( gen_opt, false);


    if( vec_output_name.size() != 1)
    {
        std::cerr << "not support output size not equal 1" << std::endl;
        throw std::runtime_error("output not equal 1");
    }

    Var loop_var("i");
    Var loop_var_j("j");

    auto var_out = input_map->at( vec_output_name[0] );

    auto t_load = ir::Load::Make( ir::Tensor( *(var_out.in_ptr) ), { Expr(loop_var), Expr(loop_var_j) });
    cinn::lang::Placeholder<float> OUT(vec_output_name[0], std::vector<int>{{10}});

    auto out_store = Store::Make( ir::Tensor(OUT), t_load, { Expr( inner_index ) });

    auto body =  ir::Block::Make( {out_store });
    auto load_inner_for = ir::For::Make(loop_var_j,
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

    vec_out.push_back( out_store_for);
}

ir::LoweredFunc process_warp_reduce(  hlir::framework::Graph * graph, CodeGenOption gen_opt, 
     const std::vector<std::string>& vec_input,  const std::vector<std::string>& vec_output_name)
{
    std::vector<Expr> out_expr;
    std::map<std::string, InputNode> map_input;
    
    build_load( &map_input, gen_opt, vec_input, out_expr );

    auto topo_order = graph->topological_order();
    auto& nodes     = std::get<0>(topo_order);

    std::vector<ir::Tensor> func_args;
    std::unordered_map<std::string, ir::Tensor> tensor_map;

    auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
    auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
    
    int reduce_block = gen_opt.reduce_block;
    int flatten_block = gen_opt.flatten_block;  
    auto num_warp = gen_opt.num_warp;
    auto  num_thread_per_warp = gen_opt.num_thread_per_warp;
    auto warp_round = flatten_block / num_warp;
    auto thread_round = reduce_block / num_thread_per_warp;
    ThreadConfig thread_config;
    thread_config.warp_round = warp_round;
    thread_config.thread_round = thread_round;
    
    for (auto& n : nodes) {

        auto node = n->safe_as<hlir::framework::Node>();
        if (!node || node->op() == nullptr) {
            continue;
        }

        std::vector<ir::Tensor> tensor_inputs =
            std::move(hlir::framework::CollectInputTensor(node, func_args, tensor_map, dtype_dict, shape_dict));

        std::cerr << " process node: " << node->id() << std::endl;
        std::cerr << " with op type: " << node->op()->name << std::endl;
        if( node->op()->name == "reduce_max")
        {
            process_reduce_max( &map_input, node, out_expr, thread_config );
        }else if ( node->op()->name == "subtract" )
        {
            process_sub( &map_input, node, out_expr, thread_config);
        }else if ( node->op()->name == "exp" )
        {
            process_exp( &map_input, node, out_expr, thread_config);
        }else if ( node->op()->name == "reduce_sum" )
        {
            process_reduce_sum( &map_input, node, out_expr, thread_config, gen_opt);
        }else if ( node->op()->name == "divide" )
        {
            process_divide( &map_input, node, out_expr, thread_config);
        }
        
    }

  build_store( &map_input, gen_opt, vec_output_name, out_expr);

  auto feed_list= vec_input;
  std::vector<ir::Argument> test_func_args;
  std::cerr << "feed list" << std::endl;
  for( auto& name : feed_list )
  {
    std::cerr << name << std::endl;
    test_func_args.emplace_back( tensor_map.at(name)->buffer, ir::Argument::IO::kInput );
  }


  auto fetch_name_list = vec_output_name;
  std::cerr << "fetch list" << std::endl;
  // build output
  for( auto& name : fetch_name_list )
  {
    std::cerr << name << std::endl;
    auto out = lang::Placeholder<float>( name, shape_dict.at( name ));
    test_func_args.emplace_back( out->buffer , ir::Argument::IO::kOutput );
  }

  std::vector<ir::Buffer> temp_buffers;
    
  
  auto group0 = graph->fusion_groups[0];
  std::cerr << "fun name " << group0->GetFuncName() << std::endl;
  auto func =
      ir::_LoweredFunc_::Make( group0->GetFuncName() , test_func_args, cinn::ir::Block::Make( out_expr ), temp_buffers);

  func->cuda_axis_info.set_grid_dim( 0, 128 * 12 * 128 / 32);
  func->cuda_axis_info.set_block_dim( 0, 256);
  //std::cerr << "func " << func << std::endl;  

  return func;


}

}  //namespace ir
}  //namespace cinn