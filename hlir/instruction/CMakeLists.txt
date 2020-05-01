set(srcs
        instruction.cc
        instructions.cc
        graph.cc
        pass.cc
        pass_pipeline.cc
        optimizer.cc
        computation.cc
        module.cc
        module_group.cc
        shape.cc
        context.cc
        lower.cc
        scope.cc
        compiler.cc
        )

add_subdirectory(primitive)
add_subdirectory(pass)

cc_test(test_instruction SRCS instruction_test.cc DEPS core hlir)
cc_test(test_computation SRCS computation_test.cc DEPS core hlir)
cc_test(test_module SRCS module_test.cc DEPS core hlir)
cc_test(test_hlir_lower SRCS lower_test.cc DEPS core hlir)
cc_test(test_compiler SRCS compiler_test.cc DEPS core hlir simple_orc_jit)
cc_test(test_optimizer SRCS optimizer_test.cc DEPS core hlir)

foreach(cpp ${srcs})
    set(hlir_src
            "${hlir_src};${CMAKE_SOURCE_DIR}/hlir/instruction/${cpp}"
            CACHE INTERNAL "")
endforeach()
