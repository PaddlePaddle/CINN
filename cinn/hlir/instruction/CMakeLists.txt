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
        shape_inference.cc
        )

add_subdirectory(primitive)
add_subdirectory(pass)
add_subdirectory(x86)

cc_test(test_instruction SRCS instruction_test.cc DEPS core)
cc_test(test_computation SRCS computation_test.cc DEPS core)
cc_test(test_module SRCS module_test.cc DEPS core)
cc_test(test_hlir_lower SRCS lower_test.cc DEPS core)
cc_test(test_compiler SRCS compiler_test.cc DEPS core)
cc_test(test_optimizer SRCS optimizer_test.cc DEPS core)

foreach(cpp ${srcs})
    set(core_src
            "${core_src};${CMAKE_SOURCE_DIR}/cinn/hlir/instruction/${cpp}"
            CACHE INTERNAL "")
endforeach()
