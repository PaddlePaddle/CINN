set(srcs
        instruction.cc
        instructions.cc
        graph.cc
        pass.cc
        optimize.cc
        computation.cc
        module.cc
        shape.cc
        context.cc
        )

cc_test(test_instruction SRCS instruction_test.cc DEPS core hlir)
cc_test(test_computation SRCS computation_test.cc DEPS core hlir)
cc_test(test_module SRCS module_test.cc DEPS core hlir)


foreach(cpp ${srcs})
    set(hlir_src
            "${hlir_src};${CMAKE_SOURCE_DIR}/hlir/instruction/${cpp}"
            CACHE INTERNAL "")
endforeach()
