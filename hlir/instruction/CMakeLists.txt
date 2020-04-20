set(srcs
        instruction.cc
        instructions.cc
        graph.cc
        pass.cc
        optimize.cc
        computation.cc
        module.cc
        shape.cc
        )

cc_test(test_instruction SRCS instruction_test.cc DEPS core)


foreach(cpp ${srcs})
    set(hlir_src
            "${hlir_src};${CMAKE_SOURCE_DIR}/hlir/instruction/${cpp}"
            CACHE INTERNAL "")
endforeach()
