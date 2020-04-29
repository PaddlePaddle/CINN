set(srcs
        dot.cc
        binary.cc
        )

cc_test(test_hlir_primitive_dot SRCS dot_test.cc DEPS core hlir)

foreach(cpp ${srcs})
    set(hlir_src
            "${hlir_src};${CMAKE_SOURCE_DIR}/hlir/instruction/primitive/${cpp}"
            CACHE INTERNAL "")
endforeach()
