set(srcs string.cc functional.cc dot_lang.cc timer.cc error.cc
        small_vector.cc)

cc_test(test_string SRCS string_test.cc DEPS cinncore)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/utils/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
