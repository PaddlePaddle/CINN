set(srcs string.cc functional.cc dot_lang.cc timer.cc error.cc
        small_vector.cc)

set(includes
    dot_lang.h
    error.h
    functional.h
    registry.h
    small_vector.h
    string.h
    timer.h
    )

cc_test(test_string SRCS string_test.cc DEPS cinncore)

foreach(cpp ${srcs})
  set(core_src
      "${core_src};cinn/utils/${cpp}"
      CACHE INTERNAL "")
endforeach()

foreach(header ${includes})
  set(core_includes "${core_includes};cinn/utils/${header}" CACHE INTERNAL "")
endforeach()
