set(TARGET llama-k9probe)
add_executable(${TARGET} k9probe.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# k9.h lives in src/ next to the decoder implementation
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)