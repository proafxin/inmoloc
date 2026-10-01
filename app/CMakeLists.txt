set(TARGET local-inference-app)

add_executable(${TARGET} llama.cpp download.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME local-inference)

target_link_libraries(${TARGET} PRIVATE
    local-inference-server-impl
    local-inference-cli-impl
    local-inference-completion-impl
    local-inference-bench-impl
    local-inference-batched-bench-impl
    local-inference-fit-params-impl
    local-inference-quantize-impl
    local-inference-perplexity-impl
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Automatically add all files from the 'licenses' directory
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*")

foreach(FILE_PATH ${EXTRA_LICENSES})
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME)
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}")
    license_add_file("${NAME}" "${FILE_PATH}")
endforeach()

license_generate(${TARGET})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
