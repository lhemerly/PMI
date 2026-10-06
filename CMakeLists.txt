cmake_minimum_required(VERSION 3.20)
project(pmi LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(PMI_WITH_LLAMA "Build llama.cpp reference and Qwen CPU connector" ON)
include(CTest)
find_package(Threads REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(PMI_CARGO_TARGET "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
set(PMI_RUST_LIBRARY "${PMI_CARGO_TARGET}/release/${CMAKE_STATIC_LIBRARY_PREFIX}pmi_runtime${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_custom_command(OUTPUT "${PMI_RUST_LIBRARY}"
  COMMAND "${CARGO_EXECUTABLE}" build --offline --locked --release
    --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/runtime/Cargo.toml"
    --target-dir "${PMI_CARGO_TARGET}"
  DEPENDS runtime/Cargo.toml runtime/Cargo.lock runtime/src/lib.rs runtime/src/direct.rs
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM)
add_custom_target(pmi_rust_build DEPENDS "${PMI_RUST_LIBRARY}")
add_library(pmi_runtime STATIC IMPORTED GLOBAL)
set_target_properties(pmi_runtime PROPERTIES IMPORTED_LOCATION "${PMI_RUST_LIBRARY}")
add_dependencies(pmi_runtime pmi_rust_build)

if(PMI_WITH_LLAMA)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/apply_llama_patch.py"
    RESULT_VARIABLE PMI_PATCH_RESULT OUTPUT_VARIABLE PMI_PATCH_OUTPUT ERROR_VARIABLE PMI_PATCH_ERROR)
  if(NOT PMI_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "${PMI_PATCH_OUTPUT}${PMI_PATCH_ERROR}")
  endif()
  message(STATUS "${PMI_PATCH_OUTPUT}")
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
  add_subdirectory(vendor/llama.cpp)
  if(GGML_VULKAN)
    # The pinned registry otherwise probes Vulkan on first CPU registration.
    # PMI registers/selects Vulkan explicitly, keeping CPU runs GPU-free.
    foreach(property COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
      get_target_property(definitions ggml ${property})
      if(definitions)
        list(REMOVE_ITEM definitions GGML_USE_VULKAN)
        set_property(TARGET ggml PROPERTY ${property} "${definitions}")
      endif()
    endforeach()
  endif()
endif()
add_library(pmi_core STATIC src/gguf.cpp src/manifest.cpp
  src/page_provider.cpp src/progress.cpp src/device_cache.cpp src/retention.cpp src/routing_trace.cpp src/connector.cpp src/paged_graph.cpp src/inference.cpp src/telemetry.cpp)
target_include_directories(pmi_core PUBLIC include)
# Build provenance for optional routing captures. Missing Git metadata remains
# unknown; a dirty checkout is explicitly distinguished from its base commit.
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE PMI_IMPLEMENTATION_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT PMI_IMPLEMENTATION_COMMIT MATCHES "^[0-9a-f]+$")
  set(PMI_IMPLEMENTATION_COMMIT "")
endif()
execute_process(COMMAND git status --porcelain --untracked-files=normal
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE PMI_WORKTREE_STATUS ERROR_QUIET)
if(PMI_WORKTREE_STATUS)
  set(PMI_IMPLEMENTATION_DIRTY 1)
else()
  set(PMI_IMPLEMENTATION_DIRTY 0)
endif()
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-pmi-external-tensors.patch" PMI_BACKEND_PATCH_SHA256)
target_compile_definitions(pmi_core PRIVATE
  PMI_IMPLEMENTATION_COMMIT="${PMI_IMPLEMENTATION_COMMIT}"
  PMI_IMPLEMENTATION_DIRTY=${PMI_IMPLEMENTATION_DIRTY}
  PMI_BACKEND_PATCH_SHA256="${PMI_BACKEND_PATCH_SHA256}")
target_link_libraries(pmi_core PUBLIC pmi_runtime Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX)
  target_link_libraries(pmi_core PUBLIC m)
endif()
if(PMI_WITH_LLAMA)
  target_sources(pmi_core PRIVATE src/ggml_device.cpp src/device_projection.cpp)
  target_include_directories(pmi_core PRIVATE vendor/llama.cpp/ggml/src)
  target_compile_definitions(pmi_core PUBLIC PMI_WITH_LLAMA=1)
  target_link_libraries(pmi_core PUBLIC llama ggml-cpu ggml-base)
  if(GGML_VULKAN)
    target_compile_definitions(pmi_core PUBLIC PMI_WITH_VULKAN=1)
    target_link_libraries(pmi_core PUBLIC ggml-vulkan)
  endif()
endif()
add_executable(pmi src/main.cpp)
target_link_libraries(pmi PRIVATE pmi_core)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME fixture_generate COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_fixture.py" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/toy.gguf")
  set_tests_properties(fixture_generate PROPERTIES FIXTURES_SETUP toy_fixture)
  add_test(NAME self_test COMMAND pmi self-test "${CMAKE_CURRENT_BINARY_DIR}/fixtures/toy.gguf")
  set_tests_properties(self_test PROPERTIES FIXTURES_REQUIRED toy_fixture)
  add_test(NAME remote_header_reader COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/inspect_remote_headers.py" --self-test)
  add_test(NAME pinned_download_tools COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/download_qwen_weights_test.py")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME bounded_command_runner COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_bounded_test.py")
    add_test(NAME linux_memory_telemetry COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/linux_memory_telemetry_test.py")
    add_test(NAME analyze_progress COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/analyze_progress_test.py")
    add_test(NAME loader_diagnostics COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/loader_diagnostics_test.py")
    add_test(NAME loader_pressure_analysis COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/analyze_loader_pressure_test.py")
    add_test(NAME pressure_policy COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/pressure_policy_test.py")
    add_test(NAME vulkan_smoke_guard COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_smoke_guard_test.py")
  endif()
  add_test(NAME rust_runtime COMMAND "${CARGO_EXECUTABLE}" test --offline --locked
    --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/runtime/Cargo.toml" --target-dir "${PMI_CARGO_TARGET}")
  add_executable(pmi_shard_manifest_test tests/shard_manifest.cpp)
  target_link_libraries(pmi_shard_manifest_test PRIVATE pmi_core)
  add_test(NAME shard_manifest COMMAND pmi_shard_manifest_test)
  add_executable(pmi_page_provider_test tests/page_provider_test.cpp)
  target_link_libraries(pmi_page_provider_test PRIVATE pmi_core)
  add_test(NAME page_provider COMMAND pmi_page_provider_test)
  add_executable(pmi_direct_page_provider_test tests/direct_page_provider_test.cpp)
  target_link_libraries(pmi_direct_page_provider_test PRIVATE pmi_core)
  add_test(NAME direct_page_provider COMMAND pmi_direct_page_provider_test "${CMAKE_CURRENT_BINARY_DIR}/direct-provider")
  add_executable(pmi_progress_writer_test tests/progress_writer_test.cpp)
  target_link_libraries(pmi_progress_writer_test PRIVATE pmi_core)
  add_test(NAME progress_writer COMMAND pmi_progress_writer_test "${CMAKE_CURRENT_BINARY_DIR}/progress-writer-test")
  add_executable(pmi_owned_buffer_test tests/owned_buffer_test.cpp)
  target_link_libraries(pmi_owned_buffer_test PRIVATE pmi_core)
  add_test(NAME owned_buffer_advice COMMAND pmi_owned_buffer_test)
  add_executable(pmi_device_cache_test tests/device_cache_test.cpp)
  target_link_libraries(pmi_device_cache_test PRIVATE pmi_core)
  add_test(NAME device_cache COMMAND pmi_device_cache_test)
  add_executable(pmi_retention_test tests/retention_test.cpp)
  target_link_libraries(pmi_retention_test PRIVATE pmi_core)
  add_test(NAME protected_retention COMMAND pmi_retention_test)
  # Optional offline replay of a locally saved routing trace; no model payloads.
  add_executable(pmi_retention_replay tests/retention_replay.cpp)
  target_link_libraries(pmi_retention_replay PRIVATE pmi_core)
  add_executable(pmi_routing_trace_test tests/routing_trace_test.cpp)
  target_link_libraries(pmi_routing_trace_test PRIVATE pmi_core)
  add_test(NAME routing_trace_writer COMMAND pmi_routing_trace_test)
  add_test(NAME routing_dataset COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/routing_dataset_test.py")
  add_test(NAME vulkan_comparison_policy COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/vulkan_comparison_policy_test.py")
  add_test(NAME moe_replay_extraction COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/moe_replay_extract_test.py")
  if(PMI_WITH_LLAMA)
    add_test(NAME teacher_forcing COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/teacher_forcing_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/teacher-forcing")
    set_tests_properties(teacher_forcing PROPERTIES TIMEOUT 60)
    add_test(NAME progress_model COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/progress_model_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/progress-model")
    set_tests_properties(progress_model PROPERTIES TIMEOUT 60)
    add_test(NAME direct_expert_model COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/direct_expert_model_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/direct-expert")
    set_tests_properties(direct_expert_model PROPERTIES TIMEOUT 60)
    add_test(NAME retained_host_model COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/retained_host_model_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/retained-host")
    set_tests_properties(retained_host_model PROPERTIES TIMEOUT 60)
    add_test(NAME split_dispatch COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/split_dispatch_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/split-dispatch")
    set_tests_properties(split_dispatch PROPERTIES TIMEOUT 60)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_test(NAME base_pages_model COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/base_pages_model_test.py"
        "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/base-pages-model")
      set_tests_properties(base_pages_model PROPERTIES TIMEOUT 60)
      add_test(NAME direct_io_model COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/direct_io_model_test.py"
        "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/direct-io-model")
      set_tests_properties(direct_io_model PROPERTIES TIMEOUT 60)
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_executable(pmi_load_observer_test tests/load_observer_test.cpp)
        target_include_directories(pmi_load_observer_test PRIVATE vendor/llama.cpp/ggml/src)
        target_link_libraries(pmi_load_observer_test PRIVATE pmi_core ${CMAKE_DL_LIBS})
        target_link_options(pmi_load_observer_test PRIVATE "LINKER:--export-dynamic")
        set_tests_properties(base_pages_model PROPERTIES
          ENVIRONMENT "PMI_OBSERVER_TEST=$<TARGET_FILE:pmi_load_observer_test>")
        add_executable(pmi_direct_io_test tests/direct_io_test.cpp vendor/llama.cpp/src/llama-mmap.cpp)
        target_include_directories(pmi_direct_io_test PRIVATE vendor/llama.cpp/src)
        target_link_libraries(pmi_direct_io_test PRIVATE llama ggml-base)
        foreach(symbol open read close fread posix_memalign free fstat lseek)
          target_link_options(pmi_direct_io_test PRIVATE "LINKER:--wrap=${symbol}")
        endforeach()
        add_test(NAME direct_io_files COMMAND pmi_direct_io_test)
      endif()
    endif()
    add_executable(pmi_ggml_device_test tests/ggml_device_test.cpp)
    target_link_libraries(pmi_ggml_device_test PRIVATE pmi_core)
    add_test(NAME ggml_device_adapter COMMAND pmi_ggml_device_test)
    add_executable(pmi_device_projection_test tests/device_projection_test.cpp)
    target_link_libraries(pmi_device_projection_test PRIVATE pmi_core)
    add_test(NAME device_projection COMMAND pmi_device_projection_test)
    add_executable(pmi_moe_projection_replay tests/moe_projection_replay.cpp)
    target_link_libraries(pmi_moe_projection_replay PRIVATE pmi_core)
    if(GGML_VULKAN)
      # GPU execution requires explicit host/VRAM guards; never run implicitly
      # in the ordinary CPU CTest suite.
      add_executable(pmi_vulkan_projection_test tests/vulkan_projection_test.cpp)
      target_link_libraries(pmi_vulkan_projection_test PRIVATE pmi_core)
      add_executable(pmi_vulkan_activation_parity_test tests/vulkan_activation_parity_test.cpp)
      target_link_libraries(pmi_vulkan_activation_parity_test PRIVATE pmi_core)
    endif()
    add_test(NAME device_full_model_paging COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/device_full_model_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/device-full-model")
    set_tests_properties(device_full_model_paging PROPERTIES TIMEOUT 180)
    add_executable(pmi_type_sizes_test tests/type_sizes.cpp)
    target_link_libraries(pmi_type_sizes_test PRIVATE pmi_core)
    add_test(NAME ggml_type_sizes COMMAND pmi_type_sizes_test)
    add_executable(pmi_paged_graph_test tests/paged_graph_test.cpp)
    target_link_libraries(pmi_paged_graph_test PRIVATE pmi_core)
    add_test(NAME paged_graph_ops COMMAND pmi_paged_graph_test)
    add_test(NAME full_model_paging COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/full_model_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/full-model")
    add_executable(pmi_connector_test tests/connector_test.cpp)
    target_link_libraries(pmi_connector_test PRIVATE pmi_core)
    add_executable(pmi_target_moe_reference tests/target_moe_reference.cpp)
    target_link_libraries(pmi_target_moe_reference PRIVATE pmi_core)
    add_test(NAME qwen_moe_connector COMMAND pmi_connector_test "${CMAKE_CURRENT_BINARY_DIR}/fixtures/moe")
    set_tests_properties(qwen_moe_connector PROPERTIES FIXTURES_SETUP moe_fixture)
    add_test(NAME target_moe_reference_fixture COMMAND pmi_target_moe_reference
      "${CMAKE_CURRENT_BINARY_DIR}/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/qwen-moe-00001-of-00002.gguf"
      "${CMAKE_CURRENT_BINARY_DIR}/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/qwen-moe-00002-of-00002.gguf"
      --layer 0 --input "${CMAKE_CURRENT_BINARY_DIR}/fixtures/moe/mixed-iq3_s-iq4_nl-q8_0/input.txt"
      --cache-bytes 18432)
    set_tests_properties(target_moe_reference_fixture PROPERTIES FIXTURES_REQUIRED moe_fixture)
    add_test(NAME moe_cli COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.py"
      "$<TARGET_FILE:pmi>" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/moe")
    set_tests_properties(moe_cli PROPERTIES FIXTURES_REQUIRED moe_fixture)
  endif()
endif()
