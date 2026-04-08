# cmake/Metal.cmake
# CMake configuration for Apple's Metal GPU backend
# Include this after setting up the project to enable Metal compilation

#[=======================================================================[
Metal.cmake - Metal GPU Backend Configuration for FIDESlib
#=======================================================================]

# Check if we're on macOS
if(NOT APPLE)
    message(WARNING "Metal is only available on macOS. Metal backend will be disabled.")
    set(FIDES_USE_METAL OFF CACHE BOOL "Enable Metal GPU backend" FORCE)
    return()
endif()

# Find Metal toolchain
find_program(XCRUN_EXECUTABLE xcrun PATHS /usr/bin /usr/local/bin)
if(NOT XCRUN_EXECUTABLE)
    message(WARNING "xcrun not found. Metal backend requires Xcode command line tools.")
    set(FIDES_USE_METAL OFF CACHE BOOL "Enable Metal GPU backend" FORCE)
    return()
endif()

# Metal SDK version
execute_process(
    COMMAND ${XCRUN_EXECUTABLE} --version
    OUTPUT_VARIABLE METAL_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "([0-9]+\.[0-9]+)" METAL_VERSION_MATCH "${METAL_VERSION}")
if(METAL_VERSION_MATCH)
    set(METAL_SDK_VERSION "${METAL_VERSION_MATCH}")
endif()

message(STATUS "Metal SDK version: ${METAL_SDK_VERSION}")

# Option to enable Metal
option(FIDES_USE_METAL "Enable Metal GPU backend on macOS" ON)

if(NOT FIDES_USE_METAL)
    message(STATUS "Metal backend disabled")
    return()
endif()

# Metal source files
set(METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/Metal")

file(GLOB METAL_SHADER_FILES "${METAL_SOURCE_DIR}/*.metal")

if(METAL_SHADER_FILES STREQUAL "")
    message(WARNING "No Metal shader files found in ${METAL_SOURCE_DIR}")
    set(FIDES_USE_METAL OFF CACHE BOOL "Enable Metal GPU backend" FORCE)
    return()
endif()

message(STATUS "Found Metal shaders: ${METAL_SHADER_FILES}")

# Custom command to compile Metal shaders to .air files
set(METAL_COMPILE_SCRIPT [[
#!/bin/bash
METAL_DIR="$1"
OUTPUT_DIR="$2"
shift 2

# Compile each shader
for shader in "$@"; do
    basename=$(basename "$shader")
    name="${basename%.metal}"
    echo "Compiling $shader..."
    xcrun metal -c "$shader" -o "$OUTPUT_DIR/${name}.air"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to compile $shader"
        exit 1
    fi
done

# Create combined metallib
AIR_FILES=()
for shader in "$@"; do
    basename=$(basename "$shader")
    name="${basename%.metal}"
    AIR_FILES+=("$OUTPUT_DIR/${name}.air")
done

xcrun metallib "${AIR_FILES[@]}" -o "$OUTPUT_DIR/fideslib_metal.metallib"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create metallib"
    exit 1
fi

echo "Metal compilation complete: $OUTPUT_DIR/fideslib_metal.metallib"
]])

# Create compile script
set(METAL_COMPILE_SCRIPT_FILE "${CMAKE_BINARY_DIR}/compile_metal.sh")
file(WRITE "${METAL_COMPILE_SCRIPT_FILE}" "${METAL_COMPILE_SCRIPT}")
execute_process(
    COMMAND chmod +x "${METAL_COMPILE_SCRIPT_FILE}"
)

# Add custom target for Metal compilation
add_custom_target(metal_lib ALL
    COMMAND "${METAL_COMPILE_SCRIPT_FILE}" "${METAL_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/metal"
    DEPENDS ${METAL_SHADER_FILES}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Compiling Metal shaders"
    VERBATIM
)

# Install metallib
install(FILES "${CMAKE_BINARY_DIR}/metal/fideslib_metal.metallib"
        DESTINATION "${INSTALL_LIBRARY_DIR}"
        RENAME "libfideslib_metal.metallib")

message(STATUS "Metal backend enabled")