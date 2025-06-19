@echo off
setlocal enabledelayedexpansion

REM Set your Visual Studio environment
echo Setting up Visual Studio environment...
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Check if import library exists
if not exist "D:\workspace\paddle_inference_win_x86_mkl\paddle\lib\paddle_inference.lib" (
    echo Warning: paddle_inference.lib not found. Attempting to generate...
    cd /d D:\workspace\paddle_inference_win_x86_mkl\paddle\lib
    
    REM Generate .def file from DLL
    dumpbin /EXPORTS paddle_inference.dll | findstr /R "^[ ]*[0-9]" > temp_exports.txt
    echo EXPORTS > paddle_inference.def
    for /f "tokens=4" %%i in (temp_exports.txt) do echo %%i >> paddle_inference.def
    
    REM Generate import library
    lib /def:paddle_inference.def /out:paddle_inference.lib /machine:x64
    
    del temp_exports.txt paddle_inference.def
    cd /d D:\workspace\lac
)

REM Create build directory
if exist build rmdir /S /Q build
mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DWITH_STATIC_LIB=OFF ^
    -DWITH_DEMO=ON ^
    -DPADDLE_ROOT=D:/workspace/paddle_inference_win_x86_mkl ^
    -DCMAKE_CXX_FLAGS="/MD" ^
    -DCMAKE_CXX_FLAGS_RELEASE="/MD" ^
    -DCMAKE_CXX_FLAGS_DEBUG="/MDd" ^
    ..

REM Build the project
echo Building project...
cmake --build . --config Release

REM Copy additional DLLs that might be needed at runtime
echo Copying additional runtime dependencies...
if not exist Release mkdir Release

REM Set base paths for convenience
set PADDLE_BASE=D:\workspace\paddle_inference_win_x86_mkl
set THIRD_PARTY=%PADDLE_BASE%\third_party\install

REM Copy Paddle main DLL
echo Checking for Paddle main DLL...
if exist "%PADDLE_BASE%\paddle\lib\paddle_inference.dll" (
    echo   - Copying paddle_inference.dll
    copy "%PADDLE_BASE%\paddle\lib\paddle_inference.dll" Release\ /Y
) else (
    echo   - WARNING: paddle_inference.dll not found
)

if exist "%PADDLE_BASE%\paddle\lib\common.dll" (
    echo   - Copyingcommon.dll
    copy "%PADDLE_BASE%\paddle\lib\common.dll" Release\ /Y
) else (
    echo   - WARNING: common.dll not found
)

REM Copy MKL-ML dependencies
echo Checking for MKL-ML dependencies...
if exist "%THIRD_PARTY%\mklml\lib\mklml.dll" (
    echo   - Copying mklml.dll
    copy "%THIRD_PARTY%\mklml\lib\mklml.dll" Release\ /Y
) else (
    echo   - WARNING: mklml.dll not found
)

if exist "%THIRD_PARTY%\mklml\lib\libiomp5md.dll" (
    echo   - Copying libiomp5md.dll
    copy "%THIRD_PARTY%\mklml\lib\libiomp5md.dll" Release\ /Y
) else (
    echo   - WARNING: libiomp5md.dll not found
)

REM Copy MKLDNN (OneDNN) dependencies
echo Checking for MKLDNN dependencies...
if exist "%THIRD_PARTY%\mkldnn\lib\mkldnn.dll" (
    echo   - Copying mkldnn.dll
    copy "%THIRD_PARTY%\mkldnn\lib\mkldnn.dll" Release\ /Y
) else if exist "%THIRD_PARTY%\mkldnn\lib\dnnl.dll" (
    echo   - Copying dnnl.dll (newer OneDNN name)
    copy "%THIRD_PARTY%\mkldnn\lib\dnnl.dll" Release\ /Y
) else (
    echo   - WARNING: mkldnn.dll/dnnl.dll not found
)

REM Copy other potential dependencies
echo Checking for other dependencies...

REM protobuf
if exist "%THIRD_PARTY%\protobuf\lib\libprotobuf.dll" (
    echo   - Copying libprotobuf.dll
    copy "%THIRD_PARTY%\protobuf\lib\libprotobuf.dll" Release\ /Y
)

REM glog
if exist "%THIRD_PARTY%\glog\lib\glog.dll" (
    echo   - Copying glog.dll
    copy "%THIRD_PARTY%\glog\lib\glog.dll" Release\ /Y
)

REM gflags
if exist "%THIRD_PARTY%\gflags\lib\gflags.dll" (
    echo   - Copying gflags.dll
    copy "%THIRD_PARTY%\gflags\lib\gflags.dll" Release\ /Y
)

REM xxhash
if exist "%THIRD_PARTY%\xxhash\lib\xxhash.dll" (
    echo   - Copying xxhash.dll
    copy "%THIRD_PARTY%\xxhash\lib\xxhash.dll" Release\ /Y
)

REM Check for additional common locations
echo Checking alternative locations for mkldnn.dll...
for %%p in (
    "%PADDLE_BASE%\paddle\lib\mkldnn.dll"
    "%PADDLE_BASE%\lib\mkldnn.dll"
    "%THIRD_PARTY%\onednn\lib\dnnl.dll"
    "%THIRD_PARTY%\onednn\lib\mkldnn.dll"
) do (
    if exist "%%p" (
        echo   - Found and copying %%p
        copy "%%p" Release\ /Y
    )
)

REM List what was actually copied
echo.
echo Files in Release directory:
dir Release\*.dll

echo.
echo Runtime dependencies copied successfully.
echo Build completed.
cd ..