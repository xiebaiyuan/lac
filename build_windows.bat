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

REM Copy DLL to output directory
echo Copying DLLs to output...
if not exist Release mkdir Release
copy D:\workspace\paddle_inference_win_x86_mkl\paddle\lib\paddle_inference.dll Release\ /Y

echo Build completed.
cd ..