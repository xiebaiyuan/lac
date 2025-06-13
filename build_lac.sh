# 代码下载
git clone https://github.com/baidu/lac.git
cd lac

# /path/to/paddle是第1步中获取的Paddle依赖库路径
# 即下载解压后的文件夹路径或编译产出的文件夹路径
PADDLE_ROOT=/Users/xiebaiyuan/PaddleOCR/deploy/cpp_infer/paddle_inference/paddle_inference_install_dir_mac_universal

# 编译
mkdir build
cd build
cmake -DPADDLE_ROOT=$PADDLE_ROOT \
      -DWITH_DEMO=ON \
      -DWITH_JNILIB=OFF \
      ../

make install # 编译产出在 ../output 下