/* Copyright (c) 2020 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <iostream>
#include <chrono>  // 添加计时所需头文件
#include "lac.h"

using namespace std;

int main(int argc, char* argv[]){
    // 统计初始化时间
    auto init_start_time = chrono::high_resolution_clock::now();
    
    // 读取命令行参数
    string model_path = "../models/lac_model";
    string dict_path = "";
    bool json_output = true;  // 默认使用JSON输出
    
    if (argc > 1){
        model_path = argv[1];
    }
    if (argc > 2){
        dict_path = argv[2];
    }
    
    // 检查是否有JSON输出参数
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--json" || string(argv[i]) == "-j") {
            json_output = true;
            break;
        }
        if (string(argv[i]) == "--normal" || string(argv[i]) == "-n") {
            json_output = false;
            break;
        }
    }

    // 装载模型和用户词典
    LAC lac(model_path);
    if (dict_path.length() > 1){
        lac.load_customization(dict_path);
    }

    // 统计初始化结束时间
    auto init_end_time = chrono::high_resolution_clock::now();
    // 计算初始化耗时（毫秒）
    auto init_duration = chrono::duration_cast<chrono::milliseconds>(init_end_time - init_start_time).count();
    cout << "LAC模型加载完成，初始化耗时: " << init_duration << " 毫秒" << endl;
    
    // 显示输出格式
    cout << "输出格式: " << (json_output ? "JSON" : "普通格式") << endl;
    cout << "使用 --json 或 -j 参数启用JSON输出，使用 --normal 或 -n 参数启用普通格式输出" << endl;
    
    string query;
    cout << "请输入文本(Enter退出): ";
    while (getline(cin, query)){
        if (query.empty()) {
            break;
        }

        // 计时开始
        auto start_time = chrono::high_resolution_clock::now();
        
        if (json_output) {
            // JSON格式输出
            string json_result = lac.run_json(query);
            cout << json_result << endl;
        } else {
            // 普通格式输出
            auto result = lac.run(query);
            for (size_t i = 0; i < result.size(); i++){
                if(result[i].tag.length() == 0){
                    cout << result[i].word << " ";
                }else{
                    cout << result[i].word << "/" << result[i].tag << " ";
                }
            }
            cout << endl;
        }
        
        // 计时结束
        auto end_time = chrono::high_resolution_clock::now();
        // 计算耗时（毫秒）
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        
        // 输出耗时统计
        cout << "处理耗时: " << duration << " 毫秒" << endl;
        
        cout << "请输入文本(Enter退出): ";
    }
    
    return 0;
}
