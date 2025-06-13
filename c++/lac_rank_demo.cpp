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
#include "lac.h"

using namespace std;

int main(int argc, char* argv[]){
    // 读取命令行参数
    string lac_model_path = "../models/lac_model";
    string rank_model_path = "../models/rank_model";
    string dict_path = "";
    
    if (argc > 1){
        lac_model_path = argv[1];
    }
    if (argc > 2){
        rank_model_path = argv[2];
    }
    if (argc > 3){
        dict_path = argv[3];
    }

    // 装载LAC模型
    LAC lac(lac_model_path);
    
    // 启用rank模式
    lac.enable_rank_mode(rank_model_path);
    
    // 可选：加载用户词典
    if (dict_path.length() > 1){
        lac.load_customization(dict_path);
    }

    string query;
    cout << "请输入文本(Enter退出): ";
    while (getline(cin, query)){
        if (query.empty()) {
            break;
        }

        // 执行并打印结果
        auto result = lac.run_rank(query);
        for (size_t i = 0; i < result.size(); i++){
            cout << result[i].word << "/" << result[i].tag << "/" << result[i].rank << " ";
        }
        cout << endl;
        cout << "请输入文本(Enter退出): ";
    }
    
    return 0;
}
