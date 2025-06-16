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
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>
#include "lac.h"

using namespace std;
using namespace chrono;

// 添加JSON构建辅助函数
string build_result_with_timing_json(const vector<OutputItem>& result, 
                                    const string& input_text,
                                    double total_time_ms) {
    ostringstream json;
    
    // 获取当前时间戳
    auto now = time(nullptr);
    
    json << "{";
    json << "\"status\":\"success\",";
    
    // 添加数据部分
    json << "\"data\":[";
    for (size_t i = 0; i < result.size(); ++i) {
        json << "{";
        json << "\"word\":\"" << result[i].word << "\",";
        json << "\"tag\":\"" << result[i].tag << "\"";
        
        // 如果有rank权重，添加到JSON中
        if (result[i].rank >= 0) {
            json << ",\"rank\":" << result[i].rank;
        }
        
        json << "}";
        if (i < result.size() - 1) {
            json << ",";
        }
    }
    json << "],";
    
    // 添加元数据
    json << "\"meta\":{";
    json << "\"input_length\":" << input_text.length() << ",";
    json << "\"word_count\":" << result.size() << ",";
    json << "\"timestamp\":" << now;
    json << "},";
    
    // 添加耗时信息
    json << "\"timing\":{";
    json << "\"total_ms\":" << fixed << setprecision(3) << total_time_ms << ",";
    json << "\"qps\":" << fixed << setprecision(2) << (1000.0 / total_time_ms);
    json << "},";
    
    // 添加性能指标
    json << "\"performance\":{";
    json << "\"words_per_second\":" << fixed << setprecision(2) << (result.size() * 1000.0 / total_time_ms) << ",";
    json << "\"chars_per_second\":" << fixed << setprecision(2) << (input_text.length() * 1000.0 / total_time_ms);
    json << "}";
    
    json << "}";
    return json.str();
}

// 添加批量处理的JSON构建函数
string build_batch_result_with_timing_json(const vector<vector<OutputItem>>& results_batch,
                                          const vector<string>& input_texts,
                                          double total_time_ms,
                                          double init_time_ms = 0) {
    ostringstream json;
    
    auto now = time(nullptr);
    
    json << "{";
    json << "\"status\":\"success\",";
    
    // 添加批量数据
    json << "\"data\":[";
    for (size_t i = 0; i < results_batch.size(); ++i) {
        json << "[";
        for (size_t j = 0; j < results_batch[i].size(); ++j) {
            json << "{";
            json << "\"word\":\"" << results_batch[i][j].word << "\",";
            json << "\"tag\":\"" << results_batch[i][j].tag << "\"";
            
            if (results_batch[i][j].rank >= 0) {
                json << ",\"rank\":" << results_batch[i][j].rank;
            }
            
            json << "}";
            if (j < results_batch[i].size() - 1) {
                json << ",";
            }
        }
        json << "]";
        if (i < results_batch.size() - 1) {
            json << ",";
        }
    }
    json << "],";
    
    // 计算总的输入长度和词数
    int total_input_length = 0;
    int total_word_count = 0;
    for (size_t i = 0; i < input_texts.size(); ++i) {
        total_input_length += input_texts[i].length();
        if (i < results_batch.size()) {
            total_word_count += results_batch[i].size();
        }
    }
    
    // 添加元数据
    json << "\"meta\":{";
    json << "\"batch_size\":" << input_texts.size() << ",";
    json << "\"total_input_length\":" << total_input_length << ",";
    json << "\"total_word_count\":" << total_word_count << ",";
    json << "\"timestamp\":" << now;
    json << "},";
    
    // 添加耗时信息
    json << "\"timing\":{";
    json << "\"total_ms\":" << fixed << setprecision(3) << total_time_ms << ",";
    json << "\"average_per_text_ms\":" << fixed << setprecision(3) << (total_time_ms / input_texts.size()) << ",";
    json << "\"qps\":" << fixed << setprecision(2) << (input_texts.size() * 1000.0 / total_time_ms);
    
    if (init_time_ms > 0) {
        json << ",\"initialization_ms\":" << fixed << setprecision(3) << init_time_ms;
    }
    
    json << "},";
    
    // 添加性能指标
    json << "\"performance\":{";
    json << "\"words_per_second\":" << fixed << setprecision(2) << (total_word_count * 1000.0 / total_time_ms) << ",";
    json << "\"chars_per_second\":" << fixed << setprecision(2) << (total_input_length * 1000.0 / total_time_ms) << ",";
    json << "\"throughput_texts_per_second\":" << fixed << setprecision(2) << (input_texts.size() * 1000.0 / total_time_ms);
    json << "}";
    
    json << "}";
    return json.str();
}

int main(int argc, char* argv[]){
    // 统计初始化时间
    auto init_start_time = high_resolution_clock::now();
    
    // 读取命令行参数
    string lac_model_path = "../models/lac_model";
    string rank_model_path = "../models/rank_model";
    string dict_path = "";
    bool json_output = true;
    bool batch_mode = false;
    
    if (argc > 1){
        lac_model_path = argv[1];
    }
    if (argc > 2){
        rank_model_path = argv[2];
    }
    if (argc > 3){
        dict_path = argv[3];
    }
    
    // 检查命令行参数
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--json" || string(argv[i]) == "-j") {
            json_output = true;
        } else if (string(argv[i]) == "--batch" || string(argv[i]) == "-b") {
            batch_mode = true;
        } else if (string(argv[i]) == "--plain" || string(argv[i]) == "-p") {
            json_output = false;
        }
    }

    // 装载LAC模型
    LAC lac(lac_model_path);
    
    // 启用rank模式
    lac.enable_rank_mode(rank_model_path);
    
    // 可选：加载用户词典
    if (dict_path.length() > 1){
        lac.load_customization(dict_path);
    }

    // 统计初始化结束时间
    auto init_end_time = high_resolution_clock::now();
    auto init_duration_ms = duration_cast<microseconds>(init_end_time - init_start_time).count() / 1000.0;
    
    // cout << "LAC模型和Rank模型加载完成，初始化耗时: " << fixed << setprecision(3) << init_duration_ms << " 毫秒" << endl;
    // cout << "模式: " << (batch_mode ? "批量处理" : "单个处理") << " | 输出: " << (json_output ? "JSON格式" : "普通格式") << endl;
    // cout << "参数说明: --json/-j (JSON输出), --plain/-p (普通输出), --batch/-b (批量模式)" << endl;
    
    if (batch_mode) {
        // 批量处理模式
        vector<string> queries;
        string line;
        cout << "批量模式: 请输入多行文本，空行结束输入:" << endl;
        
        while (getline(cin, line)) {
            if (line.empty()) {
                break;
            }
            queries.push_back(line);
        }
        
        if (!queries.empty()) {
            auto start_time = high_resolution_clock::now();
            
            if (json_output) {
                auto results_batch = lac.run_rank(queries);
                auto end_time = high_resolution_clock::now();
                auto duration_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
                
                string json_result = build_batch_result_with_timing_json(results_batch, queries, duration_ms, init_duration_ms);
                cout << json_result << endl;
            } else {
                auto results_batch = lac.run_rank(queries);
                auto end_time = high_resolution_clock::now();
                auto duration_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
                
                for (size_t i = 0; i < results_batch.size(); ++i) {
                    cout << "文本 " << (i + 1) << ": ";
                    for (size_t j = 0; j < results_batch[i].size(); ++j) {
                        cout << results_batch[i][j].word << "/" << results_batch[i][j].tag << "/" << results_batch[i][j].rank << " ";
                    }
                    cout << endl;
                }
                cout << "批量处理耗时: " << fixed << setprecision(3) << duration_ms << " 毫秒" << endl;
                cout << "平均每条: " << fixed << setprecision(3) << (duration_ms / queries.size()) << " 毫秒" << endl;
            }
        }
    } else {
        // 单个处理模式
        string query;
        cout << "请输入文本(Enter退出): ";
        while (getline(cin, query)){
            if (query.empty()) {
                break;
            }

            // 高精度计时
            auto start_time = high_resolution_clock::now();
            
            if (json_output) {
                auto result = lac.run_rank(query);
                auto end_time = high_resolution_clock::now();
                auto duration_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
                
                string json_result = build_result_with_timing_json(result, query, duration_ms);
                cout << json_result << endl;
            } else {
                auto result = lac.run_rank(query);
                auto end_time = high_resolution_clock::now();
                auto duration_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
                
                for (size_t i = 0; i < result.size(); i++){
                    cout << result[i].word << "/" << result[i].tag << "/" << result[i].rank << " ";
                }
                cout << endl;
                cout << "处理耗时: " << fixed << setprecision(3) << duration_ms << " 毫秒" << endl;
            }
            
            cout << "请输入文本(Enter退出): ";
        }
    }
    
    return 0;
}
