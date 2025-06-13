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

#ifndef BAIDU_LAC_LAC_H
#define BAIDU_LAC_LAC_H

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "paddle_inference_api.h"

/* 编码设置 */
enum CODE_TYPE
{
    CODE_GB18030 = 0,
    CODE_UTF8 = 1,
};

struct OutputItem
{
    std::string word;
    std::string tag;
    int rank;  // 添加rank字段

    // 初始化构造函数
    OutputItem() : word(""), tag(""), rank(0) {}
};


#ifndef LAC_CLASS
#define LAC_CLASS

// 前向声明, 去除头文件依赖
class Customization;

class LAC
{
private:
    CODE_TYPE _codetype;
    std::vector<std::vector<size_t>> _lod;
    std::shared_ptr<std::unordered_map<int64_t, std::string>> _id2label_dict;
    std::shared_ptr<std::unordered_map<std::string, std::string>> _q2b_dict;
    std::shared_ptr<std::unordered_map<std::string, int64_t>> _word2id_dict;
    int64_t _oov_id;
    paddle::PaddlePlace _place;
    std::shared_ptr<paddle_infer::Predictor> _predictor;
    std::shared_ptr<paddle_infer::Tensor> _input_tensor;
    std::shared_ptr<paddle_infer::Tensor> _output_tensor;
    std::vector<std::string> _seq_words;
    std::vector<std::vector<std::string>> _seq_words_batch;
    std::vector<std::string> _labels;
    std::vector<OutputItem> _results;
    std::vector<std::vector<OutputItem>> _results_batch;

    // Rank mode properties
    bool _rank_mode;
    std::shared_ptr<paddle_infer::Predictor> _rank_predictor;
    std::shared_ptr<paddle_infer::Tensor> _rank_output_tensor;

    // 添加word_length相关的成员变量
    std::vector<std::vector<int>> _words_length_batch;

public:
    LAC(const std::string& model_path, CODE_TYPE type = CODE_TYPE::CODE_UTF8);
    LAC(LAC&);
    int load_customization(const std::string& customization_file);
    int feed_data(const std::vector<std::string>& querys);
    int parse_targets(const std::vector<std::string>& tags,
                      const std::vector<std::string>& words,
                      std::vector<OutputItem>& result);
    std::vector<OutputItem> run(const std::string& query);
    std::vector<std::vector<OutputItem>> run(const std::vector<std::string>& query);

    // Rank mode methods
    void enable_rank_mode(const std::string& rank_model_path);
    std::vector<OutputItem> run_rank(const std::string& query);
    std::vector<std::vector<OutputItem>> run_rank(const std::vector<std::string>& query);
    int merge_rank_weights(const std::vector<std::vector<std::string>>& tags_for_rank_batch);
    std::string escape_json_string(const std::string& word);
    std::string results_to_json(const std::vector<OutputItem>& results);
    std::string results_to_json(const std::vector<std::vector<OutputItem>>& results_batch);
    int merge_rank_weights_with_word_length(const std::vector<std::vector<std::string>>& tags_for_rank_batch);
    int parse_rank_results(const std::shared_ptr<paddle_infer::Tensor>& rank_tensor, 
                           std::vector<std::vector<OutputItem>>& results);
    std::string run_rank_json(const std::string& query);
    std::string run_rank_json(const std::vector<std::string>& querys);
    std::string run_json(const std::string& query);
    std::string run_json(const std::vector<std::string>& querys);

    std::shared_ptr<Customization> custom;
};
#endif  // LAC_CLASS

#endif  // BAIDU_LAC_LAC_H
