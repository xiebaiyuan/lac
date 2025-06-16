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

#include "lac.h"
#include "lac_util.h"
#include "lac_custom.h"
#include <paddle_inference_api.h>
#include <iostream>
#include <sstream>
#include <iomanip>

/* LAC构造函数：初始化、装载模型和词典 */
LAC::LAC(const std::string& model_path, CODE_TYPE type)
    : _codetype(type),
      _lod(std::vector<std::vector<size_t> >(1)),
      _id2label_dict(new std::unordered_map<int64_t, std::string>),
      _q2b_dict(new std::unordered_map<std::string, std::string>),
      _word2id_dict(new std::unordered_map<std::string, int64_t>),
      _rank_mode(false),
      _input_tensor(nullptr),
      _output_tensor(nullptr),
      _rank_output_tensor(nullptr),
      custom(NULL)
{

    // 装载词典
    std::string word_dict_path = model_path + "/conf/word.dic";
    load_word2id_dict(word_dict_path, *_word2id_dict);
    std::string q2b_dict_path = model_path + "/conf/q2b.dic";
    load_q2b_dict(q2b_dict_path, *_q2b_dict);
    std::string label_dict_path = model_path + "/conf/tag.dic";
    load_id2label_dict(label_dict_path, *_id2label_dict);

    // 使用AnalysisConfig装载模型，会进一步优化模型
    this->_place = paddle::PaddlePlace::kCPU;
    paddle_infer::Config config;
    // config.SwitchIrOptim(false);       // 关闭优化
    // config.EnableMKLDNN();
    config.DisableGpu();
    config.DisableGlogInfo();
    config.SetModel(model_path + "/model");
    // config.SetProgFile(model_path + "/model/__model__");
    // config.SetParamsFile(model_path + "/model/__params__");
    // std::cout << "Load model from: " << model_path  << std::endl;
    // config.SetModel(model_path + "/model/__model__", model_path + "/model/__params__");
    config.SetCpuMathLibraryNumThreads(1);
    config.SwitchUseFeedFetchOps(false);
    this->_predictor = paddle_infer::CreatePredictor(config);
    // this->_predictor = paddle::CreatePredictor<paddle::AnalysisConfig>(config);

    // 初始化输入输出变量
    auto input_names = this->_predictor->GetInputNames();
    this->_input_tensor = this->_predictor->GetInputHandle(input_names[0]);

    // std::cout << "Input tensor name: " << input_names[0] << std::endl;
    auto output_names = this->_predictor->GetOutputNames();
    this->_output_tensor = this->_predictor->GetOutputHandle(output_names[0]);
    
    this->_oov_id = this->_word2id_dict->size() - 1;
    auto word_iter = this->_word2id_dict->find("OOV");
    if (word_iter != this->_word2id_dict->end())
    {
        this->_oov_id = word_iter->second;
    }
    // std::cout << "OOV id: " << this->_oov_id << std::endl;
}

/* 拷贝构造函数，用于多线程重载 */
LAC::LAC(LAC &lac)
    : _codetype(lac._codetype),
      _lod(std::vector<std::vector<size_t> >(1)),
      _id2label_dict(lac._id2label_dict),
      _q2b_dict(lac._q2b_dict),
      _word2id_dict(lac._word2id_dict),
      _oov_id(lac._oov_id),
      _place(lac._place),
      _predictor(lac._predictor->Clone()),
      _input_tensor(nullptr),
      _output_tensor(nullptr),
      _rank_output_tensor(nullptr),
      custom(lac.custom)
{
    auto input_names = this->_predictor->GetInputNames();
    this->_input_tensor = this->_predictor->GetInputHandle(input_names[0]);
    auto output_names = this->_predictor->GetOutputNames();
    this->_output_tensor = this->_predictor->GetOutputHandle(output_names[0]);
}

/* 装载用户词典 */
int LAC::load_customization(const std::string& filename){
    /* 多线程热加载时容易出问题，多个线程共享custom
    if (custom){
        return custom->load_dict(filename);
    }
    */
    custom = std::make_shared<Customization>(filename);
    return 0;
}

/* 将字符串输入转为Tensor */
int LAC::feed_data(const std::vector<std::string> &querys)
{
    // std::cout << "Feed data: " << querys.size() << " queries." << std::endl;
    this->_seq_words_batch.clear();
    this->_lod[0].clear();

    this->_lod[0].push_back(0);
    int shape = 0;
    for (size_t i = 0; i < querys.size(); ++i)
    {
        split_words(querys[i], this->_codetype, this->_seq_words);
        this->_seq_words_batch.push_back(this->_seq_words);
        shape += this->_seq_words.size();
        this->_lod[0].push_back(shape);
    }
    this->_input_tensor->SetLoD(this->_lod);
    this->_input_tensor->Reshape({shape, 1});

    int64_t *input_d = this->_input_tensor->mutable_data<int64_t>(this->_place);
    
    int index = 0;
    for (size_t i = 0; i < this->_seq_words_batch.size(); ++i)
    {
        for (size_t j = 0; j < this->_seq_words_batch[i].size(); ++j)
        {
            // normalization
            std::string word = this->_seq_words_batch[i][j];
            auto q2b_iter = this->_q2b_dict->find(word);
            if (q2b_iter != this->_q2b_dict->end())
            {
                word = q2b_iter->second;
            }

            // get word_id
            int64_t word_id = this->_oov_id;
            auto word_iter = this->_word2id_dict->find(word);
            if (word_iter != this->_word2id_dict->end())
            {
                word_id = word_iter->second;
            }
            input_d[index++] = word_id;
        }
    }
    return 0;
}

/* 对输出的标签进行解码转换为模型输出格式 */
int LAC::parse_targets(
    const std::vector<std::string> &tags,
    const std::vector<std::string> &words,
    std::vector<OutputItem> &result)
{
    result.clear();
    for (size_t i = 0; i < tags.size(); ++i)
    {
        // 若新词，则push_back一个新词，否则append到上一个词中
        if (result.empty() || tags[i].rfind("B") == tags[i].length() - 1 || tags[i].rfind("S") == tags[i].length() - 1)
        {
            OutputItem output_item;
            output_item.word = words[i];
            output_item.tag = tags[i].substr(0, tags[i].length() - 2);
            result.push_back(output_item);
        }
        else
        {
            result[result.size() - 1].word += words[i];
        }
    }
    return 0;
}

std::vector<OutputItem> LAC::run(const std::string &query)
{
    // std::cout << "Run LAC with query: " << query << std::endl;
    std::vector<std::string> query_vector = std::vector<std::string>({query});
    auto result = run(query_vector);
    return result[0];
}

std::vector<std::vector<OutputItem>> LAC::run(const std::vector<std::string> &querys)
{
    // std::cout << "Run LAC with " << querys.size() << " queries." << std::endl;
    this->feed_data(querys);
    // std::cout << "Input tensor shape: " << std::endl;
    this->_predictor->Run();

    // 对模型输出进行解码
    int output_size = 0;
    int64_t *output_d = this->_output_tensor->data<int64_t>(&(this->_place), &output_size);
    
    this->_labels.clear();
    this->_results_batch.clear();
    for (size_t i = 0; i < this->_lod[0].size() - 1; ++i)
    {
        for (size_t j = 0; j < _lod[0][i + 1] - _lod[0][i]; ++j)
        {

            int64_t cur_label_id = output_d[_lod[0][i] + j];
            auto it = this->_id2label_dict->find(cur_label_id);
            this->_labels.push_back(it->second);
        }

        // 装载了用户干预词典，先进行干预处理
        if (custom){
            custom->parse_customization(this->_seq_words_batch[i], this->_labels);
        }

        parse_targets(this->_labels, this->_seq_words_batch[i], this->_results);
        this->_labels.clear();
        this->_results_batch.push_back(this->_results);
    }

    return this->_results_batch;
}

/* 开启Rank模式，加载rank模型 */
void LAC::enable_rank_mode(const std::string& rank_model_path) {
    // 使用AnalysisConfig装载Rank模型
    this->_rank_mode = true;
    paddle_infer::Config rank_config;
    rank_config.DisableGpu();
    rank_config.DisableGlogInfo();
    rank_config.SetModel(rank_model_path + "/model");
    rank_config.SetCpuMathLibraryNumThreads(1);
    rank_config.SwitchUseFeedFetchOps(false);
    this->_rank_predictor = paddle_infer::CreatePredictor(rank_config);
    
    // std::cout << "Rank model loaded from: " << rank_model_path << std::endl;
    auto output_names = this->_rank_predictor->GetOutputNames();
    // std::cout << "Rank model output name: " << output_names[0] << std::endl;
    this->_rank_output_tensor = this->_rank_predictor->GetOutputHandle(output_names[0]);
}

/* Rank模式运行，单个query */
std::vector<OutputItem> LAC::run_rank(const std::string& query) {
    std::vector<std::string> query_vector = std::vector<std::string>({query});
    auto results = run_rank(query_vector);
    return results[0];
}

/* Rank模式运行，批量query */
std::vector<std::vector<OutputItem>> LAC::run_rank(const std::vector<std::string>& querys) {
    if (!this->_rank_mode) {
        std::cerr << "Rank mode not enabled! Please call enable_rank_mode() first." << std::endl;
        return run(querys);
    }
    
    // 首先进行LAC处理 - 重用现有的LAC逻辑
    this->feed_data(querys);
    this->_predictor->Run();
    
    // 获取LAC的输入输出数据
    auto input_lod = this->_input_tensor->lod();
    auto input_shape = this->_input_tensor->shape();
    auto output_lod = this->_output_tensor->lod();
    auto output_shape = this->_output_tensor->shape();
    
    int lac_input_size = 0;
    int lac_output_size = 0;
    int64_t *lac_input_d = this->_input_tensor->data<int64_t>(&(this->_place), &lac_input_size);
    int64_t *lac_output_d = this->_output_tensor->data<int64_t>(&(this->_place), &lac_output_size);
    
    // 准备Rank模型输入
    auto rank_input_names = this->_rank_predictor->GetInputNames();
    if (rank_input_names.size() < 2) {
        std::cerr << "Rank model expects 2 inputs but got " << rank_input_names.size() << std::endl;
        return run(querys);
    }
    
    // 设置Rank模型的两个输入：words和crf_decode（与Python版本一致）
    auto rank_words_input = this->_rank_predictor->GetInputHandle(rank_input_names[0]);
    auto rank_crf_input = this->_rank_predictor->GetInputHandle(rank_input_names[1]);
    
    // 设置第一个输入（words） - 直接复用LAC的输入
    rank_words_input->SetLoD(input_lod);
    rank_words_input->Reshape(input_shape);
    int64_t *rank_words_d = rank_words_input->mutable_data<int64_t>(this->_place);
    std::memcpy(rank_words_d, lac_input_d, lac_input_size * sizeof(int64_t));
    
    // 设置第二个输入（crf_decode） - 直接复用LAC的输出
    rank_crf_input->SetLoD(output_lod);
    rank_crf_input->Reshape(output_shape);
    int64_t *rank_crf_d = rank_crf_input->mutable_data<int64_t>(this->_place);
    std::memcpy(rank_crf_d, lac_output_d, lac_output_size * sizeof(int64_t));
    
    // 运行rank模型
    this->_rank_predictor->Run();
    
    // 处理LAC结果 - 需要保存tags_for_rank用于后续权重合并
    this->_labels.clear();
    this->_results_batch.clear();
    std::vector<std::vector<std::string>> tags_for_rank_batch;
    
    // 按batch处理LAC输出
    for (size_t i = 0; i < output_lod[0].size() - 1; ++i) {
        std::vector<std::string> tags_for_rank;
        
        // 解析当前句子的标签
        for (size_t j = output_lod[0][i]; j < output_lod[0][i + 1]; ++j) {
            int64_t cur_label_id = lac_output_d[j];
            auto it = this->_id2label_dict->find(cur_label_id);
            this->_labels.push_back(it->second);
            tags_for_rank.push_back(it->second);
        }
        
        // 用户自定义词典处理
        if (custom) {
            custom->parse_customization(this->_seq_words_batch[i], this->_labels);
        }
        
        // 解析为最终结果
        parse_targets(this->_labels, this->_seq_words_batch[i], this->_results);
        this->_results_batch.push_back(this->_results);
        tags_for_rank_batch.push_back(tags_for_rank);
        this->_labels.clear();
    }
    
    // 解析并合并rank权重 - 关键步骤
    merge_rank_weights_with_word_length(tags_for_rank_batch);
    
    return this->_results_batch;
}

/* 解析Rank模型的输出并合并到结果中 - 按照Python逻辑实现 */
int LAC::merge_rank_weights_with_word_length(const std::vector<std::vector<std::string>>& tags_for_rank_batch) {
    auto rank_lod = this->_rank_output_tensor->lod();
    if (rank_lod.empty() || rank_lod[0].empty()) {
        std::cerr << "Invalid rank output LOD" << std::endl;
        return -1;
    }
    
    int rank_output_size = 0;
    int64_t *rank_output = this->_rank_output_tensor->data<int64_t>(&(this->_place), &rank_output_size);
    
    size_t batch_size = rank_lod[0].size() - 1;
    
    for (size_t sent_index = 0; sent_index < batch_size && sent_index < this->_results_batch.size(); ++sent_index) {
        size_t begin = rank_lod[0][sent_index];
        size_t end = rank_lod[0][sent_index + 1];
        
        // 获取当前句子的rank权重
        std::vector<int> rank_weights;
        for (size_t j = begin; j < end; ++j) {
            rank_weights.push_back(static_cast<int>(rank_output[j]));
        }
        
        // 处理word_length - 模拟Python版本的word_length处理
        if (sent_index < tags_for_rank_batch.size() && !rank_weights.empty()) {
            const auto& tags = tags_for_rank_batch[sent_index];
            
            // 模拟Python版本的word_length扩展逻辑
            // 这里需要根据segment_tool的处理结果来重新填充权重
            std::vector<int> expanded_weights = rank_weights;
            
            // 如果使用了混合粒度（字词混合），需要处理word_length
            // 这里简化处理，实际应该根据segment_tool的结果来处理
            
            // 按照标签边界合并权重（与Python parse_result逻辑一致）
            std::vector<int> merged_weights;
            for (size_t ind = 0; ind < tags.size() && ind < expanded_weights.size(); ++ind) {
                if (merged_weights.empty() || 
                    tags[ind].find("-B") != std::string::npos || 
                    tags[ind].find("-S") != std::string::npos) {
                    merged_weights.push_back(expanded_weights[ind]);
                } else {
                    // 取最大值作为权重（与Python逻辑一致）
                    merged_weights.back() = std::max(merged_weights.back(), expanded_weights[ind]);
                }
            }
            
            // 将权重分配给结果
            for (size_t i = 0; i < this->_results_batch[sent_index].size() && i < merged_weights.size(); ++i) {
                this->_results_batch[sent_index][i].rank = merged_weights[i];
            }
        }
    }
    
    return 0;
}

// 保留原有的简化版本作为备用
int LAC::merge_rank_weights(const std::vector<std::vector<std::string>>& tags_for_rank_batch) {
    return merge_rank_weights_with_word_length(tags_for_rank_batch);
}

/* 将LAC结果转换为JSON格式字符串 */
std::string LAC::results_to_json(const std::vector<OutputItem>& results) {
    std::ostringstream json;
    json << "[";
    
    for (size_t i = 0; i < results.size(); ++i) {
        json << "{";
        json << "\"word\":\"" << escape_json_string(results[i].word) << "\",";
        json << "\"tag\":\"" << results[i].tag << "\"";
        
        // 如果有rank权重，添加到JSON中
        if (results[i].rank >= 0) {
            json << ",\"rank\":" << results[i].rank;
        }
        
        json << "}";
        if (i < results.size() - 1) {
            json << ",";
        }
    }
    
    json << "]";
    return json.str();
}

/* 将批量LAC结果转换为JSON格式字符串 */
std::string LAC::results_to_json(const std::vector<std::vector<OutputItem>>& results_batch) {
    std::ostringstream json;
    json << "[";
    
    for (size_t i = 0; i < results_batch.size(); ++i) {
        json << results_to_json(results_batch[i]);
        if (i < results_batch.size() - 1) {
            json << ",";
        }
    }
    
    json << "]";
    return json.str();
}

/* JSON字符串转义辅助函数 */
std::string LAC::escape_json_string(const std::string& input) {
    std::ostringstream escaped;
    for (char c : input) {
        switch (c) {
            case '"':  escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    escaped << c;
                }
                break;
        }
    }
    return escaped.str();
}

/* Rank模式运行并返回JSON格式结果 - 单个query */
std::string LAC::run_rank_json(const std::string& query) {
    auto results = run_rank(query);
    return results_to_json(results);
}

/* Rank模式运行并返回JSON格式结果 - 批量query */
std::string LAC::run_rank_json(const std::vector<std::string>& querys) {
    auto results = run_rank(querys);
    return results_to_json(results);
}

/* 普通模式运行并返回JSON格式结果 - 单个query */
std::string LAC::run_json(const std::string& query) {
    auto results = run(query);
    return results_to_json(results);
}

/* 普通模式运行并返回JSON格式结果 - 批量query */
std::string LAC::run_json(const std::vector<std::string>& querys) {
    auto results = run(querys);
    return results_to_json(results);
}
