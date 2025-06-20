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

#include <iostream>
#include<fstream>
#include<iostream>
#include "lac.h"
#include "lac_util.h"
#include "paddle_api.h"

/* LAC构造函数：初始化、装载模型和词典 */
LAC::LAC(std::string model_dict_path, int threads, CODE_TYPE type)
    : _codetype(type),
      _lod(std::vector<std::vector<uint64_t> >(1)),
      _word2id_dict(new std::unordered_map<std::string, int64_t>),
      _q2b_dict(new std::unordered_map<std::string, std::string>),
      _id2label_dict(new std::unordered_map<int64_t, std::string>)
{
    /* 装载词典 */
    std::string word_dict_path = model_dict_path + "/word.dic";
    load_word2id_dict(word_dict_path, *_word2id_dict);
    std::string q2b_dict_path = model_dict_path + "/q2b.dic";
    load_q2b_dict(q2b_dict_path, *_q2b_dict);
    std::string label_dict_path = model_dict_path + "/tag.dic";
    load_id2label_dict(label_dict_path, *_id2label_dict);
    // std::cout << "read word dict succeed" << std::endl;

    paddle::lite_api::MobileConfig config;
    config.set_threads(threads);  // 自行设置多线程

    /* 装载模型 */
    config.set_model_from_file(model_dict_path + "/model.nb");
    this->_predictor = paddle::lite_api::CreatePaddlePredictor(config);

    // std::cout << "load model succeed" << std::endl;

    /* 初始化输入输出变量 */
    this->_input_tensor = this->_predictor->GetInput(0);
    this->_output_tensor = this->_predictor->GetOutput(0);
    this->_oov_id = this->_word2id_dict->size() - 1;
    auto word_iter = this->_word2id_dict->find("OOV");
    if (word_iter != this->_word2id_dict->end())
    {
        this->_oov_id = word_iter->second;
    }

    // std::cout << "init succeed" << std::endl;
}


int LAC::feed_data(const std::vector<std::string> &querys)
{
    this->_seq_words_batch.clear();
    this->_lod[0].clear();

    this->_lod[0].push_back(0);
    long shape = 0;
    for (size_t i = 0; i < querys.size(); ++i)
    {
        split_words(querys[i], this->_codetype, this->_seq_words);
        this->_seq_words_batch.push_back(this->_seq_words);
        shape += this->_seq_words.size();
        this->_lod[0].push_back(shape);
    }
    this->_input_tensor->Resize({shape, 1});
    this->_input_tensor->SetLoD(this->_lod);

    int64_t *input_d = this->_input_tensor->mutable_data<int64_t>();
    int index = 0;
    for (size_t i = 0; i < this->_seq_words_batch.size(); ++i)
    {
        for (size_t j = 0; j < this->_seq_words_batch[i].size(); ++j)
        {
            /* normalization */
            std::string word = this->_seq_words_batch[i][j];
            auto q2b_iter = this->_q2b_dict->find(word);
            if (q2b_iter != this->_q2b_dict->end())
            {
                word = q2b_iter->second;
            }

            /* get word_id */
            int64_t word_id = this->_oov_id; // OOV word
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
        /* 若新词，则push_back一个新词，否则append到上一个词中 */
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

std::vector<OutputItem> LAC::lexer(const std::string &query)
{

    std::vector<std::string> query_vector = std::vector<std::string>({query});

    auto result = lexer(query_vector);
    return result[0];
}

std::vector<std::vector<OutputItem>> LAC::lexer(const std::vector<std::string> &querys)
{

    this->feed_data(querys);
    this->_predictor->Run();

    /* 对模型输出进行解码 */
    int output_size = 0;
    const int64_t *output_d = this->_output_tensor->data<int64_t>();
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
        parse_targets(this->_labels, this->_seq_words_batch[i], this->_results);
        this->_labels.clear();

        _results_batch.push_back(this->_results);
    }

    return this->_results_batch;
}
