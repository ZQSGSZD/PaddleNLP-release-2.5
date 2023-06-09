/*
 * Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Decoder transformer
 **/

#pragma once

#include <cuda_runtime.h>

#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/utils/arguments.h"
#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/functions.h"

namespace fastertransformer {

template <OperationType OpType_>
class T5DecodingSampling {
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct T5SamplingArguments args_;
  TensorParallelParam t_parallel_param_;
  LayerParallelParam l_parallel_param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;

  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *decoder_normed_result_buf_;
  DataType_ *embedding_buf_;
  DataType_ *trans_out_buf_;
  DataType_ *lm_normed_result_buf_;
  DataType_ *logits_buf_;
  int *word_ids_buf_;
  bool *finished_buf_;
  int *h_trg_length_;

  DataType_ *relative_attention_bias_;

  void *buf_;
  int *finished_count_buf_;
  bool *h_finished_buf_;

  void *topk_workspace_ = nullptr;
  size_t topk_workspace_size_ = 0;
  void *topp_workspace_ = nullptr;
  size_t topp_workspace_size_ = 0;
  void *cublas_workspace_ = nullptr;
  curandState_t *curandstate_buf_;
  int *topp_id_vals_buf_;
  int *topp_offset_buf_;
  int *begin_topp_offset_buf_;

  DataType_ *padded_embedding_kernel;
  DataType_ *padded_embedding_bias;

public:
  T5DecodingSampling(const IAllocator &allocator,
                     const int batch_size,
                     const int seq_len,
                     const int head_num,
                     const int size_per_head,
                     const int vocab_size,
                     const int decoder_layers,
                     const int memory_hidden_units,
                     const int memory_max_seq_len,
                     const int start_id,
                     const int end_id,
                     const int candidate_num = 0,
                     const float probability_threshold = 0.0,
                     const int is_fuse_qkv = false,
                     const bool normalization_before = true,
                     const ActivationType act = ActivationType::RELU,
                     const float temperature = 1.0,
                     const float repeat_penalty = 1.0,
                     const int min_length = 0,
                     const int inner_coeff = 4,
                     const int inner_size = -1,
                     const int seed = -1,
                     const int tensor_para_size = 1,
                     const int layer_para_size = 1,
                     const int num_bucket = -1,
                     const int max_distance = 128,
                     const bool tie_word_embeddings = true,
                     const bool use_gated = false)
      : allocator_(allocator) {
    args_.batch_size_ = batch_size;
    args_.seq_len_ = seq_len;
    args_.memory_max_seq_len_ = memory_max_seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.candidate_num_ = candidate_num;
    args_.probability_threshold_ = probability_threshold;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;
    args_.normalization_before_ = normalization_before;
    args_.act_ = act;

    args_.temperature_ = temperature;
    args_.repeat_penalty_ = repeat_penalty;

    args_.min_length_ = min_length;
    args_.seed_ = seed;

    args_.num_bucket_ = num_bucket;
    args_.max_distance_ = max_distance;

    // For models without parallel
    if (l_parallel_param_.layers_per_group == 0) {
      l_parallel_param_.layers_per_group = decoder_layers;
    }

    if (std::is_same<DataType_, float>::value)
      args_.vocab_size_padded_ = vocab_size;
    else if (std::is_same<DataType_, half>::value)
      args_.vocab_size_padded_ = (int)(ceil(vocab_size / 8.)) * 8;

    if (args_.candidate_num_ == 0 && args_.probability_threshold_ == 0.0) {
      printf(
          "[ERROR] Candidate_num for topk is 0 and probability threshold for "
          "top p is 0.0 \n");
      exit(-1);
    } else if (args_.candidate_num_ != 0 &&
               args_.probability_threshold_ != 0.0) {
      printf(
          "[ERROR] Candidate_num for topk is not 0 and probability threshold "
          "for top p is not 0.0 \n");
      exit(-1);
    }
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    K_cache_ = new DataType_ *[1];
    V_cache_ = new DataType_ *[1];

    K_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    V_mem_cache_ = new DataType_ *[args_.decoder_layers_];

    decoder_ = new OpenDecoder<OpType_>(head_num,
                                        size_per_head,
                                        memory_hidden_units,
                                        is_fuse_qkv,
                                        normalization_before,
                                        args_.act_,
                                        inner_coeff,
                                        inner_size,
                                        use_gated);
    decoder_->set_max_batch_size(batch_size);

    size_t from_tensor_size =
        args_.batch_size_ * args_.hidden_units_;                   // type T
    size_t decoder_workspace_size = decoder_->getWorkspaceSize();  // type T
    size_t decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.hidden_units_;  // type T

    size_t cache_size = (args_.batch_size_ * args_.seq_len_ *
                         args_.hidden_units_);  // type T
    size_t mem_cache_size = (args_.batch_size_ * memory_max_seq_len *
                                   args_.hidden_units_);  // type T
    if (tensor_para_size != 1) {                          // tensor parallel
      cache_size /= tensor_para_size;
      mem_cache_size /= tensor_para_size;
    }

    int relative_attention_bias_size =
        (args_.seq_len_ + 1) * (args_.seq_len_ + 1) * head_num;

    size_t logits_buf_size =
        args_.batch_size_ * args_.vocab_size_padded_;  // type T

    size_t word_ids_buf_size = args_.batch_size_;               // type int
    size_t finished_buf_size = args_.batch_size_;               // type bool
    size_t finished_count_size = (size_t)(ceil(1 / 32.)) * 32;  // type int

    int topp_id_vals_buf_size =
        args_.batch_size_ * args_.vocab_size_padded_;  // type int
    int topp_offset_buf_size = args_.batch_size_ + 1;  // type int
    size_t begin_topp_offset_buf_size = topp_offset_buf_size;
    size_t curandState_size = args_.batch_size_;
    size_t padded_embedding_kernel_size =
        args_.hidden_units_ * args_.vocab_size_padded_;
    size_t padded_embedding_bias_size = args_.vocab_size_padded_;
    if (std::is_same<DataType_, float>::value ||
        (std::is_same<DataType_, half>::value &&
         args_.vocab_size_ == args_.vocab_size_padded_)) {
      padded_embedding_kernel_size = 0;
      padded_embedding_bias_size = 0;
    }

    // prevent memory misalinged address
    logits_buf_size = (size_t)(ceil(logits_buf_size / 4.)) * 4;
    word_ids_buf_size = (size_t)(ceil(word_ids_buf_size / 4.)) * 4;
    finished_buf_size = (size_t)(ceil(finished_buf_size / 32.)) * 32;

    topp_id_vals_buf_size = (size_t)(ceil(topp_id_vals_buf_size / 4.)) * 4;
    topp_offset_buf_size = (size_t)(ceil(topp_offset_buf_size / 4.)) * 4;
    begin_topp_offset_buf_size = topp_offset_buf_size;

    topP_sampling_kernel_kernelLauncher_v2(topp_workspace_,
                                           topp_workspace_size_,
                                           logits_buf_,
                                           topp_id_vals_buf_,
                                           topp_offset_buf_,
                                           begin_topp_offset_buf_,
                                           finished_buf_,
                                           curandstate_buf_,
                                           args_,
                                           nullptr,
                                           nullptr,
                                           args_.vocab_size_padded_,
                                           0,
                                           args_.batch_size_);

    topK_sampling_kernel_kernelLauncher_v2(topk_workspace_,
                                           topk_workspace_size_,
                                           logits_buf_,
                                           nullptr,
                                           nullptr,
                                           finished_buf_,
                                           curandstate_buf_,
                                           args_,
                                           0,
                                           args_.batch_size_);

    size_t datatype_buf_size =
        from_tensor_size * 2 + decoder_workspace_size +
        (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ +
        decoder_normed_result_buffer_size * 3;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        ((sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0) +
        sizeof(DataType_) * (datatype_buf_size + logits_buf_size) +
        sizeof(DataType_) *
            (padded_embedding_kernel_size + padded_embedding_bias_size) +
        sizeof(int) * word_ids_buf_size + sizeof(bool) * finished_buf_size +
        sizeof(int) * finished_count_size +
        sizeof(int) * (topp_id_vals_buf_size + 2 * topp_offset_buf_size) +
        topp_workspace_size_ + topk_workspace_size_ +
        sizeof(DataType_) * relative_attention_bias_size +
        curandState_size * sizeof(curandState_t)));

    if (sizeof(DataType_) == sizeof(half)) {
      cublas_workspace_ = buf_;
      from_tensor_[0] =
          (DataType_ *)((char *)cublas_workspace_ + CUBLAS_WORKSPACE_SIZE);
    } else {
      cublas_workspace_ = nullptr;
      from_tensor_[0] = (DataType_ *)buf_;
    }
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < args_.decoder_layers_; ++i) {
      K_mem_cache_[i] =
          from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size +
                        i * mem_cache_size * 2 + mem_cache_size;
    }

    /* We use two-way buffer since we have to update KV buf at the end of each
     * step. */
    K_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size +
                  0 * cache_size * args_.decoder_layers_;
    V_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size +
                  1 * cache_size * args_.decoder_layers_;

    decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_;

    decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    // Used for post-norm.
    embedding_buf_ = (decoder_buf_ + decoder_workspace_size);

    logits_buf_ =
        decoder_normed_result_buf_ + decoder_normed_result_buffer_size;
    word_ids_buf_ = (int *)(logits_buf_ + logits_buf_size);
    finished_buf_ = (bool *)(word_ids_buf_ + word_ids_buf_size);
    finished_count_buf_ = (int *)(finished_buf_ + finished_buf_size);

    relative_attention_bias_ =
        (DataType_ *)(finished_count_buf_ + finished_count_size);

    topp_id_vals_buf_ =
        (int *)(relative_attention_bias_ + relative_attention_bias_size);
    begin_topp_offset_buf_ = (int *)(topp_id_vals_buf_ + topp_id_vals_buf_size);
    topp_offset_buf_ =
        (int *)(begin_topp_offset_buf_ + begin_topp_offset_buf_size);
    topp_workspace_ = (void *)(topp_offset_buf_ + topp_offset_buf_size);
    topk_workspace_ = (void *)((char *)topp_workspace_ + topp_workspace_size_);
    padded_embedding_kernel =
        (DataType_ *)((char *)topk_workspace_ + topk_workspace_size_);
    padded_embedding_bias =
        (DataType_ *)(padded_embedding_kernel + padded_embedding_kernel_size);
    curandstate_buf_ =
        (curandState_t *)(padded_embedding_bias + padded_embedding_bias_size);

    h_finished_buf_ = new bool[finished_buf_size];
    h_trg_length_ = new int[args_.batch_size_];

    int isConfigExist = access("decoding_gemm_config.in", 0);
    if (isConfigExist == -1) {
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    } else {
      readAlgoFromConfig(cublasAlgoMap_, 1);
      // check that the gemm_config setting is runnable
      for (auto iter = cublasAlgoMap_.begin(); iter != cublasAlgoMap_.end();
           iter++) {
        int algoId = iter->second.algoId;
        int stages = iter->second.stages;
        // only check for cublas
        if (stages != -1) continue;
        if (Traits_::OpType == OperationType::FP32) {
          if (algoId > CUBLAS_GEMM_ALGO23 || algoId < CUBLAS_GEMM_DEFAULT) {
            // the algorithm is not for FP32
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n",
                   algoId);
            exit(-1);
          }
        } else {
          if (algoId > CUBLAS_GEMM_ALGO15_TENSOR_OP ||
              algoId < CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
            // the algorithm is not for FP16
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n",
                   algoId);
            exit(-1);
          }
        }
      }
    }
  }

  void set_tensor_parallel_param(const TensorParallelParam param) {
    t_parallel_param_ = param;
    decoder_->set_tensor_parallel_param(param);
  }

  void set_layer_parallel_param(const LayerParallelParam param) {
    l_parallel_param_ = param;
    decoder_->set_layer_parallel_param(param);
  }

  void forward(const DecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = args_.batch_size_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_padded_;
    const DataType_ *embedding_kernel_ptr = nullptr;
    const DataType_ *embedding_bias_ptr = nullptr;

    int min_trg_len = 0;
    int max_trg_len = 0;

    if (decoding_params.trg_word) {
      cudaMemcpy(h_trg_length_,
                 decoding_params.trg_length,
                 sizeof(int) * args_.batch_size_,
                 cudaMemcpyDeviceToHost);
      min_trg_len = h_trg_length_[0];
      max_trg_len = h_trg_length_[0];

      for (int i = 1; i < args_.batch_size_; ++i) {
        min_trg_len = std::min(min_trg_len, h_trg_length_[i]);
        max_trg_len = std::max(max_trg_len, h_trg_length_[i]);
      }
    }

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
    */
    if (decoding_params.output_scores) {
      cudaMemsetAsync(decoding_params.output_scores, 0, sizeof(float) * m);
    }
    if (args_.candidate_num_ != 0) {
      sampling_init_kernelLauncher(finished_buf_,
                                   decoding_params.sequence_length,
                                   word_ids_buf_,
                                   args_.start_id_,
                                   args_.batch_size_,
                                   decoding_params.stream);
    } else if (args_.probability_threshold_ != 0.0) {
      topp_initialization_kernelLauncher_v2(finished_buf_,
                                            decoding_params.sequence_length,
                                            word_ids_buf_,
                                            topp_id_vals_buf_,
                                            topp_offset_buf_,
                                            begin_topp_offset_buf_,
                                            args_.vocab_size_padded_,
                                            args_,
                                            decoding_params.stream);
    }
    ker_curand_setupLauncher(curandstate_buf_, args_, decoding_params.stream);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    build_relative_attention_bias_launcher(
        relative_attention_bias_,
        decoding_params.self_relative_attention_bias_weight,
        args_.head_num_,
        (args_.seq_len_ + 1),
        args_.num_bucket_,
        false,
        args_.max_distance_,
        decoding_params.stream);

    if (std::is_same<DataType_, float>::value ||
        (std::is_same<DataType_, half>::value &&
         args_.vocab_size_ == args_.vocab_size_padded_)) {
      embedding_kernel_ptr =
          (const DataType_ *)decoding_params.embedding_kernel;
      embedding_bias_ptr = (const DataType_ *)decoding_params.embedding_bias;
    } else if (std::is_same<DataType_, half>::value) {
      kernel_padding_kernelLauncher(padded_embedding_kernel,
                                    decoding_params.embedding_kernel,
                                    args_.hidden_units_,
                                    args_.vocab_size_,
                                    args_.vocab_size_padded_,
                                    decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      bias_padding_kernelLauncher(padded_embedding_bias,
                                  decoding_params.embedding_bias,
                                  args_.vocab_size_,
                                  args_.vocab_size_padded_,
                                  decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      embedding_kernel_ptr = padded_embedding_kernel;
      embedding_bias_ptr = padded_embedding_bias;
    }

    // TODO(guosheng): move cache offset into for loop for pipeline parallel
    size_t cache_size = (args_.batch_size_ * args_.seq_len_ *
                         t_parallel_param_.local_hidden_units_);  // type T

    const int local_batch = l_parallel_param_.local_batch_size;
    for (uint step = 1; step <= args_.seq_len_; ++step) {

      words_embeddings_kernel_launcher(from_tensor_[0],
                                       decoding_params.embedding_table,
                                       word_ids_buf_,
                                       m,
                                       args_.hidden_units_,
                                       decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      int from_id, out_id;
      for (int layer = 0; layer < args_.decoder_layers_; ++layer) {
        if (l_parallel_param_.is_valid(layer)) {
          /*
             For the first layer (layer-0), from_id is 0. We also stored the
             embedding lookup
             result in from_tensor_[0]
           */
          from_id = layer & 0x1;
          out_id = 1 - from_id;

          /*
            We use one decoder_ object to process multiple decoder layers.

            At the beginning of each decoder layer, we initialize the decoder
            object
            with corresponding weights and decoder_buf_.

            The decoder_buf_ is reused.
          */
          decoder_->initialize(param[layer], decoder_buf_, cublas_workspace_);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          decoder_->forward(from_tensor_[from_id],
                            decoding_params.memory_tensor,
                            K_cache_[0] + layer * cache_size,
                            V_cache_[0] + layer * cache_size,
                            K_mem_cache_[layer],
                            V_mem_cache_[layer],
                            decoding_params.memory_sequence_length,
                            from_tensor_[out_id],
                            step,
                            args_.seq_len_,
                            true, /* is_cross_attention */
                            finished_buf_,
                            relative_attention_bias_,
                            true);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
        }
      }

      if (step > min_trg_len) {
        DataType_ alpha = (DataType_)1.0f;
        DataType_ beta = (DataType_)0.0f;

        t5_layer_norm(from_tensor_[out_id],
                      decoding_params.layernorm.gamma,
                      decoding_params.layernorm.beta,
                      decoder_normed_result_buf_,
                      m,
                      k,
                      decoding_params.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (args_.tie_word_embeddings_) {
          alpha = (DataType_) pow((float)(k), -0.5);
        }

        cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                            decoding_params.cublas_handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            n,
                                            m,
                                            k,
                                            &alpha,
                                            embedding_kernel_ptr,
                                            AType_,
                                            n,
                                            decoder_normed_result_buf_,
                                            BType_,
                                            k,
                                            &beta,
                                            logits_buf_,
                                            CType_,
                                            n,
                                            decoding_params.stream,
                                            cublasAlgoMap_,
                                            cublas_workspace_);

        if (decoding_params.logits_mask ||
            (args_.min_length_ != 0 && step <= args_.min_length_)) {
          apply_logits_mask_kernelLauncher(
              logits_buf_,
              finished_buf_,
              args_.batch_size_,
              1,
              args_.vocab_size_padded_,
              args_.vocab_size_,
              decoding_params.stream,
              decoding_params.logits_mask,
              (args_.min_length_ != 0 && step <= args_.min_length_),
              args_.end_id_);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
        }

        if (args_.temperature_ != 1.0) {
          apply_temperature_penalty_kernelLauncher(
              logits_buf_,
              (DataType_)args_.temperature_,
              args_.batch_size_,
              args_.vocab_size_,
              n,
              decoding_params.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
        }

        if (args_.candidate_num_ != 0) {
          // top k sampling
          if (decoding_params.output_scores) {
            softmax_kernelLauncher(logits_buf_,
                                   embedding_bias_ptr,
                                   args_.end_id_,
                                   finished_buf_,
                                   m,
                                   n,
                                   n,
                                   decoding_params.stream);

            // Return Score.
            topK_sampling_kernel_kernelLauncher_v3(
                topk_workspace_,
                topk_workspace_size_,
                logits_buf_,
                decoding_params.output_ids + (step - 1) * args_.batch_size_,
                decoding_params.sequence_length,
                decoding_params.output_scores,
                finished_buf_,
                curandstate_buf_,  // used as random number
                args_,
                decoding_params.stream,
                args_.batch_size_);
          } else {
            update_logits_without_softmax(logits_buf_,
                                          embedding_bias_ptr,
                                          args_.end_id_,
                                          finished_buf_,
                                          m,
                                          n,
                                          decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            topK_sampling_kernel_kernelLauncher_v2(
                topk_workspace_,
                topk_workspace_size_,
                logits_buf_,
                decoding_params.output_ids + (step - 1) * args_.batch_size_,
                decoding_params.sequence_length,
                finished_buf_,
                curandstate_buf_,  // used as random number
                args_,
                decoding_params.stream,
                args_.batch_size_);
          }
        } else if (args_.probability_threshold_ != 0.0) {
          // top p sampling
          softmax_kernelLauncher(logits_buf_,
                                 embedding_bias_ptr,
                                 args_.end_id_,
                                 finished_buf_,
                                 m,
                                 n,
                                 n,
                                 decoding_params.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          if (decoding_params.output_scores) {
            topP_sampling_kernel_kernelLauncher_v3(
                topp_workspace_,
                topp_workspace_size_,
                logits_buf_,
                topp_id_vals_buf_,
                topp_offset_buf_,
                begin_topp_offset_buf_,
                finished_buf_,
                curandstate_buf_,
                args_,
                decoding_params.output_ids + (step - 1) * args_.batch_size_,
                decoding_params.sequence_length,
                decoding_params.output_scores,
                n,
                decoding_params.stream,
                args_.batch_size_);
          } else {
            topP_sampling_kernel_kernelLauncher_v2(
                topp_workspace_,
                topp_workspace_size_,
                logits_buf_,
                topp_id_vals_buf_,
                topp_offset_buf_,
                begin_topp_offset_buf_,
                finished_buf_,
                curandstate_buf_,
                args_,
                decoding_params.output_ids + (step - 1) * args_.batch_size_,
                decoding_params.sequence_length,
                n,
                decoding_params.stream,
                args_.batch_size_);
          }
        }
      }

      if (step <= max_trg_len) {
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        update_with_force_decodingLauncher(
            decoding_params.trg_word,
            decoding_params.trg_length,
            finished_buf_,
            word_ids_buf_,
            (step > min_trg_len) ? nullptr : decoding_params.sequence_length,
            (int *)nullptr,
            (int *)nullptr,
            decoding_params.output_ids + (step - 1) * args_.batch_size_,
            (DataType_ *)nullptr,
            false,
            args_.batch_size_,
            1,
            max_trg_len,
            step,
            decoding_params.stream);
      } else {
        word_ids_buf_ =
            decoding_params.output_ids + (step - 1) * args_.batch_size_;
      }

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      if (step > max_trg_len) {
        // TODO Find a better method to check the is_finished
        cudaMemcpy(h_finished_buf_,
                   finished_buf_,
                   sizeof(bool) * args_.batch_size_,
                   cudaMemcpyDeviceToHost);
        uint sum = 0;
        for (uint i = 0; i < args_.batch_size_; i++) {
          sum += (int)h_finished_buf_[i];
        }
        if (sum == args_.batch_size_) break;
      }
    }
  }

  virtual ~T5DecodingSampling() {
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] K_mem_cache_;
    delete[] V_mem_cache_;
    delete[] h_finished_buf_;
    delete[] h_trg_length_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

}  // namespace fastertransformer
