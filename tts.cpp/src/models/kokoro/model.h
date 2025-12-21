#pragma once

#include <cstdlib>

#include "../../tokenizer.h"
#include "../../tts_model.h"
#include "models/loaders.h"
#include "phonemizer.h"

extern const struct kokoro_model_loader final : tts_model_loader {
    explicit kokoro_model_loader();

    unique_ptr<tts_generation_runner> from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                                bool cpu_only, const generation_configuration & config) const override;
} kokoro_loader;

// 说明：
// - Kokoro 的 voice id 通常用首字母标记语言类别（例如部分语音包用 'z' 表示中文）。
// - 本项目当前目标为中/英，音素化侧仅依赖：
//   - TTS.cpp 内置英文 phonemizer（从 GGUF 中读取规则/词典）
//   - zh_frontend 中文前端

struct lstm_cell {
	// 说明：LSTM 的 gate 融合权重（CPU 优化）。
	// - 原实现每个 gate 都做一次 mul_mat（共 8 次：4 个输入投影 + 4 个隐状态投影），并逐步 concat 输出，CPU 上开销很大。
	// - 这里将 I/F/G/O 四个 gate 在输出维拼成一个大矩阵：
	//   - fused_w_x: [in_dim, 4*hidden]   （输入投影，按 I/F/G/O 顺序拼接）
	//   - fused_w_h: [hidden, 4*hidden]   （隐状态投影，按 I/F/G/O 顺序拼接）
	//   - fused_b  : [4*hidden]           （bias 直接预先做 b_x + b_h，推理时少一次 add）
	// - Vulkan 路径默认不启用该融合（避免额外拷贝/兼容性风险）。
	struct fused_gates {
		ggml_tensor * w_x = nullptr;
		ggml_tensor * w_h = nullptr;
		ggml_tensor * b   = nullptr;
	};

	std::vector<ggml_tensor*> weights;
	std::vector<ggml_tensor*> biases;
	std::vector<ggml_tensor*> reverse_weights;
	std::vector<ggml_tensor*> reverse_biases;

	// 正向/反向（bidirectional）各一份融合后的 gate 权重。
	fused_gates fused;
	fused_gates fused_reverse;
};

struct lstm {
	std::vector<ggml_tensor*> hidden;
	std::vector<ggml_tensor*> states;

	bool bidirectional = false;
	std::vector<lstm_cell*> cells;
};

struct duration_predictor_layer {
	lstm * rnn;
	struct ggml_tensor * ada_norm_gamma_weight;
	struct ggml_tensor * ada_norm_gamma_bias;
	struct ggml_tensor * ada_norm_beta_weight;
	struct ggml_tensor * ada_norm_beta_bias;
};

struct ada_residual_conv_block {
	struct ggml_tensor * conv1;
	struct ggml_tensor * conv1_bias;
	struct ggml_tensor * conv2;
	struct ggml_tensor * conv2_bias;
	struct ggml_tensor * norm1_gamma;
	struct ggml_tensor * norm1_gamma_bias;
	struct ggml_tensor * norm1_beta;
	struct ggml_tensor * norm1_beta_bias;
	struct ggml_tensor * norm2_gamma;
	struct ggml_tensor * norm2_gamma_bias;
	struct ggml_tensor * norm2_beta;
	struct ggml_tensor * norm2_beta_bias;
	struct ggml_tensor * pool = nullptr;
	struct ggml_tensor * pool_bias = nullptr;
	struct ggml_tensor * upsample = nullptr;
	struct ggml_tensor * upsample_bias = nullptr;
};

struct duration_predictor {
	struct ggml_tensor * albert_encode;
	struct ggml_tensor * albert_encode_bias;
	std::vector<duration_predictor_layer*> layers;
	lstm * duration_proj_lstm;
	struct ggml_tensor * duration_proj;
	struct ggml_tensor * duration_proj_bias;
	struct ggml_tensor * n_proj_kernel;
	struct ggml_tensor * n_proj_bias;
	struct ggml_tensor * f0_proj_kernel;
	struct ggml_tensor * f0_proj_bias;
	lstm * shared_lstm;
	std::vector<ada_residual_conv_block*> f0_blocks;
	std::vector<ada_residual_conv_block*> n_blocks;
};

struct kokoro_text_encoder_conv_layer {
	struct ggml_tensor * norm_gamma;
	struct ggml_tensor * norm_beta;
	struct ggml_tensor * conv_weight;
	struct ggml_tensor * conv_bias;
};

struct kokoro_text_encoder {
	struct ggml_tensor * embd;
	std::vector<kokoro_text_encoder_conv_layer*> conv_layers;
	lstm * out_lstm;
};

struct kokoro_generator_residual_block {
	std::vector<uint32_t> conv1_dilations;
	std::vector<uint32_t> conv1_paddings;

	std::vector<ggml_tensor*> adain1d_1_gamma_weights;
	std::vector<ggml_tensor*> adain1d_2_gamma_weights;
	std::vector<ggml_tensor*> adain1d_1_gamma_biases;
	std::vector<ggml_tensor*> adain1d_2_gamma_biases;
	std::vector<ggml_tensor*> adain1d_1_beta_weights;
	std::vector<ggml_tensor*> adain1d_2_beta_weights;
	std::vector<ggml_tensor*> adain1d_1_beta_biases;
	std::vector<ggml_tensor*> adain1d_2_beta_biases;
	std::vector<ggml_tensor*> input_alphas;
	std::vector<ggml_tensor*> output_alphas;
	std::vector<ggml_tensor*> convs1_weights;
	std::vector<ggml_tensor*> convs1_biases;
	std::vector<ggml_tensor*> convs2_weights;
	std::vector<ggml_tensor*> convs2_biases;
};

struct kokoro_noise_residual_block {
	uint32_t input_conv_stride;
	uint32_t input_conv_padding;

	struct ggml_tensor * input_conv;
	struct ggml_tensor * input_conv_bias;
	struct kokoro_generator_residual_block * res_block;
};

struct kokoro_generator_upsample_block {
	uint32_t padding;
	uint32_t stride;

	// these are just conv transpose layers
	struct ggml_tensor * upsample_weight;
	struct ggml_tensor * upsample_bias;
};

struct kokoro_generator {
	// unfortunately the squared sum of the windows needs to be computed dynamically per run because it is dependent
	// on the sequence size of the generation and the hop is typically less than half the size of our window.
	struct ggml_tensor * window;
	// 说明：CPU 可读的窗函数张量副本（用于自定义 STFT/ISTFT 回退路径）。
	// - 当权重位于 Vulkan 设备内存时，window->data 不可被 CPU 侧自定义算子直接读取；
	// - 回退到 GGML_OP_CUSTOM 的 stft/istft 时必须使用该副本，否则会出现“电流声/人声消失”等异常输出。
	struct ggml_tensor * window_cpu;
	// 说明：CPU 侧缓存的窗函数数据（用于生成 window_sq_sum），避免直接读取 GPU buffer。
	std::vector<float> window_host;

	struct ggml_tensor * m_source_weight;
	struct ggml_tensor * m_source_bias;
	struct ggml_tensor * out_conv_weight;
	struct ggml_tensor * out_conv_bias;
	std::vector<kokoro_noise_residual_block*> noise_blocks;
	std::vector<kokoro_generator_residual_block*> res_blocks;
	std::vector<kokoro_generator_upsample_block*> ups;
};

struct kokoro_decoder {
	struct ggml_tensor * f0_conv;
	struct ggml_tensor * f0_conv_bias;
	struct ggml_tensor * n_conv;
	struct ggml_tensor * n_conv_bias;
	struct ggml_tensor * asr_conv;
	struct ggml_tensor * asr_conv_bias;
	std::vector<ada_residual_conv_block*> decoder_blocks;
	ada_residual_conv_block* encoder_block;
	kokoro_generator * generator;
};

struct albert_layer {
	struct ggml_tensor * ffn;
	struct ggml_tensor * ffn_out;
	struct ggml_tensor * ffn_bias;
	struct ggml_tensor * ffn_out_bias;
	struct ggml_tensor * layer_output_norm_weight;
	struct ggml_tensor * layer_output_norm_bias;
	struct ggml_tensor * q;
	struct ggml_tensor * k;
	struct ggml_tensor * v;
	struct ggml_tensor * o;
	struct ggml_tensor * q_bias;
	struct ggml_tensor * k_bias;
	struct ggml_tensor * v_bias;
	struct ggml_tensor * o_bias;
	struct ggml_tensor * attn_norm_weight;
	struct ggml_tensor * attn_norm_bias;
};

struct kokoro_model : tts_model {
	// standard configruation for Kokoro's Albert model
	// tokenization
	uint32_t bos_token_id = 0;
	uint32_t eos_token_id = 0;
	uint32_t space_token_id = 16;
	// duration prediction
	uint32_t max_context_length = 512;
	uint32_t vocab_size = 178;
	uint32_t hidden_size = 768;
	uint32_t n_attn_heads = 12;
	uint32_t n_layers = 1;
	uint32_t n_recurrence = 12;
	uint32_t head_size = 64;
	uint32_t duration_hidden_size = 512;
	uint32_t up_sampling_factor;
	float upsample_scale = 300.0f;
	float scale = 0.125f;

	// standard configuration for duration prediction
	uint32_t f0_n_blocks = 3;
	uint32_t n_duration_prediction_layers = 3;
	// while it is technically possible for the duration predictor to assign 50 values per token there is no practical need to
	// allocate that many items to the sequence as it is impossible for all tokens to require such long durations and each
	// allocation increases node allocation size by O(N)
	uint32_t max_duration_per_token = 20;
	uint32_t style_half_size = 128;

	// standard text encoding configuration
	uint32_t n_conv_layers = 3;

	// standard decoder configuration
	uint32_t n_kernels = 3;
	uint32_t n_upsamples = 2;
	uint32_t n_decoder_blocks = 4;
	uint32_t n_res_blocks = 6;
	uint32_t n_noise_blocks = 2;
	uint32_t out_conv_padding = 3;
	uint32_t post_n_fft = 11;
	uint32_t true_n_fft = 20;
	uint32_t stft_hop = 5;
	uint32_t harmonic_num = 8;
	float sin_amp = 0.1f;
	float noise_std = 0.003f;
	float voice_threshold = 10.0f;
	float sample_rate = 24000.0f;
	std::string window = "hann";

	// It is really annoying that ggml doesn't allow using non ggml tensors as the operator for simple math ops.
	// This is just the constant defined above as a tensor.
	struct ggml_tensor * n_kernels_tensor;

	// Kokoro loads albert with use_pooling = true but doesn't use the pooling outputs.
	bool uses_pooling = false;
	bool static_token_types = true;

	std::map<std::string, struct ggml_tensor *> voices;

	// Albert portion of the model
	struct ggml_tensor * embd_hidden;
	struct ggml_tensor * embd_hidden_bias;
	struct ggml_tensor * token_type_embd = nullptr;
	struct ggml_tensor * token_embd;
	struct ggml_tensor * position_embd;
	struct ggml_tensor * input_norm_weight;
	struct ggml_tensor * input_norm_bias;
	struct ggml_tensor * static_token_type_values = nullptr;
	struct ggml_tensor * pool = nullptr;
	struct ggml_tensor * pool_bias = nullptr;
	std::vector<albert_layer*> layers;

	struct ggml_tensor * harmonic_sampling_norm = nullptr; // a static 1x9 harmonic multiplier
	struct ggml_tensor * sampling_factor_scalar = nullptr; // a static scalar
	struct ggml_tensor * sqrt_tensor = nullptr; // static tensor for constant division
	// 说明：STFT/ISTFT 用的预计算基矩阵（conv 形式），用于 Vulkan 完整图执行。
	struct ggml_tensor * stft_forward_basis = nullptr;
	struct ggml_tensor * stft_inverse_basis = nullptr;
    // 说明：部分 Vulkan 设备/驱动在 STFT/ISTFT（conv 版）上可能出现音质问题（如“金属音”）。
    // 为了支持“仅把 STFT/ISTFT 固定到 CPU，其余仍走 Vulkan”的混合方案，这里额外保留一份 CPU 可读的基矩阵副本。
    // - 当主权重使用 Vulkan 设备内存时，CPU 无法直接读取 stft_*_basis；此副本用于避免回退时崩溃/读错数据。
    struct ggml_tensor * stft_forward_basis_cpu = nullptr;
    struct ggml_tensor * stft_inverse_basis_cpu = nullptr;

	// Prosody Predictor portion of the model
	struct duration_predictor * prosody_pred;

	// Text encoding portion of the model
	struct kokoro_text_encoder * text_encoder;

	// Decoding and Generation portion of the model
	struct kokoro_decoder * decoder;

	// the default hidden states need to be initialized
	std::vector<lstm*> lstms;
	// 说明：Vulkan 下部分权重需要在模型外额外分配（例如 F32 拷贝/展开后的 depthwise kernel），
	// 这些 buffer 不在基类的主权重 buffer 中，需在 free() 时额外释放。
	std::vector<ggml_backend_buffer_t> extra_buffers;

	size_t duration_node_counter = 0;
	size_t generation_node_counter = 0;
	// setting this is likely unnecessary as it is precomputed by the post load function.
	uint32_t post_load_tensor_bytes = 13000;

	size_t max_gen_nodes();
	size_t max_duration_nodes();

	lstm * prep_lstm();
	// helper functions for assigning tensors to substructs
	void assign_lstm(lstm * rnn, std::string name, ggml_tensor * tensor);
	void assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor);
	void assign_gen_resblock(kokoro_generator_residual_block * block, std::string name, ggml_tensor * tensor);
	void assign_ada_res_block(ada_residual_conv_block * block, std::string name, ggml_tensor * tensor);
	void assign_decoder_weight(std::string name, ggml_tensor * tensor);
	void assign_duration_weight(std::string name, ggml_tensor * tensor);
	void assign_text_encoder_weight(std::string name, ggml_tensor * tensor);
	void assign_albert_weight(std::string name, ggml_tensor * tensor);


	void post_load_assign();
	void free();
    void assign_weight(const char * name, ggml_tensor & tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
    	std::function<void (ggml_tensor *)> fn = ([&](ggml_tensor* cur) {
    		std::string name = ggml_get_name(cur);
    		size_t increment = 1;
    		if (name.find("lstm") != std::string::npos) {
    			increment = max_context_length;
    		}
    		if (name.find("duration_predictor") != std::string::npos) {
    			duration_node_counter += increment;
    		} else {
    			generation_node_counter += increment;
    		}
    	});
    	compute_tensor_meta_cb = &fn;
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "kokoro", 1.6, post_load_tensor_bytes);
    }
};

struct kokoro_ubatch {
    size_t n_tokens; // the number of tokens in our encoded sequence
    uint32_t * input_tokens;    // [n_tokens]
    struct kokoro_duration_response * resp = nullptr;
};

struct kokoro_duration_context : runner_context {
    kokoro_duration_context(kokoro_model * model, int n_threads): runner_context(n_threads), model(model) {};
    ~kokoro_duration_context() {
        ggml_backend_buffer_free(buf_len_output);
    }

    std::string voice{};
    struct kokoro_model * model;
    ggml_backend_buffer_t buf_len_output = nullptr;


    size_t  logits_size = 0; // capacity (of floats) for logits
    float * lens 		= nullptr;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * token_types = nullptr;

    void build_schedule() {
        runner_context::build_schedule(model->max_duration_nodes()*5);
    }
};

static struct ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types);
static struct ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias);
static struct ggml_tensor * build_ada_residual_conv(ggml_context * ctx, struct ggml_tensor * x, ada_residual_conv_block * block, struct ggml_tensor * style, struct ggml_tensor * sqrt_tensor);
static struct ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * style,
                                                             kokoro_generator_residual_block * block,
                                                             std::vector<tts_graph_const_input> * const_inputs);
static struct ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, struct ggml_tensor * x,
                                              struct ggml_tensor * style, std::vector<tts_graph_const_input> * const_inputs);
static kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key);
static kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index);
static kokoro_generator_upsample_block * kokoro_generator_upsample_block(gguf_context * meta, int index);

struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

struct kokoro_duration_response {
	size_t n_outputs;
	float * lengths;
	float * hidden_states;
};

// This struct is intended to manage graph and compute for the duration prediction portion of the kokoro model.
// Duration computation and speech generation are separated into distinct graphs because the precomputed graph structure of ggml doesn't
// support the tensor dependent views that would otherwise be necessary.
struct kokoro_duration_runner : tts_runner {
    kokoro_duration_runner(kokoro_model * model, kokoro_duration_context * context, single_pass_tokenizer * tokenizer): model(model), kctx(context), tokenizer(tokenizer) {};
    ~kokoro_duration_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        // 说明：model/kctx 由外层 kokoro_runner 统一管理，避免重复释放导致崩溃。
    }
    struct single_pass_tokenizer * tokenizer;
    kokoro_model * model;
    kokoro_duration_context * kctx;

    void init_build() {
        tts_runner::init_build(&kctx->buf_compute_meta);
    }

    void prepare_post_load();
    struct kokoro_ubatch build_worst_case_batch();
    void set_inputs(kokoro_ubatch & batch);
    struct ggml_cgraph * build_kokoro_duration_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & ubatch);
};

struct kokoro_context : runner_context {
    kokoro_context(kokoro_model * model, int n_threads): runner_context(n_threads), model(model) {};
    // 说明：资源释放统一交给基类 runner_context，避免重复 free。

    std::string voice = "af_alloy";

    struct kokoro_model * model;

    uint32_t total_duration;
    uint32_t sequence_length;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * duration_pred;
    // 每个 frame 对应的 token id（I32，长度=total_duration）。
    // 用于把 token 级别的 hidden state 直接 gather 到 frame 级别，
    // 替代原先的 duration_mask(tokens×frames)+mul_mat 的稀疏展开方式，显著降低算子数与内存占用。
    struct ggml_tensor * duration_ids;
    struct ggml_tensor * window_sq_sum; // needs to be calculatd from the generator window.
    struct ggml_tensor * uv_noise_data;
    struct ggml_tensor * stft_pad_indices = nullptr; // STFT 反射 padding 索引（I32）
    std::vector<tts_graph_const_input> graph_const_inputs; // Vulkan 图常量输入（标量）

    void build_schedule() {
        runner_context::build_schedule(model->max_gen_nodes()*30);
    }
};

// Kokoro 的输入准备（噪声、窗函数、mask 等）虽然占比不高，但在短音频/短句场景下会成为固定开销。
// 该结构用于把这部分开销分段统计出来，方便定位瓶颈与后续优化。
struct kokoro_gen_input_timings {
    double noise_ms         = 0.0;
    double window_sq_sum_ms = 0.0;
    double tensor_set_ms    = 0.0;
    double duration_mask_ms = 0.0;
    double total_ms         = 0.0;
};

// TODO: now that we are passing the context down to these methods we should clean up their parameters
static struct ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, struct ggml_tensor * style, struct ggml_tensor * f0_curve, kokoro_generator* generator, int sequence_length, struct ggml_tensor * window_sq_sum, ggml_cgraph * gf);
static struct ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std);

struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

// This manages the graph compilation of computation for the Kokoro model.
struct kokoro_runner : tts_generation_runner {
    kokoro_runner(unique_ptr<kokoro_model> model, kokoro_context * context, single_pass_tokenizer * tokenizer, kokoro_duration_runner * drunner, phonemizer * phmzr, const generation_configuration & config): tts_generation_runner{kokoro_loader}, model{move(model)}, kctx(context), tokenizer(tokenizer), drunner(drunner), phmzr(phmzr), voice{config.voice} {
    	tts_runner::sampling_rate = 24000.0f;
    	tts_runner::supports_voices = true;
    };
    ~kokoro_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        delete drunner;
        model->free();
        delete kctx;
        delete phmzr;
    }
    struct single_pass_tokenizer * tokenizer;
    unique_ptr<kokoro_model> model;
    kokoro_context * kctx;
    kokoro_duration_runner * drunner;
    phonemizer * phmzr;

    void init_build() {
        tts_runner::init_build(&kctx->buf_compute_meta);
    }

    std::vector<std::string_view> list_voices() override;
    std::vector<std::vector<uint32_t>> tokenize_chunks(std::vector<std::string> clauses);
    void assign_weight(const char * name, ggml_tensor & tensor) override;
    void prepare_post_load() override;
    kokoro_ubatch build_worst_case_batch();
    kokoro_gen_input_timings set_inputs(kokoro_ubatch & batch, uint32_t total_size);
    struct ggml_cgraph * build_kokoro_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & batch, tts_response & outputs);
    bool try_phonemize(const char * prompt, std::string & out_phonemes, const generation_configuration & config) override;
    bool try_phonemize_segments(const char * prompt,
                               std::string & out_phonemes,
                               std::vector<tts_generation_runner::phoneme_segment> & out_segments,
                               const generation_configuration & config) override;
    void generate(const char * prompt, tts_response & response, const generation_configuration & config) override;
 private:
    string voice{};
    void propagate_voice_setting();
};
