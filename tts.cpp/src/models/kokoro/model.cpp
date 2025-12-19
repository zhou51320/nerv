#include "model.h"
#include "numbers_compat.h"
#include "multilingual.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// 计时工具：为了分析 CPU 推理慢在哪里（中文前端/英文 phonemizer、duration、generator 等），
// 这里提供一个统一的 us 计时入口，并用环境变量开关避免默认污染输出/引入额外开销。
static int64_t tts_time_us() {
    tts_time_init_once();
    return ggml_time_us();
}

// 说明：在 Vulkan 后端下，默认输入张量会被调度器分配到 CPU，
// 这会导致 Vulkan 图里混入非 Vulkan buffer，从而在后端构图时崩溃。
// 这里主动将输入叶子节点绑定到 Vulkan 后端，避免 CPU/Vulkan buffer 混用。
static bool tts_backend_is_vulkan(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    return name && std::strncmp(name, "Vulkan", 6) == 0;
}

// 说明：允许运行时通过环境变量强制 Vulkan 生成阶段（便于回归测试）。
static bool kokoro_env_force_vulkan_gen() {
    const char * v = std::getenv("TTS_VK_FORCE_GEN");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static void kokoro_force_inputs_backend(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched) {
        return;
    }
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor * leaf = gf->leafs[i];
        if (!leaf) {
            continue;
        }
        if (leaf->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, leaf, ctx->backend);
        }
    }
}

// 说明：Vulkan 图中若包含自定义算子（CPU-only），会引入跨后端 buffer 混用，
// 在某些驱动下 ggml-vulkan 构图会崩溃；此处用于提前侦测并回退。
static bool kokoro_graph_has_custom_ops(const ggml_cgraph * gf) {
    if (!gf) {
        return false;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        if (!node) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_CUSTOM:
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
                return true;
            default:
                break;
        }
    }
    return false;
}

static void kokoro_force_graph_backend(runner_context * ctx, ggml_cgraph * gf, ggml_backend_t backend) {
    if (!ctx || !gf || !backend || !ctx->sched) {
        return;
    }
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor * leaf = gf->leafs[i];
        if (!leaf) {
            continue;
        }
        // 说明：若输入叶子已落在非主机 buffer（如 Vulkan 设备内存），
        // 强行改 CPU 会阻断调度器的拷贝逻辑，导致 CPU 直接访问 GPU 内存崩溃；
        // 因此对非主机 buffer 的叶子保持原后端，由调度器负责拷贝。
        if (leaf->buffer && !ggml_backend_buffer_is_host(leaf->buffer)) {
            continue;
        }
        ggml_backend_sched_set_tensor_backend(ctx->sched, leaf, backend);
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (node) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, backend);
        }
    }
}

static bool kokoro_tensor_depends_on_custom(const ggml_tensor * t, std::unordered_map<const ggml_tensor *, bool> & cache) {
    if (!t) {
        return false;
    }
    auto it = cache.find(t);
    if (it != cache.end()) {
        return it->second;
    }
    bool is_custom = t->op == GGML_OP_CUSTOM ||
                     t->op == GGML_OP_MAP_CUSTOM1 ||
                     t->op == GGML_OP_MAP_CUSTOM2 ||
                     t->op == GGML_OP_MAP_CUSTOM3;
    if (!is_custom) {
        if (t->view_src && kokoro_tensor_depends_on_custom(t->view_src, cache)) {
            is_custom = true;
        } else {
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (kokoro_tensor_depends_on_custom(t->src[i], cache)) {
                    is_custom = true;
                    break;
                }
            }
        }
    }
    cache[t] = is_custom;
    return is_custom;
}

static void kokoro_force_custom_views_cpu(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched || !ctx->backend_cpu) {
        return;
    }
    std::unordered_map<const ggml_tensor *, bool> cache;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node || !node->view_src) {
            continue;
        }
        if (kokoro_tensor_depends_on_custom(node->view_src, cache)) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
        }
    }
}

static bool tts_timings_enabled() {
    const char * v = std::getenv("TTS_TIMINGS");
    // 计时默认开启：便于直接观测端到端推理性能。
    // 如需关闭，可在运行前设置环境变量：TTS_TIMINGS=0
    if (v == nullptr) {
        return true;
    }
    if (v[0] == '\0') {
        return true;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static double us_to_ms(const int64_t us) {
    return us / 1000.0;
}

} // namespace

static struct ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types) {
	struct ggml_tensor * tinpts = ggml_cont(ctx, ggml_get_rows(ctx, model->token_embd, input_tokens));
	struct ggml_tensor * pinpts = ggml_get_rows(ctx, model->position_embd, positions);

	struct ggml_tensor * inpts = ggml_cont(ctx, ggml_add(ctx, tinpts, pinpts));
	if (!model->static_token_types) {
		// Token type embeddings are actually static for kokoro at the moment, so we should never need to compute this on the fly.
		return ggml_add(ctx, inpts, ggml_get_rows(ctx, model->token_type_embd, token_types));
	}
	struct ggml_tensor * ainpts = ggml_add(ctx, inpts, model->static_token_type_values);

	struct ggml_tensor * out = ggml_cont(ctx, build_albert_norm(ctx, ainpts, model->input_norm_weight, model->input_norm_bias));
	return ggml_add(ctx, ggml_mul_mat(ctx, model->embd_hidden, out), model->embd_hidden_bias);
}

static struct ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
	// this is the standard eps for Albert
	float eps = 0.000000000001;
    cur = ggml_norm(ctx, cur, eps);
    cur = ggml_cont(ctx, ggml_add(ctx, ggml_mul(ctx, cur, weight), bias));
    return cur;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, std::vector<ggml_tensor*> weights, std::vector<ggml_tensor*> biases, uint32_t sequence_length, bool reversed = false);

static struct ggml_tensor * build_lstm(ggml_context * ctx, ggml_tensor * input, lstm* rnn, uint32_t sequence_length, ggml_cgraph * gf) {
	struct ggml_tensor * resp = input;
	struct ggml_tensor * reverse_resp = input;

	// iterate over cells first so that at each pass to the next cell we have a fully formed vector (this improves performance as well as allocation for stacked lstms)
	for (int c = 0; c < rnn->cells.size(); c++) {
		ggml_build_forward_expand(gf, resp);
		resp = build_lstm_run(ctx, gf, resp, rnn->hidden[c], rnn->states[c], rnn->cells[c]->weights, rnn->cells[c]->biases, sequence_length);
		if (rnn->bidirectional) {
			reverse_resp = build_lstm_run(ctx, gf, reverse_resp, rnn->hidden[c], rnn->states[c], rnn->cells[c]->reverse_weights, rnn->cells[c]->reverse_biases, sequence_length, true);
		}
	}
	if (rnn->bidirectional) {
		resp = ggml_concat(ctx, resp, reverse_resp, 0);
	}
	return resp;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, std::vector<ggml_tensor*> weights, std::vector<ggml_tensor*> biases, uint32_t sequence_length, bool reversed) {
	struct ggml_tensor * I = ggml_add(ctx, ggml_mul_mat(ctx, weights[0], input), biases[0]);
	struct ggml_tensor * F = ggml_add(ctx, ggml_mul_mat(ctx, weights[2], input), biases[2]);
	struct ggml_tensor * G = ggml_add(ctx, ggml_mul_mat(ctx, weights[4], input), biases[4]);
	struct ggml_tensor * O = ggml_add(ctx, ggml_mul_mat(ctx, weights[6], input), biases[6]);

	struct ggml_tensor * outputs;

	for (int index = 0; index < sequence_length; index++) {
		int i = reversed ? sequence_length - 1 - index : index;
		struct ggml_tensor * I_cur = ggml_view_3d(ctx, I, I->ne[0], 1, I->ne[2], I->nb[0], I->nb[1], I->nb[1]*i);
		I_cur = ggml_sigmoid(ctx, ggml_add(ctx, I_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[1], h_0), biases[1])));

		struct ggml_tensor * F_cur = ggml_view_3d(ctx, F, F->ne[0], 1, F->ne[2], F->nb[0], F->nb[1], F->nb[1]*i);
		F_cur = ggml_sigmoid(ctx, ggml_add(ctx, F_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[3], h_0), biases[3])));

		struct ggml_tensor * G_cur = ggml_view_3d(ctx, G, G->ne[0], 1, G->ne[2], G->nb[0], G->nb[1], G->nb[1]*i);
		G_cur = ggml_tanh(ctx, ggml_add(ctx, G_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[5], h_0), biases[5])));

		struct ggml_tensor * O_cur = ggml_view_3d(ctx, O, O->ne[0], 1, O->ne[2], O->nb[0], O->nb[1], O->nb[1]*i);
		O_cur = ggml_sigmoid(ctx, ggml_add(ctx, O_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[7], h_0), biases[7])));

		c_0 = ggml_add(ctx, ggml_mul(ctx, F_cur, c_0), ggml_mul(ctx, I_cur, G_cur));
		h_0 = ggml_mul(ctx, ggml_tanh(ctx, c_0), O_cur);

		if (index == 0) {
			outputs = h_0;
		} else {
			outputs = reversed ? ggml_concat(ctx, h_0, outputs, 1) : ggml_concat(ctx, outputs, h_0, 1);
		}
		ggml_build_forward_expand(gf, outputs);
	}
	return outputs;
}

static struct ggml_tensor * build_ada_residual_conv(ggml_context * ctx, struct ggml_tensor * x, ada_residual_conv_block * block, struct ggml_tensor * style, struct ggml_tensor * sqrt_tensor) {
	struct ggml_tensor * cur = x;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;

	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_gamma, style), block->norm1_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_beta, style), block->norm1_beta_bias);
	cur   = ggml_norm(ctx, x, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	// 说明：ggml 0.9.4 的二元算子在“src1 非 contiguous + broadcast”时会触发断言（CPU 路径暂未实现该组合）。
	// gamma/beta 这里通过 transpose 来对齐维度，因此需要再做一次 ggml_cont 以确保 src1 contiguous。
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, gamma))));
	cur = ggml_add(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, beta)));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);

	if (block->pool) {
		// 说明：这里的 pool 是 depthwise 转置卷积（groups=channels）并带 output_padding，
		// ggml 0.9.4 的内置 conv_transpose_1d 不再支持这些扩展能力。
		// 兼容策略：项目侧提供 tts_conv_transpose_1d，在 groups>1 时走 CPU 自定义算子兜底。
		cur = tts_conv_transpose_1d(ctx, block->pool, cur,
		                            /*stride=*/2, /*padding=*/1, /*dilation=*/1,
		                            /*output_padding=*/1, /*groups=*/(int) cur->ne[1]);
		cur = ggml_add(ctx, cur, block->pool_bias);
	}

 	cur = ggml_conv_1d(ctx, block->conv1, cur, 1, 1, 1);

	cur   = ggml_add(ctx, cur, block->conv1_bias);
	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_gamma, style), block->norm2_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_beta, style), block->norm2_beta_bias);
	cur   = ggml_norm(ctx, cur, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	// 同上：确保 src1 contiguous，避免 ggml-cpu 对“非连续 broadcast”断言失败。
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, gamma))));
	cur = ggml_add(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, beta)));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
	cur = ggml_add(ctx, ggml_conv_1d(ctx, block->conv2, cur, 1, 1, 1), block->conv2_bias);

	struct ggml_tensor * res = cur;
	cur = x;
	if (block->upsample) {
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
		if (block->pool) {
			// 说明：旧版 ggml 的 ggml_upscale_ext 已弃用/变更签名，这里用 ggml_interpolate 取代。
			// 该处只是为了对齐维度（最近邻上采样即可）。
			cur = ggml_interpolate(ctx, cur,
			                       /*ne0=*/cur->ne[0], /*ne1=*/cur->ne[1] * 2, /*ne2=*/cur->ne[2], /*ne3=*/cur->ne[3],
			                       (uint32_t) GGML_SCALE_MODE_NEAREST);
		}
		cur = ggml_mul_mat(ctx, block->upsample, cur);
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
	}
	cur = ggml_div(ctx, ggml_add(ctx, res, cur), sqrt_tensor);
	return cur;
}

static struct ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * style, kokoro_generator_residual_block * block) {
	struct ggml_tensor * cur;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;
	struct ggml_tensor * inpl = x;
	for (int i = 0; i < block->convs1_weights.size(); i++) {
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_gamma_weights[i], style), block->adain1d_1_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_beta_weights[i], style), block->adain1d_1_beta_biases[i]);
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, inpl, 0.00001)));

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
		cur   = snake_1d(ctx, block->input_alphas[i], ggml_cont(ctx, ggml_transpose(ctx, cur)));

		cur   = ggml_add(ctx, ggml_conv_1d(ctx, block->convs1_weights[i], cur, 1, block->conv1_paddings[i], block->conv1_dilations[i]), block->convs1_biases[i]);
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_gamma_weights[i], style), block->adain1d_2_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_beta_weights[i], style), block->adain1d_2_beta_biases[i]);
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, cur, 0.00001)));

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta)));

		cur   = snake_1d(ctx, block->output_alphas[i], cur);
		cur   = ggml_add(ctx, ggml_conv_1d(ctx, block->convs2_weights[i], cur, 1, block->conv1_paddings[0], 1), block->convs2_biases[i]);
		inpl   = ggml_add(ctx, inpl, cur);
	}
	return inpl;
}

static struct ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, struct ggml_tensor * x, struct ggml_tensor * style) {
	// This conv_1d seems replaceable with squeezed and transposed ggml_mul_mut, but s0 and p0 are dynamic
	ggml_tensor * cur = ggml_add(ctx, ggml_conv_1d(ctx, block->input_conv, x, block->input_conv_stride, block->input_conv_padding, 1), block->input_conv_bias);
	return build_kokoro_generator_res_block(ctx, cur, style, block->res_block);
}

static struct ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std) {
	struct ggml_tensor * cur = ggml_mul(ctx, ggml_repeat(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], harmonic_num)), model->harmonic_sampling_norm);
	// 说明：旧版 ggml 有 ggml_mod(x, 1.0)；ggml 0.9.4 移除了该算子。
	// 这里使用 x - floor(x) 来实现“对 1 取模”（即取小数部分），满足 Kokoro 的相位累加需求。
	cur = ggml_sub(ctx, cur, ggml_floor(ctx, cur));
	cur = ggml_mul(ctx, ggml_cumsum(ctx, cur), model->sampling_factor_scalar);

	// 说明：旧版 ggml 的 ggml_upscale_linear / ggml_upscale_ext 已弃用/变更签名，
	// 统一替换为 ggml_interpolate（线性插值时用 BILINEAR 模式即可）。
	cur = ggml_interpolate(ctx, cur,
	                       /*ne0=*/cur->ne[0] * 300, /*ne1=*/cur->ne[1], /*ne2=*/cur->ne[2], /*ne3=*/cur->ne[3],
	                       (uint32_t) GGML_SCALE_MODE_BILINEAR);
	struct ggml_tensor * upscaled = ggml_interpolate(ctx, x,
	                                                 /*ne0=*/x->ne[0] * 300, /*ne1=*/x->ne[1], /*ne2=*/x->ne[2], /*ne3=*/x->ne[3],
	                                                 (uint32_t) GGML_SCALE_MODE_BILINEAR);

	kctx->uv_noise_data = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length*harmonic_num+4);
	ggml_set_input(kctx->uv_noise_data);

    struct ggml_tensor * fake = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sequence_length, harmonic_num, 2);

    // ggml doesn't support boolean tensors nor does it support greater than and roll ops. As a result, we represent these boolean tensors as 1.0 or 0.0 or simply perform
    // multiplications in place via a custom map.
    struct ggml_tensor * uv_noise = ggml_map_custom3(ctx, fake, upscaled, kctx->uv_noise_data, &uv_noise_compute, sequence_length, nullptr);


    struct ggml_tensor * noise = ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], uv_noise->nb[2]));
    struct ggml_tensor * uv = ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], 0));

	return ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_mul(ctx, ggml_sin(ctx, cur), uv), noise)));
}

static struct ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, struct ggml_tensor * style, struct ggml_tensor * f0_curve, kokoro_generator* generator, int sequence_length, struct ggml_tensor * window_sq_sum, ggml_cgraph * gf) {
	struct ggml_tensor * sing = build_sin_gen(ctx, model, kctx, f0_curve, model->harmonic_num + 1, f0_curve->ne[0] * 300, model->voice_threshold, model->sin_amp, model->noise_std);
	struct ggml_tensor * har = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, generator->m_source_weight, sing), generator->m_source_bias));

	har = stft(ctx, ggml_cont(ctx, ggml_transpose(ctx, har)), generator->window, model->true_n_fft, model->stft_hop, true, true);

	// stft returns a vector of shape [nfft, frames, batch, 2] where the final shape (2) separates the magnitude and the phase
	// kokoro concatenates the n_fft from the magnitude and the phase together so we have to split them up and concatenate
	// along the n_fft axis
	struct ggml_tensor * mhar  = ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], 0));
	struct ggml_tensor * phhar = ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], har->nb[3]));
	struct ggml_tensor * combined_har = ggml_cont(ctx, ggml_transpose(ctx, ggml_concat(ctx, mhar, phhar, 0)));

	struct ggml_tensor * cur = x;
	for (int i = 0; i < generator->ups.size(); i++) {
		cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
		// 说明：ggml 0.9.4 的 conv_transpose_1d 目前限制 padding==0，因此这里走项目侧兼容封装：
		// 先计算 padding=0 的更长输出，再裁剪得到等价结果（groups==1 场景）。
		cur = ggml_add(ctx,
		               tts_conv_transpose_1d(ctx,
		                                     generator->ups[i]->upsample_weight,
		                                     ggml_cont(ctx, ggml_transpose(ctx, cur)),
		                                     (int) generator->ups[i]->stride,
		                                     (int) generator->ups[i]->padding,
		                                     /*dilation=*/1),
		               generator->ups[i]->upsample_bias);
		if (i == generator->ups.size() - 1) {
			// This is a hacky way of implementing the simple reflection padding used here.
			// In general, ggml should eventually be built to support expressive reflective padding but for such simple front padding this makes more sense.
			struct ggml_tensor * temp = ggml_cont(ctx, ggml_view_3d(ctx, cur, 1, cur->ne[1], cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[0]));
			cur = ggml_concat(ctx, temp, cur, 0);
		}
		struct ggml_tensor * x_source = build_noise_block(ctx, generator->noise_blocks[i], ggml_cont(ctx, combined_har), style);
		cur = ggml_add(ctx, cur, x_source);
		struct ggml_tensor * x = cur;
		for (int ii = 0; ii < model->n_kernels; ii++) {
			if (ii == 0) {
				cur = build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii]);
			} else {
				cur = ggml_add(ctx, cur, build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii]));
			}
		}
		cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_div(ctx, cur, model->n_kernels_tensor)));
		ggml_build_forward_expand(gf, cur);
	}

	cur = ggml_leaky_relu(ctx, cur, 0.01f, false);
	cur = ggml_add(ctx, ggml_conv_1d(ctx, generator->out_conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, model->out_conv_padding, 1), generator->out_conv_bias);

	struct ggml_tensor * spec = ggml_view_3d(ctx, cur, cur->ne[0], model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], 0);
	struct ggml_tensor * phase = ggml_view_3d(ctx, cur, cur->ne[0], cur->ne[1] - model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[1] * model->post_n_fft);
	phase = ggml_sin(ctx, phase);
	spec = ggml_exp(ctx, spec);

	cur = ggml_concat(ctx, spec, phase, 3); // istft expects the magnitude and phase concatenated after the batch;
	cur = istft(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), window_sq_sum, generator->window, model->true_n_fft, model->stft_hop, true, true);
	ggml_set_name(cur, "after_res_gen");
	return cur;
}

static struct kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key) {
	struct kokoro_generator_residual_block * grb = new struct kokoro_generator_residual_block;
	// these residual blocks always have 3 convolutional layers
	for (int i = 0; i < 3; i++) {
		grb->adain1d_1_gamma_weights.push_back(nullptr);
		grb->adain1d_2_gamma_weights.push_back(nullptr);
		grb->adain1d_1_gamma_biases.push_back(nullptr);
		grb->adain1d_2_gamma_biases.push_back(nullptr);
		grb->adain1d_1_beta_weights.push_back(nullptr);
		grb->adain1d_2_beta_weights.push_back(nullptr);
		grb->adain1d_1_beta_biases.push_back(nullptr);
		grb->adain1d_2_beta_biases.push_back(nullptr);
		grb->input_alphas.push_back(nullptr);
		grb->output_alphas.push_back(nullptr);
		grb->convs1_weights.push_back(nullptr);
		grb->convs1_biases.push_back(nullptr);
		grb->convs2_weights.push_back(nullptr);
		grb->convs2_biases.push_back(nullptr);
		int padding_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".padding").c_str());
		int dilation_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".dilation").c_str());
		if (padding_key == -1 || dilation_key == -1) {
			TTS_ABORT("Could not find dilation and padding for generator residual block at key, '%s.%d'.", base_config_key.c_str(), i);
		}
		grb->conv1_dilations.push_back(gguf_get_val_u32(meta, dilation_key));
		grb->conv1_paddings.push_back(gguf_get_val_u32(meta, padding_key));
	}
	return grb;
}

static struct kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index) {
	struct kokoro_noise_residual_block * nb = new struct kokoro_noise_residual_block;
	std::string base = "kokoro.decoder.generator.noise_blocks." + std::to_string(index);
	nb->res_block = build_res_block_from_file(meta, base + ".res_block");
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro noise block.");
	}
	nb->input_conv_stride = gguf_get_val_u32(meta, stride_key);
	nb->input_conv_padding = gguf_get_val_u32(meta, padding_key);
	return nb;
}

static struct kokoro_generator_upsample_block * kokoro_generator_upsample_block(gguf_context * meta, int index) {
	struct kokoro_generator_upsample_block * usb = new struct kokoro_generator_upsample_block;
	std::string base = "kokoro.decoder.generator.up_convs." + std::to_string(index);
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro upsample block.");
	}
	usb->stride = gguf_get_val_u32(meta, stride_key);
	usb->padding = gguf_get_val_u32(meta, padding_key);
	return usb;
}

size_t kokoro_model::max_gen_nodes() {
	return std::max<size_t>(8192, generation_node_counter*2);
}

size_t kokoro_model::max_duration_nodes() {
	return std::max<size_t>(8192, duration_node_counter*2);
}

void kokoro_model::post_load_assign() {
	size_t original_offset = offset;
	n_kernels_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    // 说明：这些“后处理常量张量”会被放进与权重相同的 buffer 中；在 Vulkan 后端下必须满足 storage buffer 的对齐要求。
    alloc_tensor(n_kernels_tensor, "n_kernels_tensor");
	size_t size = ggml_nbytes(n_kernels_tensor);
	float nker = (float) n_kernels;
	ggml_backend_tensor_set(n_kernels_tensor, &nker, 0, size);

	sqrt_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    alloc_tensor(sqrt_tensor, "sqrt_tensor");
    size = ggml_nbytes(sqrt_tensor);
    float sqrt2 = sqrtf(2.0f);
	ggml_backend_tensor_set(sqrt_tensor, &sqrt2, 0, size);

	std::vector<float> data{};
	for (int l = 0; l < lstms.size(); l++) {
		lstm * rnn = lstms[l];
		const int32_t hidden_size = rnn->cells[0]->biases[0]->ne[0];
		data.resize(hidden_size);
 
 		for (int i = 0; i < rnn->cells.size(); i++) {
 			struct ggml_tensor * h = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
 			struct ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            alloc_tensor(h);
     		size_t size = ggml_nbytes(h);
 			ggml_backend_tensor_set(h, data.data(), 0, size);
 			ggml_format_name(h, "lstm%d_hidden", l);
            alloc_tensor(s);
 			ggml_backend_tensor_set(s, data.data(), 0, size);
 			ggml_format_name(s, "lstm%d_state", l);
     		rnn->hidden.push_back(h);
     		rnn->states.push_back(s);
 		}
 		data.clear();
	}

 	if (window == "hann") {
		// 说明：保留一份 CPU 侧窗函数缓存，用于输入准备时计算 window_sq_sum。
		decoder->generator->window_host.clear();
		decoder->generator->window_host.reserve(true_n_fft);
 		hann_window(true_n_fft, decoder->generator->window_host);
 		decoder->generator->window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, true_n_fft);
        alloc_tensor(decoder->generator->window, "stft_window");
 		size_t size = ggml_nbytes(decoder->generator->window);
 		ggml_backend_tensor_set(decoder->generator->window, decoder->generator->window_host.data(), 0, size);
 	} else {
 		TTS_ABORT("Window of type %s is not supported.", window.c_str());
 	}

 	harmonic_sampling_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, harmonic_num + 1);
    alloc_tensor(harmonic_sampling_norm, "harmonic_sampling_norm");
 	std::vector<float> hdata;
 	hdata.reserve(harmonic_num + 1);
 	for (int i = 0; i < harmonic_num + 1; i++) {
 		hdata.push_back(((float)i + 1.0f) / sample_rate);
 	}
 	size_t hsize = ggml_nbytes(harmonic_sampling_norm);
 	ggml_backend_tensor_set(harmonic_sampling_norm, hdata.data(), 0, hsize);
 	hdata.clear();

 	sampling_factor_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    alloc_tensor(sampling_factor_scalar, "sampling_factor_scalar");
     size_t scsize = ggml_nbytes(sampling_factor_scalar);
     // while it might appear that the upsampling_rate could be used here, the interpolation rate (i.e. the upsampling scale) is actually independent in the kokoro model implementation.
     float sample_scalar = upsample_scale*2.0f*std::numbers::pi;
 	ggml_backend_tensor_set(sampling_factor_scalar, &sample_scalar, 0, scsize);
 	post_load_tensor_bytes = 300 + offset - original_offset;
}

void kokoro_model::assign_lstm(lstm * rnn, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	int ii = std::stoi(parts[2]);
	if (parts[1] == "weights") {
		rnn->cells[i]->weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->weights[ii], tensor);
	} else if (parts[1] == "biases") {
		rnn->cells[i]->biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->biases[ii], tensor);
	} else if (parts[1] == "reverse_weights") {
		rnn->cells[i]->reverse_weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_weights[ii], tensor);
	} else if (parts[1] == "reverse_biases") {
		rnn->cells[i]->reverse_biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_biases[ii], tensor);
	}
}

void kokoro_model::assign_weight(const char * name, ggml_tensor & tensor) {
    if (const string_view name_sv{ name }; name_sv.starts_with("albert.")) {
        assign_albert_weight(string{ name_sv.substr(sizeof("albert.") - 1) }, &tensor);
    } else if (name_sv.starts_with("duration_predictor.")) {
        assign_duration_weight(string{ name_sv.substr(sizeof("duration_predictor.") - 1) }, &tensor);
    } else if (name_sv.starts_with("text_encoder.")) {
        assign_text_encoder_weight(string{ name_sv.substr(sizeof("text_encoder.") - 1) }, &tensor);
    } else if (name_sv.starts_with("decoder.")) {
        assign_decoder_weight(string{ name_sv.substr(sizeof("decoder.") - 1) }, &tensor);
    } else if (name_sv.starts_with("voice_tensors.")) {
        const string voice{ name_sv.substr(sizeof("voice_tensors.") - 1) };
        voices[voice] = ggml_dup_tensor(ctx, &tensor);
        set_tensor(voices[voice], &tensor);
    }
}

void kokoro_model::assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor) {
	if (name == "m_source_weight") {
		generator->m_source_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_weight, tensor);
	} else if (name == "m_source_bias") {
		generator->m_source_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_bias, tensor);
	} else if (name == "conv_post_weight") {
		generator->out_conv_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->out_conv_weight, tensor);
	} else if (name == "conv_post_bias") {
		generator->out_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(generator->out_conv_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[0] == "noise_blocks") {
			if (parts[2] == "conv_weight") {
				generator->noise_blocks[i]->input_conv = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->noise_blocks[i]->input_conv, tensor);
			} else if (parts[2] == "conv_bias") {
				generator->noise_blocks[i]->input_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->noise_blocks[i]->input_conv_bias, tensor);
			} else if (parts[2] == "resblock") {
				assign_gen_resblock(generator->noise_blocks[i]->res_block, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		} else if (parts[0] == "resblocks") {
			assign_gen_resblock(generator->res_blocks[i], name.substr(parts[0].size()+parts[1].size()+2), tensor);
		} else if (parts[0] == "ups") {
			if (parts[2] == "weight") {
				generator->ups[i]->upsample_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->ups[i]->upsample_weight, tensor);
			} else if (parts[2] == "bias") {
				generator->ups[i]->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->ups[i]->upsample_bias, tensor);
			}
		}
	}
}

void kokoro_model::assign_gen_resblock(kokoro_generator_residual_block * block, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	if (parts[1] == "gamma1_weight") {
		block->adain1d_1_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma2_weight") {
		block->adain1d_2_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma1_bias") {
		block->adain1d_1_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_biases[i], tensor);
	} else if (parts[1] == "gamma2_bias") {
		block->adain1d_2_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_biases[i], tensor);
	} else if (parts[1] == "beta1_weight") {
		block->adain1d_1_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_weights[i], tensor);
	} else if (parts[1] == "beta2_weight") {
		block->adain1d_2_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_weights[i], tensor);
	} else if (parts[1] == "beta1_bias") {
		block->adain1d_1_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_biases[i], tensor);
	} else if (parts[1] == "beta2_bias") {
		block->adain1d_2_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_biases[i], tensor);
	} else if (parts[1] == "convs1_weight") {
		block->convs1_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs1_weights[i], tensor);
	} else if (parts[1] == "convs2_weight") {
		block->convs2_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs2_weights[i], tensor);
	} else if (parts[1] == "convs1_bias") {
		block->convs1_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs1_biases[i], tensor);
	} else if (parts[1] == "convs2_bias") {
		block->convs2_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs2_biases[i], tensor);
	} else if (parts[1] == "alpha1") {
		block->input_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->input_alphas[i], tensor);
	} else if (parts[1] == "alpha2") {
		block->output_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->output_alphas[i], tensor);
	}
}

/**
 * Removes the last axis, for cases where it's redundantly of length 1.
 * assert x.ndim == 3; numpy.squeeze(x, axis=-1)
 */
static ggml_tensor * squeeze_3d_2d_e0(ggml_context * ctx, ggml_tensor * x) {
	TTS_ASSERT(x->ne[0] == 1);
	TTS_ASSERT(ggml_is_contiguous(x));
	return ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);
}

void kokoro_model::assign_ada_res_block(ada_residual_conv_block * block, std::string name, ggml_tensor * tensor) {
	if (name == "norm1_gamma_weight") {
		block->norm1_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma, tensor);
	} else if (name == "norm2_gamma_weight") {
		block->norm2_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma, tensor);
	} else if (name == "norm1_gamma_bias") {
		block->norm1_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma_bias, tensor);
	} else if (name == "norm2_gamma_bias") {
		block->norm2_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma_bias, tensor);
	} else if (name == "norm1_beta_weight") {
		block->norm1_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta, tensor);
	} else if (name == "norm2_beta_weight") {
		block->norm2_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta, tensor);
	} else if (name == "norm1_beta_bias") {
		block->norm1_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta_bias, tensor);
	} else if (name == "norm2_beta_bias") {
		block->norm2_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta_bias, tensor);
	} else if (name == "conv1_weight") {
		block->conv1 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv1, tensor);
	} else if (name == "conv2_weight") {
		block->conv2 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv2, tensor);
	} else if (name == "conv1_bias") {
		block->conv1_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv1_bias, tensor);
	} else if (name == "conv2_bias") {
		block->conv2_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv2_bias, tensor);
	} else if (name == "pool_weight") {
		block->pool = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->pool, tensor);
	} else if (name == "pool_bias") {
		block->pool_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->pool_bias, tensor);
	} else if (name == "conv1x1_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		block->upsample = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->upsample, tensor);
	} else if (name == "conv1x1_bias") {
		block->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->upsample_bias, tensor);
	}
}

void kokoro_model::assign_decoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "f0_conv_weight") {
		decoder->f0_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->f0_conv, tensor);
	} else if (name == "f0_conv_bias") {
		decoder->f0_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->f0_conv_bias, tensor);
	} else if (name == "n_conv_weight") {
		decoder->n_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->n_conv, tensor);
	} else if (name == "n_conv_bias") {
		decoder->n_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->n_conv_bias, tensor);
	} else if (name == "asr_conv_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		decoder->asr_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->asr_conv, tensor);
	} else if (name == "asr_conv_bias") {
		decoder->asr_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->asr_conv_bias, tensor);
	} else if (has_prefix(name, "decoder_blocks")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		assign_ada_res_block(decoder->decoder_blocks[i], parts[2], tensor);
	} else if (has_prefix(name, "encoder_block")) {
		std::vector<std::string> parts = split(name, ".");
		assign_ada_res_block(decoder->encoder_block, parts[1], tensor);
	} else if (has_prefix(name, "generator")) {
		assign_generator_weight(decoder->generator, name.substr(10), tensor);
	}
}

void kokoro_model::assign_duration_weight(std::string name, ggml_tensor * tensor) {
	if (name == "encode") {
		prosody_pred->albert_encode = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode , tensor);
	} else if (name == "encode_bias") {
		prosody_pred->albert_encode_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode_bias, tensor);
	} else if (name == "duration_proj") {
		prosody_pred->duration_proj = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj, tensor);
	} else if (name == "duration_proj_bias") {
		prosody_pred->duration_proj_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj_bias, tensor);
	} else if (name == "n_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->n_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->n_proj_kernel, tensor);
	} else if (name == "n_proj_bias") {
		prosody_pred->n_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->n_proj_bias, tensor);
	} else if (name == "f0_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->f0_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->f0_proj_kernel, tensor);
	} else if (name == "f0_proj_bias") {
		prosody_pred->f0_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->f0_proj_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		if (parts[0] == "shared_lstm") {
			assign_lstm(prosody_pred->shared_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "duration_lstm") {
			assign_lstm(prosody_pred->duration_proj_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "f0_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->f0_blocks[i], parts[2], tensor);
		} else if (parts[0] == "n_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->n_blocks[i], parts[2], tensor);
		} else if (parts[0] == "layers") {
			int i = std::stoi(parts[1]);
			i = i / 2;
			if (parts[2] == "gamma_weight") {
				prosody_pred->layers[i]->ada_norm_gamma_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_weight , tensor);
			} else if (parts[2] == "gamma_bias") {
				prosody_pred->layers[i]->ada_norm_gamma_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_bias , tensor);
			} else if (parts[2] == "beta_weight") {
				prosody_pred->layers[i]->ada_norm_beta_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_weight , tensor);
			} else if (parts[2] == "beta_bias") {
				prosody_pred->layers[i]->ada_norm_beta_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_bias , tensor);
			} else if (parts[2] == "lstm") {
				assign_lstm(prosody_pred->layers[i]->rnn, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		}
	}
}

void kokoro_model::assign_text_encoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embedding_weight") {
		text_encoder->embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(text_encoder->embd, tensor);
	} else if (has_prefix(name, "lstm")) {
		assign_lstm(text_encoder->out_lstm, name.substr(5), tensor);
	} else if (has_prefix(name, "layers")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[2] == "gamma") {
			text_encoder->conv_layers[i]->norm_gamma = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_gamma, tensor);
		} else if (parts[2] == "beta") {
			text_encoder->conv_layers[i]->norm_beta = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_beta, tensor);
		} else if (parts[2] == "weight") {
			text_encoder->conv_layers[i]->conv_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->conv_weight, tensor);
		} else if (parts[2] == "bias") {
			text_encoder->conv_layers[i]->conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
			set_tensor(text_encoder->conv_layers[i]->conv_bias, tensor);
		}
	}
}

void kokoro_model::assign_albert_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embd") {
		embd_hidden = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden, tensor);
	} else if (name == "embd_bias") {
		embd_hidden_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden_bias, tensor);
	} else if (name == "token_embd") {
		token_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(token_embd, tensor);
	} else if (name == "position_embd") {
		position_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(position_embd, tensor);
	} else if (name == "norm") {
		input_norm_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_weight, tensor);
	} else if (name == "norm_bias") {
		input_norm_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_bias, tensor);
	} else if (name == "token_type_embd") {
		static_token_type_values = ggml_dup_tensor(ctx, tensor);
		set_tensor(static_token_type_values, tensor);
	} else if (has_prefix(name, "layer")) {
		std::vector<std::string> parts = split(name, '.');
		int i = std::stoi(parts[1]);
		if (parts[2] == "ffn") {
			layers[i]->ffn = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn, tensor);
		} else if (parts[2] == "ffn_bias") {
			layers[i]->ffn_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_bias, tensor);
		} else if (parts[2] == "ffn_out") {
			layers[i]->ffn_out = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out, tensor);
		} else if (parts[2] == "ffn_out_bias") {
			layers[i]->ffn_out_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out_bias, tensor);
		} else if (parts[2] == "attn_norm") {
			layers[i]->layer_output_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_weight, tensor);
		} else if (parts[2] == "attn_norm_bias") {
			layers[i]->layer_output_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_bias, tensor);
		} else if (parts[2] == "q") {
			layers[i]->q = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q, tensor);
		} else if (parts[2] == "k") {
			layers[i]->k = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k, tensor);
		} else if (parts[2] == "v") {
			layers[i]->v = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v, tensor);
		} else if (parts[2] == "o") {
			layers[i]->o = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o, tensor);
		}  else if (parts[2] == "q_bias") {
			layers[i]->q_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q_bias, tensor);
		} else if (parts[2] == "k_bias") {
			layers[i]->k_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k_bias, tensor);
		} else if (parts[2] == "v_bias") {
			layers[i]->v_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v_bias, tensor);
		} else if (parts[2] == "o_bias") {
			layers[i]->o_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o_bias, tensor);
		} else if (parts[2] == "ffn_norm") {
			layers[i]->attn_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_weight, tensor);
		} else if (parts[2] == "ffn_norm_bias") {
			layers[i]->attn_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_bias, tensor);
		}
	}
}

lstm * kokoro_model::prep_lstm() {
	lstm * rnn = new lstm;
	lstm_cell * cell = new lstm_cell;
	for (int i = 0; i < 8; i++) {
		cell->weights.push_back(nullptr);
		cell->biases.push_back(nullptr);
		cell->reverse_weights.push_back(nullptr);
		cell->reverse_biases.push_back(nullptr);
	}
	rnn->cells.push_back(cell);
	rnn->bidirectional = true;
	lstms.push_back(rnn);
	return rnn;
}

void kokoro_model::prep_layers(gguf_context * meta) {
	prosody_pred = new duration_predictor;
	prosody_pred->shared_lstm = prep_lstm();
	prosody_pred->duration_proj_lstm = prep_lstm();
	text_encoder = new kokoro_text_encoder;
	decoder = new kokoro_decoder;
	decoder->generator = new kokoro_generator;
	decoder->encoder_block = new ada_residual_conv_block;
	text_encoder->out_lstm = prep_lstm();

	for (int i = 0; i < n_layers; i++) {
		layers.push_back(new albert_layer);
	}

	for (int i = 0; i < f0_n_blocks; i++) {
		ada_residual_conv_block * f0 = new ada_residual_conv_block;
		ada_residual_conv_block * n = new ada_residual_conv_block;
		prosody_pred->f0_blocks.push_back(f0);
		prosody_pred->n_blocks.push_back(n);
	}

	for (int i = 0; i < n_duration_prediction_layers; i++) {
		duration_predictor_layer* dpl = new duration_predictor_layer;
		dpl->rnn = prep_lstm();
		prosody_pred->layers.push_back(dpl);
	}

	for (int i = 0; i < n_decoder_blocks; i++) {
		decoder->decoder_blocks.push_back(new ada_residual_conv_block);
	}

	for (int i = 0; i < n_noise_blocks; i++) {
		struct kokoro_noise_residual_block * nb = build_noise_block_from_file(meta, i);
		decoder->generator->noise_blocks.push_back(nb);
	}

	for (int i = 0; i < n_upsamples; i++) {
		struct kokoro_generator_upsample_block * ub = kokoro_generator_upsample_block(meta, i);
		decoder->generator->ups.push_back(ub);
	}

	for (int i = 0; i < n_res_blocks; i++) {
		struct kokoro_generator_residual_block* rb = build_res_block_from_file(meta, "kokoro.decoder.generator.res_blocks." + std::to_string(i));
		decoder->generator->res_blocks.push_back(rb);
	}

	for (int i = 0; i < n_conv_layers; i++) {
		text_encoder->conv_layers.push_back(new kokoro_text_encoder_conv_layer);
	}
}

void kokoro_model::prep_constants(gguf_context * meta) {
	// get constants for the Albert duration prediction model
	int context_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);;
    }

    int vocab_size_key = gguf_find_key(meta, "kokoro.tokenizer.vocab_size");
    if (vocab_size_key != -1) {
        vocab_size = gguf_get_val_u32(meta, vocab_size_key);
    }

    int hidden_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
        head_size = (uint32_t) hidden_size / n_attn_heads;
    }

    int albert_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.layers");
    if (albert_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, albert_layers_key);
    }

    int recurrence_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.recurrence");
    if (recurrence_key != -1) {
        n_recurrence = gguf_get_val_u32(meta, recurrence_key);
    }

    int duration_hidden_key = gguf_find_key(meta, "kokoro.duration_predictor.hidden_size");
    if (duration_hidden_key != -1) {
        duration_hidden_size = gguf_get_val_u32(meta, duration_hidden_key);
    }

    int up_sampling_factor_key = gguf_find_key(meta, "kokoro.decoder.generator.up_sampling_factor");
    if (up_sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, up_sampling_factor_key);
    }

	int f0_n_blocks_key = gguf_find_key(meta, "kokoro.duration_predictor.f0_n_blocks");
    if (f0_n_blocks_key != -1) {
        f0_n_blocks = gguf_get_val_u32(meta, f0_n_blocks_key);
    }

   	int duration_pred_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.layers");
    if (duration_pred_layers_key != -1) {
        n_duration_prediction_layers = gguf_get_val_u32(meta, duration_pred_layers_key);
    }

	// get text and decoding configuration for generation
	int n_conv_layers_key = gguf_find_key(meta, "kokoro.text_encoder.layers");
    if (n_conv_layers_key != -1) {
        n_conv_layers = gguf_get_val_u32(meta, n_conv_layers_key);
    }

   	int n_kernels_key = gguf_find_key(meta, "kokoro.decoder.generator.kernels");
    if (n_kernels_key != -1) {
        n_kernels = gguf_get_val_u32(meta, n_kernels_key);
    }

    int n_upsamples_key = gguf_find_key(meta, "kokoro.decoder.generator.upsamples");
    if (n_upsamples_key != -1) {
        n_upsamples = gguf_get_val_u32(meta, n_upsamples_key);
    }

    int n_decoder_blocks_key = gguf_find_key(meta, "kokoro.decoder.generator.layers");
    if (n_decoder_blocks_key != -1) {
        n_decoder_blocks = gguf_get_val_u32(meta, n_decoder_blocks_key);
    }

    int out_conv_padding_key = gguf_find_key(meta, "kokoro.decoder.generator.padding");
    if (out_conv_padding_key != -1) {
        out_conv_padding = gguf_get_val_u32(meta, out_conv_padding_key);
    }

    int n_fft_key = gguf_find_key(meta, "kokoro.decoder.generator.n_fft");
    if (n_fft_key != -1) {
        true_n_fft = gguf_get_val_u32(meta, n_fft_key);
        post_n_fft = (uint32_t) true_n_fft / 2 + 1;
    }

    int stft_hop_key = gguf_find_key(meta, "kokoro.decoder.generator.hop");
    if (stft_hop_key != -1) {
        stft_hop = gguf_get_val_u32(meta, stft_hop_key);
    }
}

kokoro_ubatch kokoro_duration_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	return batch;
}

struct ggml_cgraph * kokoro_duration_runner::build_kokoro_duration_graph(kokoro_ubatch & batch) {
    init_build();
    // This '110000' number is coming from the number of nodes necessary for the longest possible sequence computed by of the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 110000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    if (!model->static_token_types) {
    	kctx->token_types = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    	ggml_set_input(kctx->token_types);
    }

    kctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->positions);

    inpL = build_albert_inputs(ctx, model, kctx->inp_tokens, kctx->positions, kctx->token_types);
    ggml_set_name(inpL, "albert_embeddings");
    cur = inpL;

    for (int r = 0; r < model->n_recurrence; r++) {
    	for (int l = 0; l < model->n_layers; l++) {
	        struct ggml_tensor * residual = cur ;
	        struct ggml_tensor * attn_out;

	        // self-attention
	        {
	            struct ggml_tensor * Qcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->q, cur), model->layers[l]->q_bias);
	            struct ggml_tensor * Kcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->k, cur), model->layers[l]->k_bias);
	            struct ggml_tensor * Vcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->v, cur), model->layers[l]->v_bias);

				Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
	            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

	            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
	            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
	            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

	            // Kokoro 的 duration predictor 不是自回归模型，不需要 causal mask；
	            // 且当前实现里 mask 一直是全 0。这里直接传 nullptr，避免每次推理都构造/填充 mask。
	            kq = ggml_soft_max_ext(ctx, kq, nullptr, model->scale, 0.0f);

	            struct ggml_tensor * v = ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
	            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
	            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
	            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
	            attn_out = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->o, attn_out), model->layers[l]->o_bias);
	        }
	        cur = ggml_add(ctx, attn_out, residual);
	        cur = build_albert_norm(ctx, cur, model->layers[l]->attn_norm_weight, model->layers[l]->attn_norm_bias);

	        struct ggml_tensor * residualffn = cur;

	        // ffn
	        {
	        	cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn, cur), model->layers[l]->ffn_bias));
	        	cur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn_out, cur), model->layers[l]->ffn_out_bias);
	        }

			cur = ggml_add(ctx, cur, residualffn);
			cur = build_albert_norm(ctx, cur, model->layers[l]->layer_output_norm_weight, model->layers[l]->layer_output_norm_bias);
	    }
        ggml_build_forward_expand(gf, cur);
    }

    // duration / prosody prediction
    cur = ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->albert_encode, cur), model->prosody_pred->albert_encode_bias);

	struct ggml_tensor * style_half = ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0]/2, voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]));

	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);

    for (auto l : model->prosody_pred->layers) {
    	cur = build_lstm(ctx, cur, l->rnn, batch.n_tokens, gf);

    	struct ggml_tensor * gamma = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_gamma_weight, style_half), l->ada_norm_gamma_bias);
    	struct ggml_tensor * beta = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_beta_weight, style_half), l->ada_norm_beta_bias);

    	cur = ggml_norm(ctx, cur, 0.00001);

    	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
    	cur = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
    	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);
    }

    struct ggml_tensor * d = ggml_cont(ctx, cur);
    ggml_set_name(d, "duration_hidden_states");
    ggml_build_forward_expand(gf, d);

    struct ggml_tensor * len;
    cur = build_lstm(ctx, cur, model->prosody_pred->duration_proj_lstm, batch.n_tokens, gf);
    cur = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->duration_proj, cur), model->prosody_pred->duration_proj_bias));
    // If we were to support speed we would add a constant tensor for the speed and divide here.
    len = ggml_round(ctx, ggml_sum_rows(ctx, cur));
    len = ggml_clamp(ctx, ggml_round(ctx, ggml_sum_rows(ctx, cur)), 1.0f, 50.0f);

    ggml_build_forward_expand(gf, len);

    free_build();

    return gf;
}

void kokoro_duration_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_duration_graph(batch);
    kctx->prep_schedule(gf);
}

void kokoro_duration_runner::set_inputs(kokoro_ubatch & batch) {
	ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(kctx->inp_tokens));
	// 说明：Vulkan 后端下 tensor->data 可能位于 GPU 内存，CPU 侧不可直接写入。
	// 这里在 CPU 侧准备 positions 并通过 backend API 写入。
	static thread_local std::vector<uint32_t> positions_buf;
	positions_buf.resize(batch.n_tokens);
	for (uint32_t i = 0; i < batch.n_tokens; i++) {
		positions_buf[i] = i;
	}
	ggml_backend_tensor_set(kctx->positions, positions_buf.data(), 0, positions_buf.size() * sizeof(uint32_t));
}

void kokoro_duration_runner::run(kokoro_ubatch & batch) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    ggml_backend_sched_reset(kctx->sched);
    const int64_t t_after_sched_reset_us = timings ? tts_time_us() : 0;

    size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    size_t new_size = model->max_context_length * (model->duration_hidden_size + model->style_half_size) * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
	    if (kctx->buf_output) {
	        ggml_backend_buffer_free(kctx->buf_output);
	        kctx->buf_output = nullptr;
	        kctx->logits = nullptr;
	    }
	    kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
	}

    prev_size = kctx->buf_len_output ? ggml_backend_buffer_get_size(kctx->buf_len_output) : 0;
    new_size = model->max_context_length * sizeof(float);

    if (!kctx->buf_len_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_len_output);
            kctx->buf_len_output = nullptr;
            kctx->lens = nullptr;
        }

        kctx->buf_len_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }


    batch.resp->hidden_states = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);
    batch.resp->lengths = (float *) ggml_backend_buffer_get_base(kctx->buf_len_output);
    ggml_backend_buffer_clear(kctx->buf_len_output, 0);

    const int64_t t_after_buffers_us = timings ? tts_time_us() : 0;

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_duration_graph(batch);
    const int64_t t_after_graph_build_us = timings ? tts_time_us() : 0;

    // the output is always the last tensor in the graph
    struct ggml_tensor * lens = gf->nodes[gf->n_nodes - 1];
    // the reused duration hidden states are computed before a node chunk which has a size that is sequence length dependent
    struct ggml_tensor * hidden_states = gf->nodes[gf->n_nodes - 22 - 52 * batch.n_tokens];
    kokoro_force_inputs_backend(kctx, gf);
    kokoro_force_custom_views_cpu(kctx, gf);
    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    const int64_t t_after_sched_alloc_us = timings ? tts_time_us() : 0;

    set_inputs(batch);

    const int64_t t_after_set_inputs_us = timings ? tts_time_us() : 0;

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);
    const int64_t t_after_compute_call_us = timings ? tts_time_us() : 0;

    kctx->get_ggml_node_data(lens, batch.resp->lengths, batch.n_tokens*sizeof(float), kctx->buf_len_output);
    const int64_t t_after_get_lens_us = timings ? tts_time_us() : 0;
    kctx->get_ggml_node_data(hidden_states, batch.resp->hidden_states, batch.n_tokens*(model->duration_hidden_size+model->style_half_size)*sizeof(float));
    const int64_t t_after_get_states_us = timings ? tts_time_us() : 0;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    const int64_t t_after_sync_us = timings ? tts_time_us() : 0;
    batch.resp->n_outputs = batch.n_tokens;

    if (timings) {
        uint32_t total_frames = 0;
        for (size_t i = 0; i < batch.n_tokens; ++i) {
            total_frames += (uint32_t) batch.resp->lengths[i];
        }

        fprintf(stderr,
                "[kokoro][timings] duration: sched_reset=%.2fms outbuf=%.2fms graph_build=%.2fms sched_alloc=%.2fms set_inputs=%.2fms compute=%.2fms get_len=%.2fms get_states=%.2fms sync=%.2fms total=%.2fms (tokens=%zu frames=%u)\n",
                us_to_ms(t_after_sched_reset_us - t_start_us),
                us_to_ms(t_after_buffers_us - t_after_sched_reset_us),
                us_to_ms(t_after_graph_build_us - t_after_buffers_us),
                us_to_ms(t_after_sched_alloc_us - t_after_graph_build_us),
                us_to_ms(t_after_set_inputs_us - t_after_sched_alloc_us),
                us_to_ms(t_after_compute_call_us - t_after_set_inputs_us),
                us_to_ms(t_after_get_lens_us - t_after_compute_call_us),
                us_to_ms(t_after_get_states_us - t_after_get_lens_us),
                us_to_ms(t_after_sync_us - t_after_get_states_us),
                us_to_ms(t_after_sync_us - t_start_us),
                batch.n_tokens,
                total_frames);
    }
}

kokoro_ubatch kokoro_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	batch.resp = new kokoro_duration_response;
	batch.resp->n_outputs = model->max_context_length;
	kctx->total_duration = model->max_context_length * model->max_duration_per_token;
	kctx->sequence_length = model->max_context_length;
	std::vector<float> lengths;
	lengths.reserve(model->max_context_length);
	for (int i = 0; i < model->max_context_length; i++) {
		lengths.push_back(50.0f);
	}
	batch.resp->lengths = lengths.data();
	return batch;
}

struct ggml_cgraph * kokoro_runner::build_kokoro_graph(kokoro_ubatch & batch) {
    init_build();
    // This '570000' number is coming from the number of nodes necessary for the longest possible sequence computed by the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 570000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    struct ggml_tensor * style_half = ggml_view_1d(ctx, voice, voice->ne[0]/2, voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]);
    struct ggml_tensor * cur;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    kctx->duration_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kctx->total_duration, kctx->sequence_length);
    ggml_set_input(kctx->duration_mask);

    kctx->duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->duration_hidden_size + model->style_half_size, kctx->sequence_length);
    ggml_set_input(kctx->duration_pred);

    // seeing as we are setting the inputs for these, we shouldn't need to perform tranpositions here
    cur = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)), ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_pred)));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    cur = build_lstm(ctx, cur, model->prosody_pred->shared_lstm, cur->ne[1], gf);


    struct ggml_tensor * f0_curve = cur;
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    for (auto block : model->prosody_pred->f0_blocks) {
    	f0_curve = build_ada_residual_conv(ctx, f0_curve, block, style_half, model->sqrt_tensor);
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = ggml_mul_mat(ctx, model->prosody_pred->f0_proj_kernel, f0_curve);
    f0_curve = squeeze_3d_2d_e0(ctx, f0_curve);
    f0_curve = ggml_add(ctx, f0_curve, model->prosody_pred->f0_proj_bias);
    ggml_set_name(f0_curve, "f0_out");

    struct ggml_tensor * n = cur;
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
    for (auto block : model->prosody_pred->n_blocks) {
		n = build_ada_residual_conv(ctx, n, block, style_half, model->sqrt_tensor);
    }
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
	n = ggml_mul_mat(ctx, model->prosody_pred->n_proj_kernel, n);
	n = squeeze_3d_2d_e0(ctx, n);
	n = ggml_add(ctx, n, model->prosody_pred->n_proj_bias);
	ggml_set_name(n, "n_out");
	ggml_build_forward_expand(gf, n);

	// kokoro text encoding;
	struct ggml_tensor * asr;
	//struct ggml_tensor * embd;
	{
		cur = ggml_get_rows(ctx, model->text_encoder->embd, kctx->inp_tokens);

		for (auto l : model->text_encoder->conv_layers) {
			cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_conv_1d(ctx, l->conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, 2, 1), l->conv_bias)));
			cur = ggml_norm(ctx, cur, 0.00001);
			cur = ggml_add(ctx, ggml_mul(ctx, cur, l->norm_gamma), l->norm_beta);
			cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
		}

		cur = build_lstm(ctx, cur, model->text_encoder->out_lstm, kctx->sequence_length, gf);
		asr = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)));
	}

	// decoding and generation prep
	struct ggml_tensor * asr_res;
	struct ggml_tensor * f0;
	struct ggml_tensor * n_base;
	struct ggml_tensor * style_half2 = ggml_view_1d(ctx, voice, voice->ne[0]/2, (batch.n_tokens - 3) * voice->nb[1]);

	{
		f0 = ggml_add(ctx, ggml_conv_1d(ctx, model->decoder->f0_conv, f0_curve, 2, 1, 1), model->decoder->f0_conv_bias);
		n_base = ggml_add(ctx, ggml_conv_1d(ctx, model->decoder->n_conv, n, 2, 1, 1), model->decoder->n_conv_bias);
		cur = ggml_concat(ctx, ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)), f0, 1), n_base, 1);
		cur = build_ada_residual_conv(ctx, cur, model->decoder->encoder_block, style_half2, model->sqrt_tensor);
		ggml_build_forward_expand(gf, cur);

		asr_res = ggml_mul_mat(ctx, model->decoder->asr_conv, asr);
		// 说明：同样避免 src1 非连续视图参与 broadcast（ggml 0.9.4 CPU 路径会断言失败）。
		asr_res = ggml_add(ctx, asr_res, ggml_cont(ctx, ggml_transpose(ctx, model->decoder->asr_conv_bias)));

		asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
		for (auto l : model->decoder->decoder_blocks) {
			cur = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, cur, asr_res, 1), f0, 1), n_base, 1 );
			cur = build_ada_residual_conv(ctx, cur, l, style_half2, model->sqrt_tensor);
			ggml_build_forward_expand(gf, cur);
		}
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
	}

	kctx->window_sq_sum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kctx->total_duration*model->up_sampling_factor);
	ggml_set_input(kctx->window_sq_sum);

	// run generation
	cur = build_generator(ctx, &*model, kctx, cur, style_half2, f0_curve, model->decoder->generator, (int)kctx->sequence_length, kctx->window_sq_sum, gf);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void kokoro_runner::prepare_post_load() {
    propagate_voice_setting();
	model->post_load_assign();
	drunner->prepare_post_load();
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_graph(batch);
    kctx->prep_schedule(gf);
    // batch.resp 由 new 分配，必须用 delete 释放（避免潜在堆损坏）。
    delete batch.resp;
    batch.resp = nullptr;
}

kokoro_gen_input_timings kokoro_runner::set_inputs(kokoro_ubatch & batch, uint32_t total_size) {
    kokoro_gen_input_timings out{};

    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    // 说明：Vulkan 后端下 tensor->data 可能位于 GPU 内存，CPU 侧不可直接写入。
    // 这里在 CPU 侧准备 uv/noise 输入，并通过 backend API 写入。
    const size_t uv_bytes = ggml_nbytes(kctx->uv_noise_data);
    const size_t uv_elems = uv_bytes / sizeof(float);
    static thread_local std::vector<float> uv_noise_buf;
    uv_noise_buf.resize(uv_elems);
    if (uv_elems >= 4) {
        uv_noise_buf[0] = model->voice_threshold;
        uv_noise_buf[1] = model->noise_std;
        uv_noise_buf[2] = model->sin_amp;
        uv_noise_buf[3] = model->sin_amp / 3.0f;
    }
    const size_t uv_rand_elems = uv_elems > 4 ? (uv_elems - 4) : 0;
    if (uv_rand_elems > 0) {
        random_uniform_gen((int) uv_rand_elems, uv_noise_buf.data() + 4);
    }
    ggml_backend_tensor_set(kctx->uv_noise_data, uv_noise_buf.data(), 0, uv_bytes);

    const int64_t t_after_noise_us = timings ? tts_time_us() : 0;

    const size_t window_bytes = ggml_nbytes(kctx->window_sq_sum);
    const size_t window_elems = window_bytes / sizeof(float);
    static thread_local std::vector<float> window_sq_sum_buf;
    window_sq_sum_buf.resize(window_elems);
    if (model->decoder->generator->window_host.size() != model->true_n_fft) {
        model->decoder->generator->window_host.clear();
        model->decoder->generator->window_host.reserve(model->true_n_fft);
        hann_window(model->true_n_fft, model->decoder->generator->window_host);
    }
    compute_window_squared_sum(model->true_n_fft, model->stft_hop,
                               total_size * model->up_sampling_factor / model->stft_hop,
                               window_sq_sum_buf.data(),
                               model->decoder->generator->window_host.data());
    ggml_backend_tensor_set(kctx->window_sq_sum, window_sq_sum_buf.data(), 0, window_bytes);

    const int64_t t_after_window_us = timings ? tts_time_us() : 0;

    kctx->sequence_length = batch.n_tokens;
    kctx->total_duration  = total_size;
    ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0,
                            batch.n_tokens * ggml_element_size(kctx->inp_tokens));
    ggml_backend_tensor_set(kctx->duration_pred, batch.resp->hidden_states, 0,
                            batch.n_tokens * (model->duration_hidden_size + model->style_half_size) *
                                ggml_element_size(kctx->duration_pred));

    const int64_t t_after_backend_sets_us = timings ? tts_time_us() : 0;

    // duration_mask 的每一行都是“连续的一段 1 + 其他位置为 0”。
    // 这里在 CPU 侧构造完整 mask，再写入 backend。
    const size_t duration_mask_bytes = ggml_nbytes(kctx->duration_mask);
    const size_t duration_mask_elems = duration_mask_bytes / sizeof(float);
    static thread_local std::vector<float> duration_mask_buf;
    duration_mask_buf.resize(duration_mask_elems);
    std::fill(duration_mask_buf.begin(), duration_mask_buf.end(), 0.0f);

    const size_t row_len = (size_t) total_size;
    if (row_len == 0 || batch.n_tokens == 0) {
        ggml_backend_tensor_set(kctx->duration_mask, duration_mask_buf.data(), 0, duration_mask_bytes);
        const int64_t t_end_us = timings ? tts_time_us() : 0;
        if (timings) {
            out.noise_ms         = us_to_ms(t_after_noise_us - t_start_us);
            out.window_sq_sum_ms = us_to_ms(t_after_window_us - t_after_noise_us);
            out.tensor_set_ms    = us_to_ms(t_after_backend_sets_us - t_after_window_us);
            out.duration_mask_ms = us_to_ms(t_end_us - t_after_backend_sets_us);
            out.total_ms         = us_to_ms(t_end_us - t_start_us);
        }
        return out;
    }
    if (duration_mask_elems != (size_t) batch.n_tokens * row_len) {
        TTS_ABORT("duration_mask size mismatch: elems=%zu expected=%zu (tokens=%u, frames=%zu)\n",
                  duration_mask_elems, (size_t) batch.n_tokens * row_len, batch.n_tokens, row_len);
    }

    uint32_t running = 0;
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        const uint32_t len   = (uint32_t) batch.resp->lengths[i];
        const uint32_t start = running;
        const uint32_t end   = std::min(running + len, total_size);
        if (end > start) {
            std::fill(duration_mask_buf.data() + (size_t) i * row_len + start,
                      duration_mask_buf.data() + (size_t) i * row_len + end,
                      1.0f);
        }
        running = end;
    }
    ggml_backend_tensor_set(kctx->duration_mask, duration_mask_buf.data(), 0, duration_mask_bytes);

    if (timings) {
        const int64_t t_end_us = tts_time_us();
        out.noise_ms         = us_to_ms(t_after_noise_us - t_start_us);
        out.window_sq_sum_ms = us_to_ms(t_after_window_us - t_after_noise_us);
        out.tensor_set_ms    = us_to_ms(t_after_backend_sets_us - t_after_window_us);
        out.duration_mask_ms = us_to_ms(t_end_us - t_after_backend_sets_us);
        out.total_ms         = us_to_ms(t_end_us - t_start_us);
    }

    return out;
}

void kokoro_runner::run(kokoro_ubatch & batch, tts_response & outputs) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

	batch.resp = new kokoro_duration_response;
	drunner->run(batch);

    const int64_t t_after_duration_us = timings ? tts_time_us() : 0;

	ggml_backend_sched_reset(kctx->sched);
    const int64_t t_after_sched_reset_us = timings ? tts_time_us() : 0;

    const size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    uint32_t total_length = 0;
    for (int i = 0; i < batch.resp->n_outputs; i++) {
    	total_length += (uint32_t) batch.resp->lengths[i];
    }
    const size_t new_size = total_length * model->up_sampling_factor * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_output);
            kctx->buf_output = nullptr;
            kctx->logits = nullptr;
        }
        kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    outputs.data = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);

    kctx->sequence_length = batch.n_tokens;
	kctx->total_duration = total_length;

    const int64_t t_after_outbuf_us = timings ? tts_time_us() : 0;

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_graph(batch);
    const int64_t t_after_graph_build_us = timings ? tts_time_us() : 0;

    // the output is always the last tensor in the graph
    struct ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

    const bool force_vulkan_gen = kokoro_env_force_vulkan_gen();
    const bool prefer_cpu_gen = tts_backend_is_vulkan(kctx->backend) && kokoro_graph_has_custom_ops(gf) && !force_vulkan_gen;
    if (prefer_cpu_gen) {
        // 说明：当前 ggml-vulkan 对混合自定义算子的图存在崩溃风险，先回退到 CPU 保证可用。
        kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
        fprintf(stderr,
                "[kokoro] Vulkan 后端检测到自定义算子，生成阶段切换为 CPU 以避免构图崩溃；如需强制 Vulkan 可设置 TTS_VK_FORCE_GEN=1。\n");
    } else {
        kokoro_force_inputs_backend(kctx, gf);
    }
    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    const int64_t t_after_sched_alloc_us = timings ? tts_time_us() : 0;

    const int64_t t_before_set_inputs_us = timings ? tts_time_us() : 0;
    const kokoro_gen_input_timings input_timings = set_inputs(batch, total_length);
    const int64_t t_after_set_inputs_us = timings ? tts_time_us() : 0;

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);
    const int64_t t_after_compute_call_us = timings ? tts_time_us() : 0;

    kctx->get_ggml_node_data(output, outputs.data, new_size);
    const int64_t t_after_get_call_us = timings ? tts_time_us() : 0;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    const int64_t t_after_sync_us = timings ? tts_time_us() : 0;
    outputs.n_outputs = total_length*model->up_sampling_factor;
    // batch.resp 由 new 分配，必须用 delete 释放（避免潜在堆损坏）。
    delete batch.resp;
    batch.resp = nullptr;

    if (timings) {
        const double audio_sec = outputs.n_outputs > 0 ? (double) outputs.n_outputs / (double) model->sample_rate : 0.0;
        const double total_sec = (double) (t_after_sync_us - t_start_us) / 1000000.0;
        const double rtf = audio_sec > 0.0 ? total_sec / audio_sec : 0.0;

        // 输入准备的细分耗时（该行不包含打印本身的时间）。
        fprintf(stderr,
                "[kokoro][timings] gen_inputs: call=%.2fms noise=%.2fms window_sq_sum=%.2fms tensor_set=%.2fms duration_mask=%.2fms total=%.2fms (tokens=%zu frames=%u)\n",
                us_to_ms(t_after_set_inputs_us - t_before_set_inputs_us),
                input_timings.noise_ms,
                input_timings.window_sq_sum_ms,
                input_timings.tensor_set_ms,
                input_timings.duration_mask_ms,
                input_timings.total_ms,
                batch.n_tokens,
                total_length);

        fprintf(stderr,
                "[kokoro][timings] generator: duration=%.2fms sched_reset=%.2fms outbuf=%.2fms graph_build=%.2fms sched_alloc=%.2fms set_inputs=%.2fms compute=%.2fms get=%.2fms sync=%.2fms total=%.2fms audio=%.3fs rtf=%.3f (tokens=%zu frames=%u samples=%zu)\n",
                us_to_ms(t_after_duration_us - t_start_us),
                us_to_ms(t_after_sched_reset_us - t_after_duration_us),
                us_to_ms(t_after_outbuf_us - t_after_sched_reset_us),
                us_to_ms(t_after_graph_build_us - t_after_outbuf_us),
                us_to_ms(t_after_sched_alloc_us - t_after_graph_build_us),
                us_to_ms(t_after_set_inputs_us - t_after_sched_alloc_us),
                us_to_ms(t_after_compute_call_us - t_after_set_inputs_us),
                us_to_ms(t_after_get_call_us - t_after_compute_call_us),
                us_to_ms(t_after_sync_us - t_after_get_call_us),
                us_to_ms(t_after_sync_us - t_start_us),
                audio_sec,
                rtf,
                batch.n_tokens,
                total_length,
                outputs.n_outputs);
    }
    return;
}

void kokoro_runner::assign_weight(const char * name, ggml_tensor & tensor) {
    const string_view name_sv{ name };
    GGML_ASSERT(name_sv.starts_with("kokoro."));
    const string trimmed{ name_sv.substr(sizeof("kokoro.") - 1) };
    model->assign_weight(trimmed.c_str(), tensor);
}

/*
 * #tokenize_chunks is used to split up a larger than max context size (512) token prompt into discrete
 * blocks for generation. This solution, in accordance with Kokoro's pyTorch implementation, splits
 * the prompt by sentence when possible (this can result in slower inference but generally produces cleaner
 * speech). If a disinct sentence is too long, then it splits at the nearest space.
 */
std::vector<std::vector<uint32_t>> kokoro_runner::tokenize_chunks(std::vector<std::string> clauses) {
	std::vector<std::vector<uint32_t>> chunks;
	for (auto clause : clauses) {
		clause = strip(clause);
		if (clause.empty()) {
			continue;
		}
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(clause, tokens);
		// if there are more clause tokens than the max context length then try to split by space tokens.
		// To be protective, split mid-word when there are no spaces (this should never happen).
		if (tokens.size() > model->max_context_length - 2) {
			// we skip the first token here becuase it is the bos token.
			int last_space_token = 1;
			int last_split = 1;
			for (int i = 1; i < tokens.size(); i++) {
				if (tokens[i] == model->space_token_id) {
					last_space_token = i;
				}
				if ((i - last_split) + chunks.back().size() >= model->max_context_length - 1) {
					if (last_space_token > last_split) {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + last_space_token);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = last_space_token;
					} else {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + i + 1);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = i + 1;
					}
				}
			}
			if (last_split + 1 < tokens.size()) {
				std::vector<uint32_t> portion = { model->bos_token_id };
				portion.insert(portion.end(), tokens.begin() + last_split, tokens.end());
				portion.push_back(model->eos_token_id);
				chunks.push_back(portion);
			}
		} else {
			tokens.push_back(model->eos_token_id);
			chunks.push_back(tokens);
		}
	}
	return chunks;
}

void kokoro_runner::propagate_voice_setting() {
    if (voice.empty()) {
        if (model->voices.contains("af_heart")) {
            voice = "af_heart";
        } else if (!model->voices.empty()) {
            voice = model->voices.begin()->first;
        }
    }
    if (voice.empty() || !model->voices.contains(voice)) {
        TTS_ABORT("Failed to find Kokoro voice '%s' aborting.\n", voice.c_str());
    }
    kctx->voice          = voice;
    drunner->kctx->voice = voice;
}

void kokoro_runner::generate(const char * prompt, tts_response & response, const generation_configuration & config) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    voice = config.voice;
    propagate_voice_setting();
    std::string normalized{prompt};
    normalized = replace_any(normalized, "\n", " ");

    const int64_t t_after_normalize_us = timings ? tts_time_us() : 0;

    std::string phonemized_prompt;
    // Enable built-in zh phonemization when either:
    // - voice is Mandarin (z*)  OR
    // - prompt contains CJK characters
    const bool contains_cjk = kokoro_contains_cjk(normalized);
    const bool use_multilingual = (!voice.empty() && voice[0] == 'z') || contains_cjk;
    if (use_multilingual) {
        phonemized_prompt = kokoro_phonemize_multilingual(normalized, phmzr);
    } else {
        phonemized_prompt = phmzr->text_to_phonemes(normalized);
    }

  	const int64_t t_after_phonemize_us = timings ? tts_time_us() : 0;

    double tokenize_ms = 0.0;
    double run_ms = 0.0;
    size_t n_tokens_total = 0;

  	// Kokoro users a utf-8 single character tokenizer so if the size of the prompt is smaller than the max context length without the
  	// beginning of sentence and end of sentence tokens then we can compute it all at once.
  	if (phonemized_prompt.size() < model->max_context_length - 2) {
  		// we preserved punctuation and Kokoro interprets these tokens as end of sentence tokens, so we have to remove them for all-at-once compute.
  		phonemized_prompt = strip(replace_any(phonemized_prompt, ".!?", ""));
  		if (phonemized_prompt.empty()) {
  			return;
  		}
        const int64_t t_before_tokenize_us = timings ? tts_time_us() : 0;
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(phonemized_prompt, tokens);
		tokens.push_back(model->eos_token_id);
        const int64_t t_after_tokenize_us = timings ? tts_time_us() : 0;
        tokenize_ms = us_to_ms(t_after_tokenize_us - t_before_tokenize_us);
        n_tokens_total = tokens.size();

		kokoro_ubatch batch;
		batch.n_tokens = tokens.size();
		batch.input_tokens = tokens.data();
        const int64_t t_before_run_us = timings ? tts_time_us() : 0;
		run(batch, response);
        const int64_t t_after_run_us = timings ? tts_time_us() : 0;
        run_ms = us_to_ms(t_after_run_us - t_before_run_us);
  	} else {
  		// TODO: determine the performance to memory trade off in using a batched compute approach verse this chunking approach.
  		// This approach is likely to be slower than a batched approach, but given the already huge memory overhead of Kokoro's graph it
  		// might be preferable to use this chunking approach.
  		std::vector<std::string> clauses = split(phonemized_prompt, ".!?");
        const int64_t t_before_tokenize_us = timings ? tts_time_us() : 0;
        const auto chunks = tokenize_chunks(clauses);
        const int64_t t_after_tokenize_us = timings ? tts_time_us() : 0;
        tokenize_ms = us_to_ms(t_after_tokenize_us - t_before_tokenize_us);
        for (const auto & t : chunks) {
            n_tokens_total += t.size();
        }

        const int64_t t_before_run_us = timings ? tts_time_us() : 0;
  		for (const auto & tokens : chunks) {
			kokoro_ubatch batch;
			batch.n_tokens = tokens.size();
			batch.input_tokens = (uint32_t *) tokens.data();
			tts_response partial{};
			run(batch, partial);
			append_to_response(response, partial);
		}
        const int64_t t_after_run_us = timings ? tts_time_us() : 0;
        run_ms = us_to_ms(t_after_run_us - t_before_run_us);
  	}

    if (timings) {
        const int64_t t_end_us = tts_time_us();
        const char * const mode = use_multilingual ? "multilingual" : "tts_phonemizer";
        fprintf(stderr,
                "[kokoro][timings] frontend: normalize=%.2fms phonemize=%.2fms tokenize=%.2fms run=%.2fms total=%.2fms (mode=%s cjk=%d prompt_bytes=%zu phoneme_bytes=%zu tokens=%zu)\n",
                us_to_ms(t_after_normalize_us - t_start_us),
                us_to_ms(t_after_phonemize_us - t_after_normalize_us),
                tokenize_ms,
                run_ms,
                us_to_ms(t_end_us - t_start_us),
                mode,
                contains_cjk ? 1 : 0,
                std::strlen(prompt),
                phonemized_prompt.size(),
                n_tokens_total);
    }
}

std::vector<std::string_view> kokoro_runner::list_voices() {
    std::vector<std::string_view> voices;
    voices.reserve(model->voices.size());
    for (const auto & kv : model->voices) {
        voices.emplace_back(kv.first);
    }
    return voices;
}

struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_duration_context * kctx = new kokoro_duration_context(model, n_threads);
    if (!use_cpu) {
        kctx->backend = tts_backend_init_accel();
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_duration_nodes()*5 + ggml_graph_overhead_custom(model->max_duration_nodes()*5, false));
    return kctx;
}


struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_context * kctx = new kokoro_context(model, n_threads);
    if (!use_cpu) {
        kctx->backend = tts_backend_init_accel();
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_gen_nodes()*30 + ggml_graph_overhead_custom(model->max_gen_nodes()*30, false));
    return kctx;
}
