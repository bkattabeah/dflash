import time
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .model_mlx import DFlashDraftModel

# Optional: GatedDeltaNet support for Qwen3.5 (recurrent layers)
try:
    import mlx_lm.models.gated_delta as _gd_mod

    _HAS_GDN = True
except ImportError:
    _HAS_GDN = False


class _LayerHook:
    """Wraps a layer to capture its output."""

    def __init__(self, layer, idx: int, storage: list):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        self._storage[self._idx] = out = self._layer(*args, **kwargs)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model: nn.Module):
    """Return the transformer layers list, handling different model structures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _patch_model(model: nn.Module, layer_ids: Tuple[int, ...]) -> None:
    """Patch model to capture hidden states from specific layers."""
    if hasattr(model, "_hidden_states"):
        return
    model._hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._hidden_states)


class _GDNStateCapture:
    """Captures intermediate recurrent states from GatedDeltaNet layers.

    Monkey-patches gated_delta_update to use the ops path during the verify
    forward pass, saving the recurrent state after each token position.
    This allows rolling back to any intermediate state without re-running
    the model (1x forward pass instead of 2x).
    """

    def __init__(self):
        self.states = []  # Per GDN layer: [state_0, state_1, ..., state_T]
        self.conv_data = []  # Per GDN layer: (conv_input, kernel_size)
        self._patched_modules = []
        self._orig_gdn_call = None

    def enter(self):
        self.states.clear()
        self.conv_data.clear()
        import sys
        from mlx_lm.models.qwen3_5 import GatedDeltaNet, RMSNormGated

        self._patched_modules = []
        capture = self
        orig_fn = _gd_mod.gated_delta_update

        def _capturing_update(
            q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
        ):
            beta = mx.sigmoid(b)
            g = _gd_mod.compute_g(A_log, a, dt_bias)
            B, T, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            if state is None:
                state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)
            if (r := Hv // Hk) > 1:
                q = mx.repeat(q, r, -2)
                k = mx.repeat(k, r, -2)
            intermediates = [state]
            ys = []
            for t in range(T):
                y, state = _gd_mod._gated_delta_step_ops(
                    q[:, t],
                    k[:, t],
                    v[:, t],
                    g[:, t],
                    beta[:, t],
                    state,
                    None if mask is None else mask[:, t],
                )
                ys.append(y)
                intermediates.append(state)
            capture.states.append(intermediates)
            return mx.stack(ys, axis=1), state

        # Patch mlx_lm model modules that imported gated_delta_update
        for name, mod in sys.modules.items():
            if mod is None or not name.startswith("mlx_lm.models."):
                continue
            try:
                fn = mod.__dict__.get("gated_delta_update")
            except Exception:
                continue
            if fn is orig_fn:
                self._patched_modules.append((mod, fn))
                mod.gated_delta_update = _capturing_update

        # Patch GatedDeltaNet.__call__ to capture conv_input
        self._orig_gdn_call = GatedDeltaNet.__call__

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            if self_layer.sharding_group is not None:
                from mlx_lm.models.qwen3_5 import sum_gradients
                inputs = sum_gradients(self_layer.sharding_group)(inputs)
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(B, S, self_layer.num_v_heads, self_layer.head_v_dim)
            b = self_layer.in_proj_b(inputs)
            a = self_layer.in_proj_a(inputs)
            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros((B, self_layer.conv_kernel_size - 1, self_layer.conv_dim), dtype=inputs.dtype)
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            # CAPTURE conv_input
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1):]
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            # This calls the patched gated_delta_update (captures intermediate states)
            out, state = _capturing_update(
                q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask, use_kernel=False
            )
            if cache is not None:
                cache[1] = state
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if self_layer.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        GatedDeltaNet.__call__ = _capturing_gdn_call

    def exit(self):
        for mod, orig in self._patched_modules:
            mod.gated_delta_update = orig
        self._patched_modules.clear()
        if self._orig_gdn_call is not None:
            from mlx_lm.models.qwen3_5 import GatedDeltaNet
            GatedDeltaNet.__call__ = self._orig_gdn_call
            self._orig_gdn_call = None

    def rollback(self, cache, accepted, trim):
        """Roll back cache to state after accepted+1 tokens."""
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                # Fix recurrent state from captured intermediates
                c.cache[1] = self.states[j][accepted + 1]
                # Fix conv state: extract correct slice from captured conv_input
                conv_input, K = self.conv_data[j]
                c.cache[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1


@dataclass
class SpeculativeResult:
    """Result from one speculative decode step."""

    draft_tokens: List[int]
    verified_tokens: List[int]
    accepted: int
    new_hidden: mx.array
    verified_logprobs: List[mx.array]


def speculative_step(
    model: nn.Module,
    draft: DFlashDraftModel,
    target_cache,
    draft_cache,
    hidden: mx.array,
    current_token: int,
    sampler: Callable,
    block_size: int,
) -> SpeculativeResult:
    """Execute one DFlash speculative decode step.

    This is the core logic shared by standalone generation and server integration.

    Args:
        model: Target model (must be patched with _patch_model)
        draft: DFlash draft model (must be bound to target)
        target_cache: Target model's KV cache
        draft_cache: Draft model's KV cache
        hidden: Target hidden states from previous step [B, S, concat_dim]
        current_token: The last accepted token id
        sampler: Sampling function (logits -> token)
        block_size: Number of tokens to draft

    Returns:
        SpeculativeResult with draft/verified tokens, acceptance count,
        new hidden states, and verified logprobs
    """
    mask_id = int(draft.config.mask_token_id)
    target_can_trim = can_trim_prompt_cache(target_cache)

    # Phase 1: Draft — single forward pass
    with mx.stream(generation_stream):
        block = mx.array([[current_token] + [mask_id] * (block_size - 1)])
        draft_logits = draft(block, hidden, draft_cache)
        # Trim draft cache if it advanced past target cache
        target_offset = getattr(target_cache[0], "offset", None)
        if target_offset is not None:
            trim_n = draft_cache[0].offset - target_offset
            if trim_n > 0:
                trim_prompt_cache(draft_cache, trim_n)
        draft_tokens = sampler(draft_logits[:, 1 - block_size:])
    mx.async_eval(draft_tokens)

    # Phase 2: Target verify
    capture = None
    if not target_can_trim:
        if _HAS_GDN:
            capture = _GDNStateCapture()
            capture.enter()

    with mx.stream(generation_stream):
        verify_input = mx.concatenate(
            [mx.array([[current_token]]), draft_tokens], axis=1
        )
        main_logits = model(verify_input, cache=target_cache)
        new_hidden = mx.concatenate(model._hidden_states, axis=-1)
        verified_tokens = sampler(main_logits)

    if capture is not None:
        capture.exit()

    mx.async_eval(verified_tokens, new_hidden)
    mx.eval(verified_tokens, draft_tokens, new_hidden)

    d_list = draft_tokens[0].tolist()
    v_list = verified_tokens[0].tolist()

    # Phase 3: Accept/reject
    n = 0
    while n < len(d_list) and v_list[n] == d_list[n]:
        n += 1

    # Phase 3b: Cache rewind
    trim = block_size - n - 1
    if trim > 0:
        if target_can_trim:
            trim_prompt_cache(target_cache, trim)
        elif capture is not None:
            capture.rollback(target_cache, n, trim)
        trim_prompt_cache(draft_cache, trim)
        for c in draft_cache:
            if hasattr(c, "keys") and c.keys is not None and c.keys.shape[-2] > c.offset:
                c.keys = c.keys[..., :c.offset, :]
                c.values = c.values[..., :c.offset, :]

    # Build per-position logprobs from main_logits
    logits_2d = main_logits.squeeze(0)
    all_lp = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
    verified_logprobs = [all_lp[i:i + 1] for i in range(block_size)]

    return SpeculativeResult(
        draft_tokens=d_list,
        verified_tokens=v_list,
        accepted=n,
        new_hidden=new_hidden[:, :n + 1, :],
        verified_logprobs=verified_logprobs,
    )


@dataclass
class GenerationResponse:
    """Response from DFlash generation."""

    text: str
    tokens: List[int]
    accepted: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def stream_generate(
    model: nn.Module,
    draft: DFlashDraftModel,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    block_size: Optional[int] = None,
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array] = None,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate tokens using DFlash speculative decoding.

    Args:
        model: Target model
        draft: DFlash draft model
        tokenizer: Tokenizer
        prompt: Input prompt (string, token IDs, or array)
        block_size: Number of tokens to speculate per step (default: from draft config)
        max_tokens: Maximum tokens to generate
        sampler: Sampling function (default: greedy)

    Yields:
        GenerationResponse with generated text and stats
    """
    _patch_model(model, draft.config.target_layer_ids)
    block_size = block_size if block_size is not None else int(draft.config.block_size)
    sampler = sampler or make_sampler(temp=0.0)

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    tokens = prompt.tolist()

    target_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft)
    draft.bind(model)
    _target_can_trim = can_trim_prompt_cache(target_cache)
    _capture = _GDNStateCapture() if (not _target_can_trim and _HAS_GDN) else None

    # Prefill
    tic = time.perf_counter()
    with mx.stream(generation_stream):
        logits = model(prompt[None], target_cache)
        hidden = mx.concatenate(model._hidden_states, axis=-1)
    mx.eval(logits, hidden)
    prompt_tps = prompt.size / (time.perf_counter() - tic)

    tic = time.perf_counter()
    token = sampler(logits[:, -1:])[0, 0].item()
    tokens.append(token)
    n = 1

    # Check EOS
    if token in tokenizer.eos_token_ids:
        detokenizer.add_token(token)
        detokenizer.finalize()
        yield GenerationResponse(
            detokenizer.last_segment,
            [token],
            1,
            prompt.size,
            prompt_tps,
            n,
            n / (time.perf_counter() - tic),
            mx.get_peak_memory() / 1e9,
            "stop",
        )
        return

    detokenizer.add_token(token)
    yield GenerationResponse(
        detokenizer.last_segment,
        [token],
        1,
        prompt.size,
        prompt_tps,
        n,
        n / (time.perf_counter() - tic),
        mx.get_peak_memory() / 1e9,
        None,
    )

    # Speculative decoding loop
    while n < max_tokens:
        bs = min(block_size, max_tokens - n + 1)
        if bs <= 1:
            break

        # Draft step
        with mx.stream(generation_stream):
            block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
            draft_logits = draft(block, hidden, draft_cache)
            if (trim_n := draft_cache[0].offset - (prompt.size + n - 1)) > 0:
                trim_prompt_cache(draft_cache, trim_n)
            draft_tokens = sampler(draft_logits[:, 1 - bs :])
        mx.async_eval(draft_tokens)

        # Target verify step (with intermediate state capture for recurrent models)
        if _capture is not None:
            _capture.enter()
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
            logits = model(verify_input, target_cache)
            hidden = mx.concatenate(model._hidden_states, axis=-1)
            target_tokens = sampler(logits)
        if _capture is not None:
            _capture.exit()
        mx.async_eval(target_tokens, hidden)

        d_list, t_list = draft_tokens[0].tolist(), target_tokens[0].tolist()
        accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list))
        new_tokens = d_list[:accepted] + [t_list[accepted]]
        new_tokens = new_tokens[: max_tokens - n]

        # Check for EOS
        eos_idx = next((i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids), None)
        if eos_idx is not None:
            new_tokens = new_tokens[: eos_idx + 1]
            for t in new_tokens:
                detokenizer.add_token(t)
            detokenizer.finalize()
            tokens.extend(new_tokens)
            n += len(new_tokens)
            yield GenerationResponse(
                detokenizer.last_segment,
                new_tokens,
                accepted + 1,
                prompt.size,
                prompt_tps,
                n,
                n / (time.perf_counter() - tic),
                mx.get_peak_memory() / 1e9,
                "stop",
            )
            return

        for t in new_tokens:
            detokenizer.add_token(t)
        tokens.extend(new_tokens)
        n += len(new_tokens)

        if n % 256 == 0:
            mx.clear_cache()

        yield GenerationResponse(
            detokenizer.last_segment,
            new_tokens,
            accepted + 1,
            prompt.size,
            prompt_tps,
            n,
            n / (time.perf_counter() - tic),
            mx.get_peak_memory() / 1e9,
            None,
        )

        trim = bs - accepted - 1
        if trim > 0:
            if _target_can_trim:
                trim_prompt_cache(target_cache, trim)
            elif _capture is not None:
                _capture.rollback(target_cache, accepted, trim)
        hidden = hidden[:, : accepted + 1, :]

    detokenizer.finalize()
    yield GenerationResponse(
        detokenizer.last_segment,
        [],
        0,
        prompt.size,
        prompt_tps,
        n,
        n / (time.perf_counter() - tic),
        mx.get_peak_memory() / 1e9,
        "length",
    )
