"""Architecture adapters: where the residual stream lives in each model family.

Adapters find the text model by searching the module tree for the module that owns
``embed_tokens``, ``layers`` and ``norm``. That way nested multimodal wrappers such as
Gemma 3 need no hard-coded paths.
"""

from torch import nn

from vectorlab.errors import UserError


class Adapter:
    """Default adapter for models whose residual writers are ``o_proj`` and ``down_proj``.

    Attributes:
        base: The text model that owns the embeddings, decoder layers, and final norm.
        layers: The decoder layers, in order.
        embed: The token embedding module.
        final_norm: The norm applied before the unembedding.
        unembed: The projection from the residual stream to vocabulary logits.
        tied: Whether the unembedding shares its weight tensor with the embedding.
        norm_offset: Constant the architecture adds to RMSNorm weights (1.0 for Gemma's ``1 + w``).
        template_kwargs: Extra keyword arguments for ``tokenizer.apply_chat_template``.
        bakes_bias: Whether an ``o_proj`` bias reproduces the addition hook exactly, which needs
            the attention projection's output to join the residual stream unchanged.
    """

    norm_offset = 0.0
    template_kwargs: dict = {}
    bakes_bias = True

    def __init__(self, model: nn.Module):
        self.model = model
        self.base = next(m for m in model.modules() if all(hasattr(m, a) for a in ("embed_tokens", "layers", "norm")))
        self.layers = self.base.layers
        self.embed = self.base.embed_tokens
        self.final_norm = self.base.norm
        self.unembed = model.get_output_embeddings()
        self.tied = self.unembed.weight is self.embed.weight

    def writers(self, layer: nn.Module) -> list[nn.Module]:
        """Return the modules in ``layer`` whose outputs are added to the residual stream, attention first."""
        return [layer.self_attn.o_proj, layer.mlp.down_proj]

    def projections(self, layer: nn.Module) -> list[nn.Linear]:
        """Return the attention and MLP output projections, the linear maps behind each residual write."""
        return [layer.self_attn.o_proj, layer.mlp.down_proj]

    def post_norms(self, layer: nn.Module) -> list[nn.Module | None]:
        """Return the norm applied to each projection's output before it joins the residual stream, if any."""
        return [None, None]


class Qwen3(Adapter):
    """Qwen3. Thinking is disabled in the chat template so prompts end where the answer starts."""

    template_kwargs = {"enable_thinking": False}


class Llama(Adapter):
    """Llama 2/3. The date the Llama 3 template writes into its system header is pinned, so prompts never change."""

    template_kwargs = {"date_string": "26 Jul 2024"}


class Gemma3(Adapter):
    """Gemma 3. Attention and MLP outputs pass through post-norms before joining the residual stream."""

    norm_offset = 1.0
    bakes_bias = False

    def writers(self, layer: nn.Module) -> list[nn.Module]:
        """Return the post-attention and post-feedforward norms, whose outputs are the residual writes."""
        return [layer.post_attention_layernorm, layer.post_feedforward_layernorm]

    def post_norms(self, layer: nn.Module) -> list[nn.Module | None]:
        """Return the post-attention and post-feedforward norms."""
        return self.writers(layer)


ADAPTERS = {"qwen3": Qwen3, "llama": Llama, "gemma3": Gemma3, "gemma3_text": Gemma3}


def check_supported(model_type: str) -> None:
    """Raise UserError unless an adapter exists for ``model_type``."""
    if model_type not in ADAPTERS:
        raise UserError(f"unsupported architecture '{model_type}'; supported: {', '.join(sorted(ADAPTERS))}")


def adapter_for(model: nn.Module) -> Adapter:
    """Build the adapter that matches ``model``'s architecture."""
    check_supported(model.config.model_type)
    return ADAPTERS[model.config.model_type](model)
