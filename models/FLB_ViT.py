import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import PatchEmbed

class FLBViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 3,
        num_heads: int = 6,
        dropout: float = 0.0
    ):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        # Spatial Patch Tokenizer (no synthetic CLS token)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_tokens = self.patch_embed.num_tokens

        # Recurrent FLB Encoder Blocks
        self.blocks = nn.ModuleList([
            FLBBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Multiscale Decoder: generates top-down feedback every tick and classifies once settled
        self.decoder = MultiscaleDecoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        images: torch.Tensor,
        max_steps: int = 6,
        threshold: float = 1e-2,
        labels: torch.Tensor = None,
        criterion: nn.Module = None
    ):
        batch_size = images.size(0)
        device = images.device

        # Sample a single consistent dropout mask per layer to reuse across all ticks
        self.reset_dropout_masks()

        input_tokens = self.patch_embed(images)
        lateral_states, backward_states, top_backward = self.init_hidden_states(
            batch_size, device
        )
        prev_forward = None

        total_aux_loss = torch.tensor(0.0, device=device)
        steps_run = 0
        min_steps = self.depth

        # Track convergence step per individual image in the batch
        converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        image_steps = torch.zeros(batch_size, dtype=torch.long, device=device)

        run_step_backward = self.training and (labels is not None) and (criterion is not None)

        for step in range(max_steps):
            steps_run = step + 1
            image_steps += (~converged_mask).long()

            # Detach recurrent states between ticks for O(1) temporal memory
            if step > 0:
                lateral_in = [state.detach() for state in lateral_states]
                backward_in = [state.detach() for state in backward_states]
                top_backward_in = top_backward.detach()
            else:
                lateral_in = lateral_states
                backward_in = backward_states
                top_backward_in = top_backward

            # Recompute patch_embed if calling backward per tick so its graph is fresh each step
            if run_step_backward and step > 0:
                tokens_in = self.patch_embed(images)
            else:
                tokens_in = input_tokens

            # Step 0 is the initial feedforward sweep (zeros in h and g); turn on JEPA aux_loss for step >= 1
            compute_aux = run_step_backward and (step > 0)

            final_forward, next_lateral, next_backward, next_top_backward, aux_loss = self.forward_tick(
                tokens_in, lateral_in, backward_in, top_backward_in, with_aux=compute_aux
            )

            # Check per-image full-graph convergence once signals complete a minimum round-trip
            batch_converged = False
            if threshold > 0.0 and step >= (min_steps - 1):
                deltas = []
                for i in range(self.depth):
                    deltas.append(self.compute_relative_delta(next_lateral[i], lateral_states[i]))
                    deltas.append(self.compute_relative_delta(next_backward[i], backward_states[i]))
                deltas.append(self.compute_relative_delta(final_forward, prev_forward))
                deltas.append(self.compute_relative_delta(next_top_backward, top_backward))

                # Worst-case stream delta for each individual image: shape (batch_size,)
                image_max_delta = torch.stack(deltas, dim=0).max(dim=0).values
                converged_mask = converged_mask | (image_max_delta < threshold)

                if converged_mask.all():
                    batch_converged = True

            is_final_step = batch_converged or (step == max_steps - 1)
            lateral_states = next_lateral
            backward_states = next_backward
            top_backward = next_top_backward
            prev_forward = final_forward.detach()

            if run_step_backward:
                if compute_aux:
                    total_aux_loss = total_aux_loss + aux_loss.detach()

                if not is_final_step:
                    if compute_aux:
                        aux_loss.backward()
                else:
                    logits = self.decoder.classify(top_backward)
                    ce_loss = criterion(logits, labels)
                    final_loss = ce_loss + (aux_loss if compute_aux else 0.0)
                    final_loss.backward()

                    aux_steps = max(1, steps_run - 1)
                    mean_aux_loss = (total_aux_loss / aux_steps).item()
                    avg_image_steps = image_steps.float().mean().item()
                    return logits.detach(), ce_loss.item(), mean_aux_loss, avg_image_steps
            else:
                if is_final_step:
                    break

        logits = self.decoder.classify(top_backward)
        avg_image_steps = image_steps.float().mean().item()
        return logits, avg_image_steps

    def forward_tick(
        self,
        input_tokens: torch.Tensor,
        prev_lateral_states: list[torch.Tensor],
        prev_backward_states: list[torch.Tensor],
        prev_top_backward: torch.Tensor,
        with_aux: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        next_lateral_states = []
        next_backward_states = []
        tick_aux_loss = torch.tensor(0.0, device=input_tokens.device)

        forward_stream = input_tokens
        for layer_idx in range(self.depth):
            lateral_prev = prev_lateral_states[layer_idx]

            if layer_idx + 1 < self.depth:
                backward_above = prev_backward_states[layer_idx + 1]
            else:
                backward_above = prev_top_backward

            forward_new, lateral_new, backward_new, block_aux = self.blocks[layer_idx](
                forward_stream, lateral_prev, backward_above, with_aux=with_aux
            )

            if with_aux and block_aux is not None:
                tick_aux_loss = tick_aux_loss + (block_aux / self.depth)

            forward_stream = forward_new
            next_lateral_states.append(lateral_new)
            next_backward_states.append(backward_new)

        # Compute multiscale decoder state to serve as top_backward on the next tick (and final readout)
        next_top_backward = self.decoder.forward_state(next_lateral_states, forward_stream)

        return forward_stream, next_lateral_states, next_backward_states, next_top_backward, tick_aux_loss

    def init_hidden_states(self, batch_size: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        zero_shape = (batch_size, self.num_tokens, self.embed_dim)
        lateral_states = [
            torch.zeros(zero_shape, device=device) for _ in range(self.depth)
        ]
        backward_states = [
            torch.zeros(zero_shape, device=device) for _ in range(self.depth)
        ]
        top_backward = torch.zeros(zero_shape, device=device)
        return lateral_states, backward_states, top_backward

    def reset_dropout_masks(self):
        for module in self.modules():
            if isinstance(module, LockedDropout):
                module.reset_mask()

    def compute_relative_delta(self, current_state: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        diff_norm = torch.norm(current_state - prev_state, p=2, dim=(1, 2))
        prev_norm = torch.norm(prev_state, p=2, dim=(1, 2)) + 1e-6
        return diff_norm / prev_norm

class TriadicAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Bottom-up Forward projections (W_q, W_k, W_v)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Lateral (U_q) and Backward (R_q) query projections (bias=False so 0 input -> 0 output)
        self.U_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.R_q = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = LockedDropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = LockedDropout(dropout)

    def forward(
        self,
        forward_in: torch.Tensor,
        lateral_in: torch.Tensor,
        backward_in: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = forward_in.shape

        # Project bottom-up Q, K, V and lateral/backward query gates
        q_f = self.W_q(forward_in)
        k_f = self.W_k(forward_in)
        v_f = self.W_v(forward_in)

        q_l = self.U_q(lateral_in)
        q_b = self.R_q(backward_in)

        # Multiplicative query gating in [0, 2]; equals 1.0 when lateral_in or backward_in is 0
        q_gated = q_f * (1.0 + torch.tanh(q_l * q_b))

        # Reshape (batch_size, num_tokens, embed_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        q = q_gated.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_f.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_f.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention across spatial patches
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Aggregate values and merge heads back to (batch_size, num_tokens, embed_dim)
        context = (attn_probs @ v).transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)
        return self.proj_drop(self.out_proj(context))

class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.psi_x = nn.Linear(embed_dim * 2, embed_dim)
        self.psi_h = nn.Linear(embed_dim * 3, embed_dim)
        self.psi_g = nn.Linear(embed_dim * 2, embed_dim)

    def cosine_distance(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(target, pred, dim=-1).mean()

    def forward(
        self,
        forward_in: torch.Tensor,
        lateral_prev: torch.Tensor,
        backward_above: torch.Tensor,
        forward_new: torch.Tensor,
        lateral_new: torch.Tensor,
        backward_new: torch.Tensor
    ) -> torch.Tensor:
        # Detach incoming targets so only the current layer's outgoing streams receive gradients
        x_target = forward_in.detach()
        h_target = lateral_prev.detach()
        g_target = backward_above.detach()

        # Reconstruct each detached incoming stream from its corresponding active subspace
        pred_x = self.psi_x(torch.cat([forward_new, g_target], dim=-1))
        pred_h = self.psi_h(torch.cat([lateral_new, x_target, g_target], dim=-1))
        pred_g = self.psi_g(torch.cat([backward_new, x_target], dim=-1))

        loss_f = self.cosine_distance(x_target, pred_x)
        loss_l = self.cosine_distance(h_target, pred_h)
        loss_b = self.cosine_distance(g_target, pred_g)

        return (loss_f + loss_l + loss_b) / 3.0

class FLBBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        # Pre-LayerNorms for the three incoming streams
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_h = nn.LayerNorm(embed_dim)
        self.norm_g = nn.LayerNorm(embed_dim)

        # Triadic Attention
        self.attn = TriadicAttention(embed_dim, num_heads, dropout=dropout)

        # Parallel 3-Branch MLP: expands D -> 3D, splits into 3 tracks of D, and projects each D -> D
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp_in = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.mlp_out_f = nn.Linear(embed_dim, embed_dim)
        self.mlp_out_h = nn.Linear(embed_dim, embed_dim)
        self.mlp_out_g = nn.Linear(embed_dim, embed_dim)
        self.mlp_drop_f = LockedDropout(dropout)
        self.mlp_drop_h = LockedDropout(dropout)
        self.mlp_drop_g = LockedDropout(dropout)

        # Separate LayerNorms for each decoupled candidate subspace
        self.norm_cand_f = nn.LayerNorm(embed_dim)
        self.norm_cand_h = nn.LayerNorm(embed_dim)
        self.norm_cand_g = nn.LayerNorm(embed_dim)

        # Parallel Selective Gates for Forward (f), Lateral (z), and Backward (s)
        self.gate_f = nn.Linear(embed_dim * 2, embed_dim)
        self.gate_z = nn.Linear(embed_dim * 2, embed_dim)
        self.gate_s = nn.Linear(embed_dim * 2, embed_dim)

        self.jepa = JEPAPredictor(embed_dim)

    def forward(
        self,
        forward_in: torch.Tensor,
        lateral_prev: torch.Tensor,
        backward_above: torch.Tensor,
        with_aux: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Triadic Attention with residual connection
        attn_out = forward_in + self.attn(
            self.norm_x(forward_in), self.norm_h(lateral_prev), self.norm_g(backward_above)
        )

        # Expand to 3D, split into 3 tracks of size D, and project each independently
        hidden_f, hidden_h, hidden_g = self.mlp_in(self.norm_mlp(attn_out)).chunk(3, dim=-1)

        raw_f = self.mlp_drop_f(self.mlp_out_f(hidden_f))
        raw_h = self.mlp_drop_h(self.mlp_out_h(hidden_h))
        raw_g = self.mlp_drop_g(self.mlp_out_g(hidden_g))

        cand_forward = self.norm_cand_f(attn_out + raw_f)
        cand_lateral = self.norm_cand_h(attn_out + raw_h)
        cand_backward = self.norm_cand_g(attn_out + raw_g)

        # Parallel Gate Updates to route information across the three streams
        f_gate = torch.sigmoid(self.gate_f(torch.cat([cand_forward, forward_in], dim=-1)))
        forward_new = f_gate * forward_in + (1.0 - f_gate) * cand_forward

        z_gate = torch.sigmoid(self.gate_z(torch.cat([cand_lateral, lateral_prev], dim=-1)))
        lateral_new = z_gate * lateral_prev + (1.0 - z_gate) * cand_lateral

        s_gate = torch.sigmoid(self.gate_s(torch.cat([cand_backward, backward_above], dim=-1)))
        backward_new = s_gate * backward_above + (1.0 - s_gate) * cand_backward

        aux_loss = None
        if with_aux:
            aux_loss = self.jepa(
                forward_in, lateral_prev, backward_above,
                forward_new, lateral_new, backward_new
            )

        return forward_new, lateral_new, backward_new, aux_loss

class MultiscaleDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        dropout: float = 0.0
    ):
        super().__init__()
        fused_dim = embed_dim * (depth + 1)

        self.multiscale_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            LockedDropout(dropout)
        )

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.attn_drop = LockedDropout(dropout)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Learned attention pooling to collapse N patch tokens into 1 vector for classification
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_state(
        self,
        lateral_states: list[torch.Tensor],
        final_forward: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate all layers' lateral states and the top forward state along the channel dim
        multiscale_tokens = torch.cat(lateral_states + [final_forward], dim=-1)
        fused_tokens = self.multiscale_proj(multiscale_tokens)

        # Mix spatial patches via self-attention to form the top-down expectation state
        normed_tokens = self.norm_attn(fused_tokens)
        attn_out, _ = self.self_attn(normed_tokens, normed_tokens, normed_tokens, need_weights=False)
        decoder_state = self.norm_out(fused_tokens + self.attn_drop(attn_out))
        return decoder_state

    def classify(self, decoder_state: torch.Tensor) -> torch.Tensor:
        # Collapse N patches into a single global vector via learned attention weights
        pool_weights = F.softmax(self.attn_pool(decoder_state), dim=1)
        pooled_vector = (pool_weights * decoder_state).sum(dim=1)
        return self.classifier(pooled_vector)

class LockedDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.mask = None

    def reset_mask(self):
        self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if self.mask is None:
            # Sample binary mask once on step 0 and scale by 1 / (1 - p)
            keep_prob = 1.0 - self.p
            self.mask = torch.bernoulli(torch.full_like(x, keep_prob)) / keep_prob
        return x * self.mask