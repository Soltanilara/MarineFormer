import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange

from .network_utils import init

USV_NAVIGATION_ENV_ID = "USVNavigation-v0"


class GRUSequenceEncoder(nn.Module):
    def __init__(self, args, use_edge_state):
        super(GRUSequenceEncoder, self).__init__()
        self.args = args

        if use_edge_state:
            self.gru = nn.GRU(
                args.graph_edge_embedding_size,
                args.graph_edge_hidden_size,
            )
        else:
            self.gru = nn.GRU(
                args.graph_embedding_size * 3,
                args.graph_hidden_size,
            )

        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            seq_len, nenv, agent_num, _ = x.size()
            x = x.view(seq_len, nenv * agent_num, -1)
            mask_agent_num = masks.size()[-1]
            hxs_times_masks = hxs * masks.view(seq_len, nenv, mask_agent_num, 1)
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv * agent_num, -1)
            x, hxs = self.gru(x, hxs_times_masks)
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)
        else:
            T, N, agent_num, _ = x.size()
            masks = masks.view(T, N)
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]

            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                x_in = x[start_idx:end_idx]
                x_in = x_in.view(
                    x_in.size(0), x_in.size(1) * x_in.size(2), x_in.size(3)
                )
                hxs = hxs.view(hxs.size(0), N, agent_num, -1)
                hxs = hxs * masks[start_idx].view(1, -1, 1, 1)
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                sequence_outputs, hxs = self.gru(x_in, hxs)
                outputs.append(sequence_outputs)

            x = torch.cat(outputs, dim=0)
            x = x.view(T, N, agent_num, -1)
            hxs = hxs.view(1, N, agent_num, -1)

        return x, hxs


class DynamicObstacleAlignment(nn.Module):
    """Computes A_t, the pairwise alignment matrix for dynamic obstacles."""

    def __init__(self, args, input_size=12):
        super(DynamicObstacleAlignment, self).__init__()
        self.args = args
        if args.env_name == USV_NAVIGATION_ENV_ID:
            self.input_size = input_size
        else:
            raise NotImplementedError(f"Unsupported environment: {args.env_name}")
        self.embed_dim = 64
        self.linear = nn.Linear(self.input_size, self.embed_dim)
        self.relu = nn.ReLU()
        self.diagonal = nn.Parameter(torch.ones(1, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.diagonal)

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_dynamic_obstacle_num):

        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_dynamic_obstacle_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_dynamic_obstacle_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.0
        mask = torch.logical_not(mask.cumsum(dim=1))

        mask = mask[:, :-1].unsqueeze(-2)
        return mask

    def forward(self, dynamic_obstacle_states, detected_obstacle_num):
        seq_len, nenv, max_dynamic_obstacle_num, _ = dynamic_obstacle_states.size()
        if self.args.sort_dynamic_obstacles:
            attn_mask = self.create_attn_mask(
                detected_obstacle_num, seq_len, nenv, max_dynamic_obstacle_num
            )
            attn_mask = attn_mask.squeeze(1)
        else:
            attn_mask = detected_obstacle_num.reshape(
                seq_len * nenv, max_dynamic_obstacle_num
            )

        obstacle_embeddings = self.relu(self.linear(dynamic_obstacle_states)).view(
            seq_len * nenv, max_dynamic_obstacle_num, -1
        )

        obstacle_embeddings = attn_mask.unsqueeze(-1) * obstacle_embeddings
        obstacle_embeddings = obstacle_embeddings * self.diagonal

        alignment_matrix = obstacle_embeddings.bmm(obstacle_embeddings.transpose(1, 2))
        return torch.softmax(alignment_matrix, dim=-1)


class CurrentFlowAttention(nn.Module):
    def __init__(self, args):
        super(CurrentFlowAttention, self).__init__()
        self.args = args
        self.num_attn_heads = 8
        self.flow_feature_size = int(
            self.args.flow_grid_num * self.args.flow_grid_num * 8 / 2
        )
        self.temporal_size = 256

        self.embedding_layer = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1, groups=2))
        self.q_linear = nn.Linear(self.temporal_size, self.temporal_size)
        self.v_linear = nn.Linear(self.flow_feature_size, self.temporal_size)
        self.k_linear = nn.Linear(self.flow_feature_size, self.temporal_size)

        self.multihead_attn = torch.nn.MultiheadAttention(
            self.temporal_size, self.num_attn_heads
        )

    def forward(self, ego_features, current_flow):
        seq_len, nenv, flow_channel_num, input_size = current_flow.size()
        ego_features = torch.repeat_interleave(
            ego_features.reshape(seq_len * nenv, 1, -1), 2, 1
        )
        input_emb = self.embedding_layer(
            current_flow.view(
                seq_len * nenv,
                flow_channel_num,
                int(np.sqrt(input_size)),
                int(np.sqrt(input_size)),
            )
        )
        input_emb = torch.stack(
            [
                input_emb[:, :4].reshape(seq_len * nenv, -1),
                input_emb[:, 4:].reshape(seq_len * nenv, -1),
            ],
            dim=1,
        )
        input_emb = torch.transpose(input_emb, dim0=0, dim1=1)
        ego_features = torch.transpose(ego_features, dim0=0, dim1=1)
        q = self.q_linear(ego_features)
        k = self.k_linear(input_emb)
        v = self.v_linear(input_emb)

        z, _ = self.multihead_attn(q, k, v)
        z = torch.transpose(z, dim0=0, dim1=1)
        return z


class StaticObstacleAttention(nn.Module):
    def __init__(self, args):
        super(StaticObstacleAttention, self).__init__()

        self.args = args

        self.static_obstacle_feature_size = 64
        self.attention_size = args.attention_size

        self.temporal_edge_layer = nn.ModuleList()
        self.spatial_edge_layer = nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(512, self.attention_size))

        self.spatial_edge_layer.append(
            nn.Linear(self.static_obstacle_feature_size, self.attention_size)
        )

        self.agent_num = 1
        self.num_attention_head = 1

    def att_func(self, temporal_embed, spatial_embed, context_features):
        seq_len, nenv, num_edges, h_size = context_features.size()
        attn = temporal_embed * spatial_embed

        attn = torch.sum(attn, dim=3)

        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        attn = attn.view(seq_len, nenv, self.agent_num, num_edges)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        context_features = context_features.view(
            seq_len, nenv, self.agent_num, num_edges, h_size
        )
        context_features = context_features.view(
            seq_len * nenv * self.agent_num, num_edges, h_size
        ).permute(0, 2, 1)
        attn = attn.view(seq_len * nenv * self.agent_num, num_edges).unsqueeze(-1)
        weighted_value = torch.bmm(context_features, attn)

        weighted_value = weighted_value.squeeze(-1).view(
            seq_len, nenv, self.agent_num, h_size
        )
        return weighted_value, attn

    def forward(self, query_features, context_features):
        seq_len, nenv, static_obstacle_num, _ = context_features.size()

        self.static_obstacle_num = static_obstacle_num // self.agent_num

        weighted_value_list, attn_list = [], []
        for i in range(self.num_attention_head):

            temporal_embed = self.temporal_edge_layer[i](query_features)

            spatial_embed = self.spatial_edge_layer[i](context_features)

            temporal_embed = temporal_embed.repeat_interleave(
                static_obstacle_num, dim=2
            )

            weighted_value, attn = self.att_func(
                temporal_embed, spatial_embed, context_features
            )
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return (
                self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)),
                attn_list,
            )
        else:
            return weighted_value_list[0], attn_list[0]


class DynamicObstacleAttention(nn.Module):
    def __init__(self, args):
        super(DynamicObstacleAttention, self).__init__()

        self.args = args

        self.graph_edge_hidden_size = args.graph_edge_hidden_size
        self.graph_hidden_size = args.graph_hidden_size
        self.attention_size = args.attention_size

        self.temporal_edge_layer = nn.ModuleList()
        self.spatial_edge_layer = nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(512, self.attention_size))

        self.spatial_edge_layer.append(
            nn.Linear(
                self.graph_edge_hidden_size,
                self.attention_size,
            )
        )

        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_dynamic_obstacle_num):

        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_dynamic_obstacle_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_dynamic_obstacle_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.0
        mask = torch.logical_not(mask.cumsum(dim=1))

        mask = mask[:, :-1].unsqueeze(-2)
        return mask

    def att_func(self, temporal_embed, spatial_embed, context_features, attn_mask=None):
        seq_len, nenv, num_edges, h_size = context_features.size()
        attn = temporal_embed * spatial_embed

        attn = torch.sum(attn, dim=3)

        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn = attn.view(seq_len, nenv, self.agent_num, self.dynamic_obstacle_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        context_features = context_features.view(
            seq_len, nenv, self.agent_num, self.dynamic_obstacle_num, h_size
        )
        context_features = context_features.view(
            seq_len * nenv * self.agent_num, self.dynamic_obstacle_num, h_size
        ).permute(0, 2, 1)
        attn = attn.view(
            seq_len * nenv * self.agent_num, self.dynamic_obstacle_num
        ).unsqueeze(-1)
        weighted_value = torch.bmm(context_features, attn)

        weighted_value = weighted_value.squeeze(-1).view(
            seq_len, nenv, self.agent_num, h_size
        )
        return weighted_value, attn

    def forward(self, query_features, context_features, each_seq_len):
        seq_len, nenv, max_dynamic_obstacle_num, _ = context_features.size()

        self.dynamic_obstacle_num = max_dynamic_obstacle_num // self.agent_num

        weighted_value_list, attn_list = [], []
        for i in range(self.num_attention_head):

            temporal_embed = self.temporal_edge_layer[i](query_features)

            spatial_embed = self.spatial_edge_layer[i](context_features)

            temporal_embed = temporal_embed.repeat_interleave(
                self.dynamic_obstacle_num, dim=2
            )

            if self.args.sort_dynamic_obstacles:
                attn_mask = self.create_attn_mask(
                    each_seq_len, seq_len, nenv, max_dynamic_obstacle_num
                )
                attn_mask = attn_mask.squeeze(-2).view(
                    seq_len, nenv, max_dynamic_obstacle_num
                )
            else:
                attn_mask = each_seq_len
            weighted_value, attn = self.att_func(
                temporal_embed, spatial_embed, context_features, attn_mask=attn_mask
            )
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return (
                self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)),
                attn_list,
            )
        else:
            return weighted_value_list[0], attn_list[0]


class RecurrentPolicyHead(GRUSequenceEncoder):
    def __init__(self, args):
        super(RecurrentPolicyHead, self).__init__(args, use_edge_state=False)

        self.args = args

        self.temporal_hidden_size = args.graph_hidden_size
        self.output_size = args.policy_feature_size
        self.embedding_size = args.graph_embedding_size
        self.input_size = args.dynamic_obstacle_state_size
        self.graph_edge_hidden_size = args.graph_edge_hidden_size

        self.encoder_linear = nn.Linear(512, self.embedding_size)

        self.relu = nn.ReLU()

        self.dynamic_obstacle_context_embed = nn.Linear(
            self.graph_edge_hidden_size, self.embedding_size
        )

        self.static_obstacle_context_embed = nn.Linear(64, self.embedding_size)

        self.output_linear = nn.Linear(
            3 * self.embedding_size, self.output_size + self.embedding_size
        )

    def forward(
        self,
        ego_features,
        dynamic_obstacle_context,
        static_obstacle_context,
        h,
        masks,
    ):
        encoded_input = self.encoder_linear(ego_features)
        encoded_input = self.relu(encoded_input)

        dynamic_obstacle_context_features = self.relu(
            self.dynamic_obstacle_context_embed(dynamic_obstacle_context)
        )

        static_obstacle_context_features = self.relu(
            self.static_obstacle_context_embed(static_obstacle_context)
        )

        concat_encoded = torch.cat(
            (
                encoded_input,
                dynamic_obstacle_context_features,
                static_obstacle_context_features,
            ),
            -1,
        )

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)

        return outputs, h_new


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        V = self.values(values)
        K = self.keys(keys)
        Q = self.queries(query)

        V = V.reshape(N, value_len, self.num_heads, self.head_size)
        K = K.reshape(N, key_len, self.num_heads, self.head_size)
        Q = Q.reshape(N, query_len, self.num_heads, self.head_size)

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")
            )

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            N, query_len, self.num_heads * self.head_size
        )

        out = self.fc_out(out)

        return out, attention


class TransformerSequenceEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerSequenceEncoder, self).__init__()
        self.args = args

        self.attention = MultiHeadAttention(3 * args.graph_embedding_size, 1)
        self.norm1 = nn.LayerNorm(3 * args.graph_embedding_size)

    def _forward_transformer(self, h_0_t, h_previous, masks):

        seq_len, nenv, agent_num, _ = h_0_t.size()
        h_0_t = h_0_t.view(seq_len, nenv * agent_num, -1)
        h_previous = h_previous.view(seq_len, nenv * agent_num, -1)
        mask_agent_num = masks.size()[-1]

        masked_history = h_previous * (masks.view(seq_len, nenv, mask_agent_num))
        masked_history = masked_history.view(seq_len, nenv * agent_num, -1)

        transformer_query = self.norm1(h_0_t)
        attention_output, _ = self.attention(
            masked_history, masked_history, transformer_query
        )

        h_L_t = attention_output + h_0_t
        encoded_graph = self.norm1(h_L_t)
        encoded_graph = encoded_graph.view(seq_len, nenv, agent_num, -1)

        h_previous = h_previous.view(seq_len, nenv, agent_num, -1)

        return encoded_graph, h_L_t, h_previous


class TransformerPolicyHead(TransformerSequenceEncoder):
    def __init__(self, args):
        super(TransformerPolicyHead, self).__init__(args)

        self.args = args

        self.temporal_hidden_size = args.graph_hidden_size
        self.output_size = args.policy_feature_size
        self.embedding_size = args.graph_embedding_size
        self.input_size = args.dynamic_obstacle_state_size
        self.graph_edge_hidden_size = args.graph_edge_hidden_size

        self.encoder_linear = nn.Linear(512, self.embedding_size)

        self.relu = nn.ReLU()

        self.dynamic_obstacle_context_embed = nn.Linear(
            self.graph_edge_hidden_size, self.embedding_size
        )

        self.static_obstacle_context_embed = nn.Linear(64, self.embedding_size)

        self.output_linear = nn.Linear(
            3 * self.embedding_size, self.output_size + self.embedding_size
        )
        self.norm1 = nn.LayerNorm(3 * args.graph_embedding_size)

    def forward(
        self,
        ego_features,
        dynamic_obstacle_context,
        static_obstacle_context,
        h,
        masks,
    ):
        encoded_input = self.encoder_linear(ego_features)
        encoded_input = self.relu(encoded_input)

        dynamic_obstacle_context_features = self.relu(
            self.dynamic_obstacle_context_embed(dynamic_obstacle_context)
        )
        static_obstacle_context_features = self.relu(
            self.static_obstacle_context_embed(static_obstacle_context)
        )
        h_0_t = torch.cat(
            (
                encoded_input,
                dynamic_obstacle_context_features,
                static_obstacle_context_features,
            ),
            -1,
        )

        h_L_t, temporal_residual, hxs = self._forward_transformer(h_0_t, h, masks)

        outputs = self.output_linear(h_L_t)
        temporal_residual = temporal_residual.unsqueeze(-2)
        outputs = self.norm1(outputs) + temporal_residual

        return outputs, hxs


class MarineFormerNetwork(nn.Module):
    """MarineFormer PPO policy network."""

    def __init__(self, obs_space_dict, args, infer=False):
        super(MarineFormerNetwork, self).__init__()
        self.infer = infer
        self.args = args
        self.is_recurrent = self.args.recurrent_policy

        self.dynamic_obstacle_num = obs_space_dict["dynamic_obstacle_states"].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        self.graph_hidden_size = args.graph_hidden_size
        self.graph_edge_hidden_size = args.graph_edge_hidden_size
        self.output_size = args.policy_feature_size
        self.embedding_size = args.graph_embedding_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.ego_state_encoder = nn.Sequential(init_(nn.Linear(9, 256)), nn.ReLU())

        self.static_obstacle_encoder = nn.Sequential(
            init_(nn.Linear(3, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, 64)),
            nn.ReLU(),
        )

        self.current_flow_attention = CurrentFlowAttention(args)

        self.static_obstacle_attention = StaticObstacleAttention(args)

        if self.args.use_self_attn:

            self.dynamic_obstacle_alignment = DynamicObstacleAlignment(
                args, obs_space_dict["dynamic_obstacle_states"].shape[1]
            )

            self.dynamic_obstacle_encoder = nn.Sequential(
                init_(
                    nn.Linear(
                        obs_space_dict["dynamic_obstacle_states"].shape[0]
                        + obs_space_dict["dynamic_obstacle_states"].shape[1],
                        128,
                    )
                ),
                nn.ReLU(),
                init_(nn.Linear(128, 512)),
                nn.ReLU(),
                init_(nn.Linear(512, 256)),
                nn.ReLU(),
            )
        else:

            self.dynamic_obstacle_encoder = nn.Sequential(
                init_(
                    nn.Linear(obs_space_dict["dynamic_obstacle_states"].shape[1], 128)
                ),
                nn.ReLU(),
                init_(nn.Linear(128, 512)),
                nn.ReLU(),
                init_(nn.Linear(512, 256)),
                nn.ReLU(),
            )

        self.dynamic_obstacle_attention = DynamicObstacleAttention(args)

        if self.is_recurrent:
            self.temporal_encoder = RecurrentPolicyHead(args)
        else:
            self.temporal_encoder = TransformerPolicyHead(args)

        num_inputs = hidden_size = self.output_size
        self.critic = nn.Sequential(
            init_(nn.Linear(self.embedding_size + num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(self.embedding_size + num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.ego_velocity = [0]
        self.dynamic_obstacle_states = np.arange(1, self.dynamic_obstacle_num + 1)

        dummy_dynamic_obstacle_mask = [0] * self.dynamic_obstacle_num
        dummy_dynamic_obstacle_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_dynamic_obstacle_mask = Variable(
                torch.Tensor([dummy_dynamic_obstacle_mask]).cpu()
            )
        else:
            self.dummy_dynamic_obstacle_mask = Variable(
                torch.Tensor([dummy_dynamic_obstacle_mask]).cuda()
            )

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:

            seq_length = 1
            nenv = self.nenv

        else:

            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch
        ego_state = reshapeT(inputs["ego_state"], seq_length, nenv)
        ego_velocity = reshapeT(inputs["ego_velocity"], seq_length, nenv)
        dynamic_obstacle_states = reshapeT(
            inputs["dynamic_obstacle_states"], seq_length, nenv
        )
        static_obstacle_states = reshapeT(
            inputs["static_obstacle_states"], seq_length, nenv
        )

        if not hasattr(self.args, "sort_dynamic_obstacles"):
            self.args.sort_dynamic_obstacles = True
        if self.args.sort_dynamic_obstacles:
            detected_dynamic_obstacle_num = (
                inputs["detected_dynamic_obstacle_num"].squeeze(-1).cpu().int()
            )
        else:
            dynamic_obstacle_masks = reshapeT(
                inputs["visible_masks"], seq_length, nenv
            ).float()

            dynamic_obstacle_masks[dynamic_obstacle_masks.sum(dim=-1) == 0] = (
                self.dummy_dynamic_obstacle_mask
            )

        if self.is_recurrent:
            temporal_state = reshapeT(rnn_hxs["graph_temporal_state"], 1, nenv)
        else:
            temporal_state = reshapeT(rnn_hxs["graph_temporal_state"], seq_length, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        if self.args.no_cuda:
            next_edge_state = Variable(
                torch.zeros(
                    1,
                    nenv,
                    1 + self.dynamic_obstacle_num,
                    rnn_hxs["graph_edge_state"].size()[-1],
                ).cpu()
            )
        else:
            next_edge_state = Variable(
                torch.zeros(
                    1,
                    nenv,
                    1 + self.dynamic_obstacle_num,
                    rnn_hxs["graph_edge_state"].size()[-1],
                ).cuda()
            )

        current_flow = reshapeT(inputs["current_flow"], seq_length, nenv)

        ego_features = self.ego_state_encoder(
            torch.cat((ego_velocity, ego_state), dim=-1)
        )
        encoded_static_obstacles = self.static_obstacle_encoder(static_obstacle_states)

        ego_features = self.current_flow_attention(ego_features, current_flow)
        ego_features = ego_features.reshape(seq_length, nenv, 1, -1)

        static_obstacle_context, _ = self.static_obstacle_attention(
            ego_features, encoded_static_obstacles
        )

        if self.args.sort_dynamic_obstacles:

            if self.args.use_self_attn:

                alignment_matrix = self.dynamic_obstacle_alignment(
                    dynamic_obstacle_states,
                    detected_dynamic_obstacle_num,
                ).view(seq_length, nenv, self.dynamic_obstacle_num, -1)

                dynamic_obstacle_node_input = torch.cat(
                    (dynamic_obstacle_states, alignment_matrix), dim=-1
                )
            else:
                dynamic_obstacle_node_input = dynamic_obstacle_states

            encoded_dynamic_obstacles = self.dynamic_obstacle_encoder(
                dynamic_obstacle_node_input
            )

            dynamic_obstacle_context, _ = self.dynamic_obstacle_attention(
                ego_features,
                encoded_dynamic_obstacles,
                detected_dynamic_obstacle_num,
            )
        else:

            if self.args.use_self_attn:
                dynamic_obstacle_node_input = self.spatial_attn(
                    dynamic_obstacle_states, dynamic_obstacle_masks
                ).view(seq_length, nenv, self.dynamic_obstacle_num, -1)
            else:
                dynamic_obstacle_node_input = dynamic_obstacle_states
            encoded_dynamic_obstacles = self.dynamic_obstacle_encoder(
                dynamic_obstacle_node_input
            )

            dynamic_obstacle_context, _ = self.dynamic_obstacle_attention(
                ego_features, encoded_dynamic_obstacles, dynamic_obstacle_masks
            )

        outputs, h_nodes = self.temporal_encoder(
            ego_features,
            dynamic_obstacle_context,
            static_obstacle_context,
            temporal_state,
            masks,
        )

        next_temporal_state = h_nodes
        outputs_return = outputs

        if self.is_recurrent:
            rnn_hxs["graph_temporal_state"] = next_temporal_state
        else:
            rnn_hxs["graph_temporal_state"] = outputs_return
        rnn_hxs["graph_edge_state"] = next_edge_state

        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return (
                self.critic_linear(hidden_critic).squeeze(0),
                hidden_actor.squeeze(0),
                rnn_hxs,
            )
        else:
            return (
                self.critic_linear(hidden_critic).view(-1, 1),
                hidden_actor.view(-1, self.output_size),
                rnn_hxs,
            )


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))
