from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu
import time
from .latentnet import *
from .unet import *
from choices import *
from einops import rearrange, reduce, repeat, einsum



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            no_of_fonts = conf.no_of_fonts,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()
        
        self.style_pred = MLP(512, 1024, conf.no_of_fonts)


        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x, font):
        cond = self.encoder.forward(x, font)
        return cond
    
    def get_style(self, cond):
        style = self.style_pred(cond)
        return style

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S
        
    def local_encoder_pullback_xt(
            self, x=None, t=None, cond=None, op=None, block_idx=None, pca_rank=16, chunk_size=1,
            min_iter=1, max_iter=100, convergence_threshold=1e-3,
        ):
        '''
        Args
            - sample : zt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        # necessary variables
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1
        get_h = lambda x : self.get_h(x, t=t, cond=cond) # , op=op, block_idx=block_idx)
        h_shape = get_h(x).shape

        c_i, w_i, h_i = x.size(1), x.size(2), x.size(3)
        c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

        a = th.tensor(0., device=x.device)

        # Algorithm 1
        vT = th.randn(c_i*w_i*h_i, pca_rank, device=x.device)
        vT, _ = th.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)
        for i in range(max_iter):
            v_prev = v.detach().cpu().clone()
            u = []
            time_s = time.time()

            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                # g = lambda a : get_h(x + a*vi.unsqueeze(0) if vi.size(0) == v.size(-1) else x + a*vi)
                g = lambda a : get_h(x + a*vi)
                ui = th.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
                u.append(ui.detach().cpu().clone())
            time_e = time.time()
            print('single v jacfwd t ==', time_e - time_s)
            u = th.cat(u, dim=0)
            u = u.to(x.device)
            # time_s = time.time()
            # g = lambda a : get_h(x + a*v)
            # u = th.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            # time_e = time.time()
            # print('single vi jacfwd t ==', time_e - time_s)

            g = lambda x : einsum(u, get_h(x), 'b c w h, i c w h -> b')
            v_ = th.autograd.functional.jacobian(g, x)
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = th.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            u = u.view(-1, c_o, w_o, h_o)
            convergence = th.dist(v_prev, v.detach().cpu())
            print(f'power method : {i}-th step convergence : ', convergence)
            if th.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

            if i == max_iter - 1:
                print('last convergence : ', convergence)

        u, s, vT = u.view(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.view(-1, c_i*w_i*h_i).detach()
        return u, s, vT
             
    def get_h(self, x, t, cond=None, y=None, **kwargs):
            # if isinstance(t, int):
            #     t = th.tensor([t]).to(self.device)
            # elif isinstance(t, th.Tensor):
            #     t = t.unsqueeze(0) if len(t.shape) == 0 else t
            # else:
            #     raise ValueError('t must be int or torch.Tensor')

        t_cond = t
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None
            
            # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb # it's just condition
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            
        return h


        
    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                font = None,
                label=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            cond = self.encode(x_start, font)
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb # it's just condition
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        style_pred = self.style_pred(cond)
        return AutoencReturn(pred=pred, cond=cond, style_pred=style_pred)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None
    style_pred: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
