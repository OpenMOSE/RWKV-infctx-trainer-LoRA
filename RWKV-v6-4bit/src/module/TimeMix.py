# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp
import os
from .LoraTrainer import *
from .rwkv_inner import rwkv_inner

# Current code file path
code_file_path = os.path.realpath(__file__)
code_dir = os.path.dirname(code_file_path)

# the cuda kernel (if its used)
global wkv6state_cuda_kernel
wkv6state_cuda_kernel = None

# WKV6STATE_CUDA autograd module
class WKV6STATE_CUDA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, 
                B:int, T:int, C:int, H:int, 
                r:torch.Tensor, k:torch.Tensor, 
                v:torch.Tensor, w:torch.Tensor, 
                u:torch.Tensor, s:torch.Tensor):
        with torch.no_grad():
            # Save the sizing & dtype
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            dtype = r.dtype
            ctx.dtype = dtype

            assert s.is_contiguous()
            assert w.is_contiguous()

            # Rest can be their respective types, but they are expected
            # to be consistent with each other
            assert dtype == k.dtype
            assert dtype == v.dtype
            assert dtype == u.dtype
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert u.is_contiguous()

            # Lets pre-compute the exp(-w)
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u, s.clone())

            s = s.to(dtype)

            # Output logits
            y = torch.empty((B, T, C), device=r.device, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

            # Call the cuda kernel
            if dtype == torch.bfloat16:
                wkv6state_cuda_kernel.forward(B, T, C, H, r, k, v, ew, u, s, y)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV6_CUDA")
            
            # Logits (without state)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            # Get the sizing & dtype
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            dtype = ctx.dtype

            # GY dtype
            assert gy.dtype == dtype
            assert gy.is_contiguous()
            r, k, v, ew, u, s = ctx.saved_tensors

            # Initialize all the backward pass vars required
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

            # Perform the backward pass
            if dtype == torch.bfloat16:
                wkv6state_cuda_kernel.backward(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV6_CUDA")

            #gw = torch.sum(gw, 0).view(H, C//H) # FIXME - not needed, because w is a different shape now in v6?
            gu = torch.sum(gu, 0).view(H, C//H)
            return (
                # B, T, C, H,
                None, None, None, None, 
                gr, gk, gv, gw, gu, gs)

@TCompileDisable 
@torch.jit.ignore
def RUN_WKV6STATE_CUDA(
    B:int, T:int, C:int, H:int, 
    r:torch.Tensor, k:torch.Tensor, 
    v:torch.Tensor, w:torch.Tensor, 
    u:torch.Tensor, s:torch.Tensor):
    return WKV6STATE_CUDA.apply(B, T, C, H, r, k, v, w, u, s)


def compute_wkv_state(
        k, v, r,
        time_faaaa: torch.nn.Parameter,
        time_decay: torch.nn.Parameter,
        wkv_state, 
        n_head:int, head_size:int,
        B:int, TT:int
    ):
    # Compute attent and the initial output tensor
    at = k @ v
    u = time_faaaa.view(1,1,n_head, 1, -1)

    # Slightly inefficent, but it works, lets compute all the tokens
    w = time_decay.exp().neg().exp().reshape(1, n_head,-1,1)

    out = (u * r) @ at
    for t in range(TT):
        out[:,t] += r[:,t] @ wkv_state
        
        # We make a clone copy, so the previous object backprop state is tracked seperately
        wkv_state = wkv_state.clone()
        wkv_state *= w
        wkv_state += at[:,t]

    return wkv_state, out


# @TCompileMax
# @JITFunction
# def x_logits_output_parsing(out_emb, head_size_divisor, B, TT, C, self_ln_x, self_output, g):
#     x_logits = out_emb.view(-1, C)
#     x_logits = self_ln_x(x_logits / head_size_divisor).view(B, TT, C)
#     return self_output(x_logits * g)
# RWKV TimeMix module

# RWKV TimeMix module
class RWKV6_0_TimeMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dropout, dim_att, lora_r, lora_alpha, lora_dropout,lora_quant,lora_quant_type,chunk_len:int = 128, precision:int = 64, max_ctx_len:int = 4096):
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.lora_quant = lora_quant
        self.lora_quant_type = lora_quant_type



        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            self.time_maa_x = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_g = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            TIME_MIX_EXTRA_DIM = 32
            if self.n_embd==4096:
                TIME_MIX_EXTRA_DIM = TIME_MIX_EXTRA_DIM*2
            #self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-0.01, 0.01))
            #self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd))
            self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-0.01, 0.01))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd))
            W_MIX_EXTRA_DIM = 64
            if self.n_embd==4096:
                W_MIX_EXTRA_DIM= W_MIX_EXTRA_DIM*2
            #self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, W_MIX_EXTRA_DIM).uniform_(-0.01, 0.01))
            #self.time_decay_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, n_embd))
            self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, W_MIX_EXTRA_DIM).uniform_(-0.01, 0.01))
            self.time_decay_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, dim_att))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            #self.time_decay = torch.ones(1, 1, 2048)#nn.Parameter(1,1,2048)#nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))

            
            #self.time_decay.device('cuda')
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))


        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        #self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        #self.key = nn.Linear(n_embd, dim_att, bias=False)
        #self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance = make_linear_ffn(n_embd, dim_att, bias=False, lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout , lora_quant=self.lora_quant,lora_quant_type=self.lora_quant_type)
        self.key = make_linear_ffn(n_embd, dim_att, bias=False, lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout , lora_quant=self.lora_quant,lora_quant_type=self.lora_quant_type)
        self.value = make_linear_ffn(n_embd, dim_att, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout , lora_quant=self.lora_quant,lora_quant_type=self.lora_quant_type)



        self.output = make_linear_ffn(n_embd, dim_att, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout , lora_quant=self.lora_quant,lora_quant_type=self.lora_quant_type)#nn.Linear(dim_att, n_embd, bias=False)
        self.gate = make_linear_ffn(n_embd, dim_att, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout , lora_quant=self.lora_quant,lora_quant_type=self.lora_quant_type)#nn.Linear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att)

        self.max_ctx_len = max_ctx_len
        self.use_cuda = True
        self._preload_cuda()
        self.chunk_len = chunk_len
        self.precision = precision
        
    # forwarding time mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last states containing of shape [
    #       [batch_size, state_size] ## Channel mix state,
    #       [batch_size, n_head, head_size, head_size] ## WKV state
    #   ]
    #
    # Returns a pair 
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [
    #       [batch_size, state_size] ## Channel mix state,
    #       [batch_size, n_head, head_size, head_size] ## WKV state
    #   ]
    def _preload_cuda(self):
        global wkv6state_cuda_kernel, RWKV_NO_CUDA

        # Skip preload if cuda is disabled
        if RWKV_NO_CUDA is True:
            self.use_cuda = False
            return

        # Load cuda if needed
        if wkv6state_cuda_kernel is None:
            # Log the compillation block
            print("---")
            print(f"[RWKV.TimeMix] Compiling CUDA kernel with HEAD_SIZE={self.head_size}")

            wkv6state_cuda_kernel = torch.utils.cpp_extension.load(
                name="wkv6state", 
                sources=[
                    os.path.join(code_dir, "cuda/wkv6state_op.cpp"), 
                    os.path.join(code_dir, "cuda/wkv6state_cuda_v1.cu")
                ],
                verbose=True, 
                extra_cuda_cflags=[
                    "-res-usage", 
                    "--use_fast_math", 
                    "-O3", 
                    "-Xptxas -O3", 
                    "--extra-device-vectorization", 
                    f"-D_N_={self.head_size}", 
                    f"-D_T_={self.max_ctx_len}"
                ]
            )

            # Close log the compillation block
            print(f"[RWKV.TimeMix6_0] CUDA kernel compiled & loaded globally")
            print("---")
        
        # Initialize the cuda kernel
        self.use_cuda = True
     
    #@JITModMethod
    @TCompileDisable
    @torch.jit.ignore
    def forward(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        # Run with cuda
        if self.use_cuda is True:
           return self._forward_cuda(x, last_state)
        
        # Run without cuda (cpu mode, etc)
        return self.forward_nocuda(x, last_state)
    def _forward_cuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:
        shift_state_out = x[:,-1]

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head

        assert T <= self.max_ctx_len, "max_ctx_len exceeded"

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # Get the xk, xv, xr, xg, xw, and rkvg
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)
        xw = x + dxprev * (self.time_maa_w + mw)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        u = self.time_faaaa

        # Logits and state
        wkv_state = last_state[1].to(r.dtype).clone().contiguous()

        # Perform the cuda forward pass
        x_logits = RUN_WKV6STATE_CUDA(
            B, T, C, H, 
            r, k, v, 
            w, 
            u,
            wkv_state
        )

        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits).view(B, T, C)
        x_logits = self.output(x_logits * g)
        #print('Cuda Kernel Mode')

        # Return the logits and the state

        return (x_logits, (shift_state_out,wkv_state))
    #@TCompileMax 
    #@JITModMethod 
    @TCompileDisable
    @torch.jit.ignore 
    def forward_nocuda(self, x, last_state: tuple[torch.Tensor,torch.Tensor])  -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]] :
        shift_state_out = x[:,-1]

        # 24 is optimal chunk length (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
        chunk_len = 24

        # padding to support fast path for non-exact chunk size multiple sequence lengths        
        n_padding = (chunk_len - x.size(-2) % chunk_len) % chunk_len
        if n_padding != 0:
            x = F.pad(x, [0, 0, 0, n_padding, 0, 0])

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        V = K

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(xg))

        w = self.time_decay.float().view(1,H,1,K).to('cuda')
        #self.time_decay_w1.to('cuda')
        #   self.time_decay_w2.to('cuda')
        xw.to('cuda')
        w = w + (torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).view(B, T, H, K).transpose(1, 2) # BHTK
        w = torch.exp(-torch.exp(w))

        u = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # Logits and state
        wkv_state = last_state[1].to(r.dtype)

        x_logits, wkv_state = rwkv_inner(r, k, v, w, u, wkv_state, chunk_len=chunk_len) 
        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = self._x_logits_gate(x_logits, g)

        if n_padding != 0:
            x_logits = x_logits[..., :-n_padding, :] # BHTV

        # Return the logits and the state
        return (x_logits, (shift_state_out,wkv_state))

    @TCompileMax 
    @JITModMethod  
    def _x_logits_gate(self, x_logits, gate):
        B, T, C = x_logits.size()
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / self.head_size_divisor).view(B, T, C)
        return self.output(x_logits * gate)
    