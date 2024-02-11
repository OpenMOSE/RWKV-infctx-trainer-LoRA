# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp
from .LoraTrainer import *



class RWKV_ChannelMix(JITModClass):
    
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn,lora_r,lora_alpha,lora_dropout):
        super().__init__()
        # self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout  

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))



        #with torch.no_grad():  # fancy init of time_mix
        #    ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
        ##    ddd = torch.ones(1, 1, n_embd)
        #    for i in range(n_embd):
        #        ddd[0, 0, i] = i / n_embd
        #    self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        #    self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        #self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        #self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        #self.value = nn.Linear(dim_ffn, n_embd, bias=False)
        self.key = make_linear_att(n_embd, dim_ffn, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout )
        self.receptance = make_linear_att(n_embd, n_embd, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout )
        self.value = make_linear_att(dim_ffn, n_embd, bias=False,  lora_r=self.lora_r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout  )

    # forwarding channel mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last shift states of the various batches [batch_size, state_size]
    #
    # Returns a pair 
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [batch_size, state_size]
    @TCompileMax
    @JITModMethod
    def forward(self, x, last_state: torch.Tensor):
        # out_emb, out_state = channelMix_batchForward(
        #     self.time_mix_k,self.time_mix_r,
        #     self.key.bi
        #     self.key.weight, self.receptance.weight, self.value.weight,
        #     x, last_state.shift_state
        # )
        # return (out_emb, (out_state))
    
        #xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
        #                  dim=1)
        
        #xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        #xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        #kv = self.value( torch.relu( self.key(xk) ) ** 2 )
                          
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
    
        return (torch.sigmoid(self.receptance(xr)) * kv,
                (x[:, -1]))

# Pure lambda implementation, of forwarding channel mix given the model weights
# and the input tokens and states.
#
# Returns a pair 
# - of output embedding of shape [batch_size, seq_len, embedding_size]
# - and the output state of shape [batch_size, seq_len, state_size]
#
# @ TCompileMax
# @ JITFunction
def channelMix_batchForward(
    # Various weights from the channel mix layer
    time_maa_k,
    time_maa_r,
    w_key: torch.nn.Module, 
    receptance: torch.nn.Module, 
    value: torch.nn.Module,
    # Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    x_embedding,
    # Last shift states of the various batches [batch_size, state_size]
    last_shift_state
):
    # Compute accordingly the full state shift
    # [batch_size, seq_len, state_size]
    full_state_shift = torch.concat((last_shift_state.unsqueeze(1), x_embedding[:, :-1]), dim=1)

    # Computing the channel mix components
    xk = modified_lerp(x_embedding, time_maa_k, full_state_shift)
    xr = modified_lerp(x_embedding, time_maa_r, full_state_shift)
    k  = w_key(xk)
    kv = value( torch.relu(k) ** 2 )

    # Compute the output embeddings, and the last_shift_state
    return (torch.sigmoid(receptance(xr)) * kv, x_embedding[:, -1])