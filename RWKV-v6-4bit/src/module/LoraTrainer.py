########################################################################################################
# LoRA
########################################################################################################
from .CoreDependencies import *
import bitsandbytes as bnb
import functools
LORA_CONFIG = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.01,
    "parts": {"att", "ln", "time","ffn"},
}    
@TCompileDisable
@torch.jit.ignore
class LoraLinear(nn.Module):

    @TCompileDisable
    @torch.jit.ignore
    def __init__(self, in_features: int, out_features: int, bias: bool, lora_r:float, lora_alpha:float, lora_dropout:float, lora_quant:bool, lora_quant_type:str):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.is_quant = False
        self.grad = None
        self.lora_quant = lora_quant
        self.lora_quant_type = lora_quant_type

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r = int(self.lora_r)
        alpha = int(self.lora_alpha)
        dropout = self.lora_dropout

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    @TCompileDisable
    @torch.jit.ignore
    def quant(self, quant_type):
        self.is_quant = True
        self.quant_type = quant_type
        #self.dummy_tensor = nn.Parameter(torch.zeros(1))
        # 現在のデバイスを取得
        current_device = self.weight.data.device
        if self.quant_type=='4bit':
            self.weight.data, self.qstate= bnb.functional.quantize_4bit((self.weight.data).to('cuda'))
        elif self.quant_type=='nf4':
            self.weight.data, self.qstate= bnb.functional.quantize_nf4((self.weight.data).to('cuda'))
        elif self.quant_type=='fp4':
            self.weight.data, self.qstate= bnb.functional.quantize_fp4((self.weight.data).to('cuda'))
    @TCompileDisable
    @torch.jit.ignore
    def forward(self, x):
        if self.is_quant:
            if self.quant_type=='4bit':
                return F.linear(x, bnb.functional.dequantize_4bit(self.weight.data,quant_state=self.qstate).to(torch.bfloat16)) + self.scaling * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            elif self.quant_type=='nf4':
                return F.linear(x, bnb.functional.dequantize_nf4(self.weight.data,quant_state=self.qstate).to(torch.bfloat16)) + self.scaling * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            elif self.quant_type=='fp4':
                return F.linear(x, bnb.functional.dequantize_fp4(self.weight.data,quant_state=self.qstate).to(torch.bfloat16)) + self.scaling * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        else:
            #print('unquant forward')
            return F.linear(x, self.weight) + self.scaling * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        #return (
        #    F.linear(x, self.weight) + self.scaling *
        #    F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    return LoraLinear(*args, **kwargs)
    #global LORA_CONFIG
    #if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
    #    #print("LORA MODE")
    #    return LoraLinear(*args, **kwargs)
    #else:
    #    print("Not LORA MODe")
    #    return nn.Linear(*args, **kwargs)
        


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    return LoraLinear(*args, **kwargs)
    #global LORA_CONFIG
    #if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
    #    #print("LORA MODE")
    #    return LoraLinear(*args, **kwargs)
    #else:
    #    print("Not LORA MODe")
    #    return nn.Linear(*args, **kwargs)
    