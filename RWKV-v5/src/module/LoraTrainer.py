########################################################################################################
# LoRA
########################################################################################################
from .CoreDependencies import *
import functools
LORA_CONFIG = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.01,
    "parts": {"att", "ln", "time","ffn"},
}    
class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool, lora_r:float, lora_alpha:float, lora_dropout:float):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        #print(f'on loratrainer lora_r={lora_r}')

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        #r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
        #    "alpha"], LORA_CONFIG["dropout"]
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

    def forward(self, x):
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


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
    