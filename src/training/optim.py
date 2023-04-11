import torch
from collections import defaultdict
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)



def custom_params(model, params):
    """
    Custom parameters for Bert Models to handle weight decay and differentiated learning rates.
    Args:
        model (torch model]): Transformer model
        weight_decay (int, optional): Weight decay. Defaults to 0.
        lr (float, optional): LR of layers not belonging to the transformer. Defaults to 1e-3.
        lr_transfo (float, optional): LR of the last layer of the transformer. Defaults to 3e-5.
        lr_decay (float, optional): Factor to multiply lr_transfo when going deeper. Defaults to 1.
    Returns:
        list: Optimizer params.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_params = []
    for n, p in model.named_parameters():
        key = n.split('.')[1] if n.startswith('module.') else n.split('.')[0]
        lr, wd = params.get(key, params['logits'])
        wd = 0 if any(nd in n for nd in no_decay) else wd
        
#         print(n, lr, wd)
        opt_params.append({
            "params": [p],
            "weight_decay": wd,
            "lr": lr,
        })
    return opt_params


def define_optimizer(model, name, lr=1e-3, weight_decay=0, betas=(0.9, 0.999)):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.
    TODO

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.
    Raises:
        NotImplementedError: Specified optimizer name is not supported.
    Returns:
        torch optimizer: Optimizer
    """

    if weight_decay:
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        opt_params = []
        for n, p in model.named_parameters():
            wd = (
                0
                if any(nd in n for nd in no_decay)
                else weight_decay
            )
            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr,
                }
            )
    else:
        opt_params = model.parameters()
        
#     params = {
#         "encoder": [lr / 6, 0],
#         "frame_transformer_1": [lr / 3, 0],
#         "frame_transformer_2": [lr / 3 * 2, 0],
#         "frame_transformer_3": [lr, 0],
#         "logits": [lr * 2, 0],
#     }
#     opt_params = custom_params(model, params)

    if name.lower() == "ranger":
        radam = getattr(torch.optim, "RAdam")(opt_params, lr=lr, betas=betas)
        return Lookahead(radam, alpha=0.5, k=5)
    try:
        optimizer = getattr(torch.optim, name)(opt_params, lr=lr, betas=betas)
    except AttributeError:
        raise NotImplementedError(name)

    return optimizer


def freeze(model, prefix=""):
    """
    Freezes a model
    Arguments:
        model {torch model} -- Model to freeze
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix) or name.startswith("module." + prefix):
#             print(name)
            param.requires_grad = False

    for name, module in model.named_modules():
        if name.startswith(prefix) or name.startswith("module." + prefix):
            module.eval()
    
    
def unfreeze(model, prefix=""):
    """
    Unfreezes a model
    Arguments:
        modem {torch model} -- Model to unfreeze
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix) or name.startswith("module." + prefix):
            param.requires_grad = True

    for name, module in model.named_modules():
        if name.startswith(prefix) or name.startswith("module." + prefix):
            module.train()


def freeze_batchnorm(model):
    """
    Freezes the batch normalization layers of a model.
    Args:
        model (torch model): Model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
