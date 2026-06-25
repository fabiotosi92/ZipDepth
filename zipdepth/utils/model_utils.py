import torch
import torch.nn as nn


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    device = conv.weight.device
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True,
    ).to(device)
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused.weight.data = conv.weight.data.clone() * t
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=device)
    fused.bias.data = bn.bias + (b - bn.running_mean) * bn.weight / std
    return fused


def fuse_remaining_conv_bn(model: nn.Module) -> int:
    """Fuse consecutive Conv+BN pairs inside Sequential modules."""
    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        ch = list(module.named_children())
        for i in range(len(ch) - 1):
            n1, m1 = ch[i]
            n2, m2 = ch[i + 1]
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                setattr(module, n1, _fuse_conv_bn(m1, m2))
                setattr(module, n2, nn.Identity())
                count += 1
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            keep = [(n, m) for n, m in module.named_children() if not isinstance(m, nn.Identity)]
            if len(keep) < len(list(module.children())):
                for n, _ in list(module.named_children()):
                    delattr(module, n)
                for n, m in keep:
                    module.add_module(n, m)
    return count
