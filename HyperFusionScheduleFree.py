import torch
import torch.optim
import math
from typing import Callable
from muon import muon_update

class HyperFusionScheduleFree(torch.optim.Optimizer):
    r"""
    A simplified and stabilized HyperFusion Optimizer.
    Core features: Schedule-Free, RAdam-Rectify, ADOPT, Cautious.
    Removed experimental features (DeMo, FALCON, GEACS, Compression) for stability.
    """
    def __init__(self,
                 params,
                 lr: float = 0.0025,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 # --- Stability Options ---
                 use_radam_rectify: bool = True,   # Default True for stability
                 use_adopt_denominator: bool = True, # Default True for ADOPT behavior
                 cautious: bool = True,            # Default True for Cautious behavior
                 # --- Muon Options ---
                 use_muon: bool = False,
                 muon_momentum: float = 0.95,
                 muon_ns_steps: int = 5,
                 muon_nesterov: bool = True,
                 ):

        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        use_radam_rectify=use_radam_rectify,
                        use_adopt_denominator=use_adopt_denominator,
                        cautious=cautious,
                        use_muon=use_muon,
                        muon_momentum=muon_momentum,
                        muon_ns_steps=muon_ns_steps,
                        muon_nesterov=muon_nesterov,
                        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Safety check for NaN/Inf
                        z = state['z']
                        if not torch.isfinite(z).all(): continue
                        p.data.lerp_(end=z, weight=1 - 1 / group['betas'][0])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        z = state['z']
                        if not torch.isfinite(z).all(): continue
                        p.data.lerp_(end=z, weight=1 - group['betas'][0])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            k = group['k']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            # --- Phase 1: RAdam Warmup Check ---
            is_radam_warmup = False
            radam_rectify_term = 1.0
            
            if group['use_radam_rectify']:
                beta2_t = beta2 ** (k + 1)
                N_sma_max = 2 / (1 - beta2) - 1
                if N_sma_max > 0:
                    N_sma = N_sma_max - 2 * (k + 1) * beta2_t / (1 - beta2_t) if (1 - beta2_t) > 0 else N_sma_max
                    if N_sma < 5:
                        is_radam_warmup = True
                    else:
                        r_t_num = (N_sma - 4) * (N_sma - 2) * N_sma_max
                        r_t_den = (N_sma_max - 4) * (N_sma_max - 2) * N_sma
                        if r_t_den > 0:
                            radam_rectify_term = math.sqrt(r_t_num / r_t_den)
            
            if is_radam_warmup:
                # RAdam Warmup: Simple Momentum Update
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    if not torch.isfinite(grad).all(): continue

                    state = self.state[p]
                    if 'exp_avg' not in state:
                        state['z'] = torch.clone(p.data)
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # Standard Adam second moment
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                group['k'] += 1
                continue
            
            # --- Normal Phase: Coefficient Calculation ---
            if k < group['warmup_steps']: sched = (k + 1) / group['warmup_steps']
            else: sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k + 1)
            if bias_correction2 <= 0: bias_correction2 = 1e-8 # Safety

            effective_lr = group['lr'] * sched * math.sqrt(bias_correction2)
            if group['use_radam_rectify']:
                effective_lr *= radam_rectify_term

            lr_max = group['lr_max'] = max(effective_lr, group['lr_max'])
            
            weight_term = (lr_max ** group['weight_lr_power'])
            if math.isinf(weight_term): weight_term = 1e10

            weight = ((k + 1) ** group['r']) * weight_term
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0
            
            adaptive_y_lr = effective_lr * (1 - beta1 * (1 - ckp1))

            # --- Data Collection Phase ---
            params_with_grad = []
            grads_original = []
            exp_avgs = []
            exp_avg_sqs = []
            zs = []
            
            for p in group['params']:
                if p.grad is None: continue
                # Improved Stability: Skip update if grad is infinite
                if not torch.isfinite(p.grad).all(): continue

                params_with_grad.append(p)
                grads_original.append(p.grad.data)
                
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                zs.append(state['z'])

            if not params_with_grad:
                group['k'] += 1
                continue

            # --- Calculation Phase ---
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads_original, alpha=1 - beta1)

            # ADOPT Logic
            if group['use_adopt_denominator'] and k > 0:
                # Use previous step's second moment for denominator
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denom, eps)
            else:
                # Standard update sequence
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads_original, grads_original, value=1 - beta2)
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denom, eps)

            grads_for_update = list(torch._foreach_div(grads_original, denom))

            if group['use_muon']:
                for i, p in enumerate(params_with_grad):
                    if p.ndim < 2:
                        continue
                    state = self.state[p]
                    if 'muon_momentum_buffer' not in state:
                        state['muon_momentum_buffer'] = torch.zeros_like(p.data)
                    muon_vec = muon_update(grads_original[i], state['muon_momentum_buffer'],
                                           beta=group['muon_momentum'],
                                           ns_steps=group['muon_ns_steps'],
                                           nesterov=group['muon_nesterov'])
                    if muon_vec.shape != p.data.shape:
                        muon_vec = muon_vec.reshape(p.data.shape)
                    grads_for_update[i] = muon_vec

            # Weight Decay (Standard L2)
            if group['weight_decay'] != 0:
                torch._foreach_add_(grads_for_update, [p.data for p in params_with_grad], alpha=group['weight_decay'])

            # --- Update Phase (Cautious or Standard) ---
            if group['cautious']:
                # Hybrid-like execution for Cautious logic
                for i in range(len(params_with_grad)):
                    p, grad_orig, grad_norm = params_with_grad[i], grads_original[i], grads_for_update[i]
                    y, z = p.data, zs[i]
                    
                    # Calculate "Standard" update step u
                    # u = ckp1 * (y - z) + adaptive_y_lr * grad_norm
                    u = (y - z).mul(ckp1).add(grad_norm, alpha=adaptive_y_lr)
                    
                    # Cautious Mask: Only update where u and grad_orig have same sign
                    mask = (u * grad_orig > 0).to(grad_orig.dtype)
                    
                    # Apply update: y = y - u * mask
                    y.sub_(u.mul(mask))
            else:
                # Fully Foreach Path (Faster, Standard ScheduleFree)
                y_list = [p.data for p in params_with_grad]
                # y = y * (1-ckp1) + z * ckp1 - adaptive_y_lr * grad
                torch._foreach_mul_(y_list, 1.0 - ckp1)
                torch._foreach_add_(y_list, zs, alpha=ckp1)
                torch._foreach_sub_(y_list, grads_for_update, alpha=adaptive_y_lr)

            # --- Finalize ---
            # Update Z (Slow Weights): z = z - effective_lr * grad_norm
            torch._foreach_sub_(zs, grads_for_update, alpha=effective_lr)

            # ADOPT: Update second moment AFTER parameter update
            if group['use_adopt_denominator']:
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads_original, grads_original, value=1 - beta2)

            group['k'] += 1
        return loss
