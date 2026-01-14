# -*- coding: utf-8 -*-
"""
GASDU (Gauss-Southwell Dynamic Update) Manager
Implements Algorithm 1 from the paper with a memory-optimized
"streaming selection" pass to find the Top-k threshold,
avoiding the creation of a massive concatenated gradient tensor.
"""

import torch
import math
from typing import Dict, Optional, TYPE_CHECKING


class GasduManager:
    """
    Manages the GASDU (Gauss-Southwell Dynamic Update) logic.
    
    [V2 - Memory Optimized Selection]
    This class handles the periodic refresh and reuse of the gradient mask
    as described in Algorithm 1 of the paper.
    
    It implements a streaming *selection* to find the Top-k
    threshold, avoiding the torch.cat() of all gradients.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        k_percent: float,
        m_period: int,
        device: torch.device
    ):
        """
        Initializes the GASDU manager.

        Args:
            model: The model being fine-tuned (must have .args attribute for grad_clip).
            optimizer: The optimizer (e.g., AdamW) used for training.
            k_percent: The percentage of parameters to update (e.g., 0.01).
            m_period: The refresh period (in epochs/steps) for the mask.
            device: The torch device (e.g., 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.k_percent = k_percent
        self.m_period = m_period
        self.device = device
        
        # This dictionary will store the boolean mask for each parameter
        # e.g., { 'param_name': torch.Tensor([True, False, ...]) }
        self.mask: Dict[str, torch.Tensor] = {}
        
        # Get total number of trainable parameters
        self.total_trainable_params = self._get_total_trainable_params()
        if self.total_trainable_params == 0:
            raise ValueError("[GASDU] Model has no trainable parameters.")
            
        # k (k_count) is the absolute number of parameters to update
        self.k_count = max(1, int(self.total_trainable_params * (self.k_percent / 100.0)))
        
        print(f"[GasduManager V2] Initialized. k={self.k_count} ({self.k_percent}%) | "
              f"M_period={self.m_period} | "
              f"Total Trainable Params={self.total_trainable_params}")

    def _get_total_trainable_params(self) -> int:
        """Helper to count trainable parameters."""
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                count += param.numel()
        return count

    @torch.no_grad()
    def _refresh_mask(self):
        """
        [V2] Refreshes the gradient mask using a memory-efficient streaming *selection*.
        
        This finds the global Top-k threshold by iterating through parameter
        gradients one by one, using only an O(k) tensor pool, thus avoiding
        the torch.cat() of all gradients.
        
        This corresponds to Algorithm 1, lines 4-5.
        """
        print(f"[GASDU V2] Refreshing mask (k={self.k_count}) using streaming selection...")
        
        # 1. Initialize the O(k) candidate pool for Top-k values.
        # We find the k-th largest by sorting and picking the smallest.
        top_k_pool = torch.full((self.k_count,), -1.0, device=self.device, dtype=torch.float32)
        current_threshold = -1.0
        
        # --- Pass 1: Find the global Top-k threshold ---
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
                
            # Get flat, absolute gradients for this parameter
            g_flat = param.grad.abs().detach().flatten()
            
            # Find contenders from this param that are larger than our
            # current k-th largest value (the minimum of the pool).
            contenders = g_flat[g_flat > current_threshold]
            
            if contenders.numel() > 0:
                # Combine our current pool with the new contenders
                combined = torch.cat([top_k_pool, contenders])
                
                # Sort and keep only the new k-largest
                # This is the most expensive part of the loop
                top_k_pool = torch.topk(combined, self.k_count, sorted=False).values
                
                # The new threshold is the smallest value in our k-largest pool
                current_threshold = top_k_pool.min()
                
            del g_flat, contenders

        # After Pass 1, `current_threshold` holds the true k-th largest gradient magnitude
        final_threshold = current_threshold.item()
        del top_k_pool # We only needed this to find the threshold
        
        if final_threshold < 0:
            print("[GASDU] Warning: Gradient threshold is negative. All gradients might be zero.")
            final_threshold = 0.0

        print(f"[GASDU V2] Pass 1 complete. Global gradient threshold={final_threshold:.2e}")

        # --- Pass 2: Create the boolean masks ---
        self.mask.clear() # Clear the old mask
        total_masked_params = 0
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            
            # Create boolean mask based on the global threshold
            mask = (param.grad.abs().detach() >= final_threshold)
            self.mask[name] = mask.to(self.device) # Store the mask
            total_masked_params += mask.sum().item()

        print(f"[GASDU V2] Pass 2 complete. Mask created. "
              f"Actual params in mask={total_masked_params} (Target k={self.k_count})")


    @torch.no_grad()
    def _apply_mask_to_grads(self):
        """
        Applies the stored mask to the model's current gradients in-place.
        This corresponds to Algorithm 1, line 9.
        """
        if not self.mask:
            print("[GASDU] Warning: Tried to apply an empty mask. Skipping.")
            return

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name in self.mask:
                    # Apply mask (element-wise multiplication)
                    param.grad.mul_(self.mask[name])
                else:
                    # This param was not present during last refresh (e.g., frozen)
                    # or had no gradient. Zero it out just in case.
                    param.grad.zero_()

    def step(self, loss: torch.Tensor, epoch: int):
        """
        Performs a full GASDU step: backward, mask refresh/reuse, and optimizer step.
        This function orchestrates the logic from Algorithm 1.
        
        Args:
            loss: The loss tensor for the current step.
            epoch: The current epoch number (or step number).
        """
        
        # 1. Calculate full gradients (Algorithm 1, line 4 or 9)
        self.optimizer.zero_grad()
        loss.backward() # This is the unavoidable gradient materialization step
        
        is_refresh_step = (epoch % self.m_period == 0)
        
        # 2. Refresh or Reuse Mask
        if is_refresh_step:
            # Algorithm 1, lines 4-5: Calculate and store new \Lambda_t
            self._refresh_mask()
        else:
            # Algorithm 1, line 7: Reuse \Lambda_{t-1} (i.e., do nothing)
            pass
            
        # 3. Apply Mask (Algorithm 1, line 9: \tilde{G}_t <- \Lambda_t \odot \nabla f(W_t))
        if self.mask:
            self._apply_mask_to_grads()
        else:
            if is_refresh_step:
                print("[GASDU] Warning: Mask is empty after refresh. Performing full gradient step.")
            # If mask is empty on a reuse step, something is wrong, but we proceed
            # with a full gradient step to avoid crashing.
            
        # 4. Optimizer Step (Algorithm 1, line 10)
        # We clip gradients *after* masking
        # 假设 model 实例上有 .args 属性
        grad_clip = getattr(self.model.args, "grad_clip", 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
        self.optimizer.step()

