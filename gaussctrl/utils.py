import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND

def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(disparity_map[None]) 

    detected_maps = np.concatenate(disparity_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')

def create_reprojection_mask(
    target_depth, # [B, H, W] depth map of target view
    K_target,     # [B, 3, 3] intrinsics of target
    E_target,     # [B, 4, 4] extrinsics (world-to-camera) of target
    K_ref,        # [B, 3, 3] intrinsics of reference
    E_ref,        # [B, 4, 4] extrinsics of reference
    tolerance=2.0 # How strict we want the mask to be in pixels
):
    B, H, W = target_depth.shape
    device = target_depth.device
    
    # 1. Create a grid of all pixels in the target image
    # Note: meshgrid indexing 'ij' gives (y, x) so we stack as (x, y) to get (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), indexing='ij')
    
    # [H, W, 3] -> [B, H*W, 3]
    pixels_target = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    pixels_target = pixels_target.view(-1, 3).unsqueeze(0).repeat(B, 1, 1) # [B, H*W, 3]
    
    # Flatten depth [B, H, W] -> [B, H*W]
    depth_flat = target_depth.view(B, -1)
    
    # 2. Unproject Target Pixels into 3D Space using Depth
    # Formula: P_3D = K^-1 * p_2D * depth
    K_inv = torch.inverse(K_target) # [B, 3, 3]
    
    # bmm: [B, 3, 3] x [B, 3, H*W] -> [B, 3, H*W]
    rays_camera = torch.bmm(K_inv, pixels_target.transpose(1, 2))
    points_camera_target = rays_camera * depth_flat.unsqueeze(1) # [B, 3, H*W]
    
    # Convert to homogeneous coordinates (add a 1 to the end)
    points_camera_target = torch.cat([points_camera_target, torch.ones((B, 1, H*W), device=device)], dim=1) # [B, 4, H*W]
    
    # Transform to World Space (Target Camera -> World)
    # E_target is World -> Camera, so we need inverse for Camera -> World
    E_inv = torch.inverse(E_target) # [B, 4, 4]
    points_world = torch.bmm(E_inv, points_camera_target) # [B, 4, H*W]
    
    # 3. Reproject 3D World Points into Reference Camera (World -> Reference Camera -> 2D)
    points_camera_ref = torch.bmm(E_ref, points_world) # [B, 4, H*W]
    points_camera_ref = points_camera_ref[:, :3, :] # drop homogeneous 1: [B, 3, H*W]
    
    # Project to 2D using Reference Intrinsics
    pixels_ref_homogeneous = torch.bmm(K_ref, points_camera_ref) # [B, 3, H*W]
    
    # Divide by Z to get actual 2D (u,v) coordinates in the reference image
    Z = pixels_ref_homogeneous[:, 2, :] + 1e-6
    u_ref = pixels_ref_homogeneous[:, 0, :] / Z # [B, H*W]
    v_ref = pixels_ref_homogeneous[:, 1, :] / Z # [B, H*W]
    
    # Filter out points that fall behind the camera
    valid_depth = (Z > 0).float()
    
    # 4. Build the N x N Attention Mask
    # We want a mask of size [B, H*W, H*W] where mask[b, i, j] = 1 if target pixel i 
    # projects to reference pixel j, and 0 otherwise.
    
    N = H * W
    
    # Target pixel projected coords
    u_ref = u_ref.unsqueeze(2) # [B, N, 1]
    v_ref = v_ref.unsqueeze(2) # [B, N, 1]
    
    # Reference image pixel coordinates
    x_ref = pixels_target[..., 0].unsqueeze(1) # [B, 1, N]
    y_ref = pixels_target[..., 1].unsqueeze(1) # [B, 1, N]
    
    # Calculate squared distance
    dist_x_sq = (u_ref - x_ref) ** 2 # [B, N, N]
    dist_y_sq = (v_ref - y_ref) ** 2 # [B, N, N]
    distance_squared = dist_x_sq + dist_y_sq
    
    # Add penalty for points that projected behind the reference camera
    valid_mask = valid_depth.unsqueeze(2) # [B, N, 1]
    
    # Create mask: 0 if valid and close enough, -10000.0 if invalid
    mask = torch.where(
        (distance_squared < tolerance**2) & (valid_mask > 0.5), 
        torch.zeros_like(distance_squared), 
        torch.full_like(distance_squared, -10000.0)
    )
    
    return mask

def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask, epipolar_mask=None, num_refs=0):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    
    # Apply standard query/key multiplication
    attention_scores = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    
    # Apply Epipolar Mask if provided
    if epipolar_mask is not None:
        # Note: attention_scores might inherently apply softmax in newer Diffusers versions.
        # We assume get_attention_scores returns the post-softmax probabilities here
        # based on the original code, but if so we must intercept it.
        # However, the diffusers source code `get_attention_scores` literally calls softmax.
        # For epipolar, we must multiply the epipolar mask BEFORE softmax if we can,
        # but since diffusers doesn't expose it cleanly, we can multiply the probabilities
        # and re-normalize.
        
        # Epipolar mask is expected to be [B, N, N] where 1 is valid, 0 is invalid
        # But we need it to match the attention heads shape
        b_heads, seq_q, seq_k = attention_scores.shape
        B = epipolar_mask.shape[0]  # chunk_size (target frames only)
        # b_heads = 2 * video_length * num_attn_heads  (2 from CFG doubling)
        num_attn_heads = b_heads // (2 * video_length)

        # Build full mask: ones (no constraint) for ref frames, epipolar mask for target frames
        full_mask = torch.ones(b_heads, seq_q, seq_k, device=epipolar_mask.device, dtype=epipolar_mask.dtype)
        for j in range(B):
            target_frame_idx = num_refs + j
            for cfg_pass in range(2):
                batch_idx = cfg_pass * video_length + target_frame_idx
                head_start = batch_idx * num_attn_heads
                head_end = head_start + num_attn_heads
                full_mask[head_start:head_end] = epipolar_mask[j:j+1].expand(num_attn_heads, -1, -1)

        # Zero out invalid probabilities and re-normalize
        attention_scores = attention_scores * full_mask
        attention_scores = attention_scores / (attention_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
    hidden_states_ref_cross = torch.bmm(attention_scores, value_ref_cross) 
    return hidden_states_ref_cross

class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff
        self.camera_data = None
        self.epipolar_masks = None
        
    def set_camera_data(self, target_depths, K_target, E_target, K_refs, E_refs):
        """
        Calculates and caches the epipolar masks for this chunk to avoid re-calculating them
        for every attention layer.
        target_depths: [B, 512, 512]
        K_target: [B, 3, 3] etc.
        """
        self.camera_data = True
        self.epipolar_masks = []
        import torch.nn.functional as F
        
        # We need a mask for each reference view (typically 4)
        num_refs = K_refs.shape[0] // target_depths.shape[0] # assuming K_refs is stacked or expanded
        B = target_depths.shape[0]
        
        # The K_refs and E_refs should be tensors of shape [num_refs, 3, 3] etc.
        # We expand them to match Batch size
        for r in range(num_refs):
            # Extract this specific reference's matrices
            K_r = K_refs[r:r+1].expand(B, -1, -1)
            E_r = E_refs[r:r+1].expand(B, -1, -1)
            
            # Calculate the full resolution mask [B, 512*512, 512*512]
            # Since this is extremely memory heavy (262144x262144), 
            # we must only calculate downsampled masks dynamically inside __call__ based on current patch size
            # OR we store the tensors here and compute on-the-fly.
            pass
            
        # Instead of precomputing the massive mask, we store the raw geometry
        self.target_depths = target_depths
        self.K_target = K_target
        self.E_target = E_target
        self.K_refs = K_refs
        self.E_refs = E_refs

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            ref0_frame_index = [0] * video_length
            ref1_frame_index = [1] * video_length
            ref2_frame_index = [2] * video_length
            ref3_frame_index = [3] * video_length

            # Epipolar MASK Computation
            mask0 = mask1 = mask2 = mask3 = None
            num_refs_in_batch = 0
            if self.camera_data is not None:
                # Calculate mask for the current UNet feature map resolution (e.g. 64x64, 32x32, 16x16)
                res = int(np.sqrt(key.shape[1])) # e.g. 4096 -> 64
                if res * res == key.shape[1]:
                    import torch.nn.functional as F
                    # Rescale depth map to current resolution
                    downsampled_depth = F.interpolate(
                        self.target_depths.unsqueeze(1).float(), 
                        size=(res, res), mode='bilinear', align_corners=False
                    ).squeeze(1)
                    
                    # Adjust Intrinsics for downsampling
                    scale = res / 512.0
                    K_target_scaled = self.K_target.clone()
                    K_target_scaled[:, 0, 0] *= scale
                    K_target_scaled[:, 1, 1] *= scale
                    K_target_scaled[:, 0, 2] *= scale
                    K_target_scaled[:, 1, 2] *= scale
                    
                    K_refs_scaled = self.K_refs.clone()
                    K_refs_scaled[:, 0, 0] *= scale
                    K_refs_scaled[:, 1, 1] *= scale
                    K_refs_scaled[:, 0, 2] *= scale
                    K_refs_scaled[:, 1, 2] *= scale
                    
                    B = downsampled_depth.shape[0]
                    num_refs_in_batch = video_length - B
                    # Compute masks. `create_reprojection_mask` returns [B, res*res, res*res] of -10000.0 or 0.0
                    # We pass exp(mask) since we modified compute_attn to multiply probabilities.
                    # so 0.0 becomes 1.0 (valid), -10000.0 becomes 0.0 (invalid)
                    mask0 = torch.exp(create_reprojection_mask(
                        downsampled_depth, K_target_scaled, self.E_target,
                        K_refs_scaled[0:1].expand(B, -1, -1), self.E_refs[0:1].expand(B, -1, -1), tolerance=2.0
                    ))
                    mask1 = torch.exp(create_reprojection_mask(
                        downsampled_depth, K_target_scaled, self.E_target,
                        K_refs_scaled[1:2].expand(B, -1, -1), self.E_refs[1:2].expand(B, -1, -1), tolerance=2.0
                    ))
                    mask2 = torch.exp(create_reprojection_mask(
                        downsampled_depth, K_target_scaled, self.E_target,
                        K_refs_scaled[2:3].expand(B, -1, -1), self.E_refs[2:3].expand(B, -1, -1), tolerance=2.0
                    ))
                    mask3 = torch.exp(create_reprojection_mask(
                        downsampled_depth, K_target_scaled, self.E_target,
                        K_refs_scaled[3:4].expand(B, -1, -1), self.E_refs[3:4].expand(B, -1, -1), tolerance=2.0
                    ))
            
            hidden_states_ref0 = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask, mask0, num_refs_in_batch)
            hidden_states_ref1 = compute_attn(attn, query, key, value, video_length, ref1_frame_index, attention_mask, mask1, num_refs_in_batch)
            hidden_states_ref2 = compute_attn(attn, query, key, value, video_length, ref2_frame_index, attention_mask, mask2, num_refs_in_batch)

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, ref3_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, ref3_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Same mask logic for the 4th reference view
        if not is_cross_attention and mask3 is not None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            b_heads, seq_q, seq_k = attention_probs.shape
            B = mask3.shape[0]
            num_attn_heads = b_heads // (2 * video_length)
            full_mask = torch.ones(b_heads, seq_q, seq_k, device=mask3.device, dtype=mask3.dtype)
            for j in range(B):
                target_frame_idx = num_refs_in_batch + j
                for cfg_pass in range(2):
                    batch_idx = cfg_pass * video_length + target_frame_idx
                    head_start = batch_idx * num_attn_heads
                    head_end = head_start + num_attn_heads
                    full_mask[head_start:head_end] = mask3[j:j+1].expand(num_attn_heads, -1, -1)
            attention_probs = attention_probs * full_mask
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-8)
            hidden_states_ref3 = torch.bmm(attention_probs, value)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states_ref3 = torch.bmm(attention_probs, value)
        
        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * torch.mean(torch.stack([hidden_states_ref0, hidden_states_ref1, hidden_states_ref2, hidden_states_ref3]), dim=0) if not is_cross_attention else hidden_states_ref3 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
