from cgi import test
from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx] # This is the augmented dataset
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        angle_bins = np.linspace(0, 180, 9)[:-1]
        angle = data['angle'].numpy()
        # Convert negative angle to positive
        if angle<0:
            angle = 180 + angle
        center_point = data['center_point']
        rgb_img = data['rgb'] #(128, 128, 3) tensor
        
        # Label the keypoints on the original rgb image
        kpt = KeypointsOnImage([Keypoint(x=center_point[0], y=center_point[1])], shape=rgb_img.shape)
        # Find what bin the angle belongs to 
        bin_idx = np.argmin(np.absolute(angle_bins-angle))
        rot_angle = angle_bins[bin_idx]
        # Rotate the image BACK
        rot = iaa.Rotate(-rot_angle)
        img_rot, kpt_rot = rot(image=rgb_img.numpy(), keypoints=kpt) # img_rot is (128, 128, 3)
        # Scale img_rot value to [0,1], also put it to a tensor
        scaled_img_rot = torch.tensor(img_rot/255, dtype=torch.float32)
        # Permute the scaled img into (3, 128, 128) as the input
        input = scaled_img_rot.permute(2,0,1)
        kpt_rot_np = np.array([kpt_rot.keypoints[0].x, kpt_rot.keypoints[0].y]) 
        target = torch.tensor(get_gaussian_scoremap(shape = img_rot.shape[:2], keypoint=kpt_rot_np),
                               dtype = torch.float32).unsqueeze(0)
        data = {
            'input': input,
            'target': target
        }
        return data 


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img
    
    @staticmethod
    def get_action_from_pred(affordance_map, rgb_obs, rot_imgs, rot_angles):
        '''
        Self defined function for convenience
        '''
        reshaped_map = affordance_map.flatten()
        idx = torch.argmax(reshaped_map)
        val = reshaped_map[idx]

        # Convert idx to coord
        angle_idx = idx // (affordance_map.shape[-2]*affordance_map.shape[-1])

        # Since KeypointsOnImage method takes in a shape (H, W, (C)), we need to permute the best affordance map or the rot_img
        best_rot_img_permute = rot_imgs[angle_idx].permute(1,2,0) #(128, 128, 3)
        coord_idx = idx % (affordance_map.shape[-2]*affordance_map.shape[-1])
        row_idx = coord_idx // rgb_obs.shape[1] # y
        col_idx = coord_idx % rgb_obs.shape[1] # x
        x = col_idx.detach().numpy().item()
        y = row_idx.detach().numpy().item() # Unrecovered coord
        angle = rot_angles[angle_idx]

        # Define kpt and recoverer:
        kpt = KeypointsOnImage([Keypoint(x, y)], shape=best_rot_img_permute.shape)
        recoverer = iaa.Rotate(-angle)

        # Recover the center point by rotating back the best affordance map (permuted)
        recovered_img, coord_kpt = recoverer(image=best_rot_img_permute.cpu().detach().numpy(), keypoints=kpt)
        coord = (int(coord_kpt.keypoints[0].x), int(coord_kpt.keypoints[0].y))
        angle = 180 - angle

        return coord, angle, angle_idx.cpu().detach().numpy(), x, y, val

    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_imgkeypoints
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # Return whether we are using GPU or CPU
        # ===============================================================================
        coord, angle = None, None 
        # rgb_obs.to(device) #(128, 128, 3) numpy
        # Rotate the img 8 times
        rot_angles = np.linspace(0, 180, 9)[:-1] 
        rot_imgs = torch.zeros((len(rot_angles), 3, rgb_obs.shape[0], rgb_obs.shape[1]), dtype=torch.float32) # N=8, 3, 128, 128
        for i in range(len(rot_angles)):
            # Define the rotater
            rotater = iaa.Rotate(rot_angles[i])
            # Rotate the img
            rot_img = rotater.augment_image(rgb_obs)
            # Convert into tersor, scale to [0,1] and reshape into (3, 128, 128)
            rot_img_tensor = torch.tensor(rot_img/255, dtype=torch.float32).permute(2,0,1)
            rot_imgs[i] = rot_img_tensor
        rot_imgs = rot_imgs.to(device)
        pred = self.predict(rot_imgs) # N affordance map. 
        # The highest vote represents the grasping center.
        # (N=8,1,128,128)

        affordance_map = torch.clip(pred, 0, 1).cpu() # Clip values to (0,1), no change to shape
        affordance_map_suppressed = affordance_map.clone()
        # Step 1: for all max_coord (represented in (bin, x, y)), downvote the overall affordance map by subtracting a supression map

        for max_coord in list(self.past_actions):        
            bin = max_coord[0]
            affordance_map_to_suppress = affordance_map_suppressed[bin,0]
            # Penalize around all the max_coord positions in the current affordance map
            suppression_map = get_gaussian_scoremap(shape=affordance_map_to_suppress.shape, keypoint=np.array(max_coord[1:]), sigma=4)
            # supress past actions and select next-best action
            affordance_map_to_suppress -= torch.from_numpy(0.2*suppression_map)
            affordance_map_suppressed[bin,0] = affordance_map_to_suppress

        # Step 2: Calculate the second best action 

        coord, angle, angle_idx, x, y, val = self.get_action_from_pred(affordance_map_suppressed, rgb_obs, rot_imgs, rot_angles) 
        coord_univ = (angle_idx, x, y)

        # ===============================================================================
        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================

        # After the first attempt, if the grasp failed (the robot failed to 
        # interact with any object in the scene), then we should penalize this position,
        # and then using tprch.argmax to find the best action again.

        # Since in the new round of attempt, the robot would receive an identical observation,
        # it will try to predict the same action (coord). However since we already downvoted that
        # action, the model will predict the next biggest affordace value.
        # We will have at most 8 chances to try different actions.


        # if above calculated action appears in the memory, indicating that the previous grasp failed, and 
        # now the model attempts to redo the failure

        # Steo 3: Append the failed action to the deque
        self.past_actions.append(coord_univ)

        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================

        # The visualize method takes in rgb images of shape (3,H,W) and permutes the rgb image into (H,W,3)

        # Visualize all 8 preditions along with their rgb inputs
        vis_img_8 = np.zeros((0, 2*rgb_obs.shape[1], 3)) # (8*128, 256, 3)
        for i in range(len(rot_angles)): # Iterate over all 8 imgs
            vis_rgb = rot_imgs[i].cpu().detach().numpy() # 3, 128, 128 numpy
            if i == angle_idx:
                # the winner image
                # first convert the image to 128, 128, 3 and scale back to [0, 255]
                vis_rgb = (np.moveaxis(vis_rgb*255, 0, -1)).astype(np.uint8)
                vis_rgb = np.ascontiguousarray(vis_rgb)
                # draw the grasping indicator
                draw_grasp(vis_rgb, (x, y), angle=0) # The angle is 0 because this is before recovering
                # then transform vis_rgb back to 3, 128, 128 and scale to [0, 1]
                vis_rgb = (np.moveaxis(vis_rgb/255, -1, 0)).astype(np.float32)
            vis_pred = pred[i].cpu().detach().numpy() # 1, 128, 128 numpy
            concat_img = self.visualize(vis_rgb, vis_pred) # 128, 256, 3
            # Draw the split lines
            concat_img[-1] = np.ones((concat_img.shape[1],3)) * 127
            vis_img_8 = np.concatenate((vis_img_8, concat_img), axis=0)
        upper_half = vis_img_8[:512,:,:]
        lower_half = vis_img_8[512:,:,:]
        vis_img = np.concatenate((upper_half, lower_half), axis=1).astype(np.uint8)
        # ===============================================================================
        return coord, angle, vis_img

