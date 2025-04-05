import clip
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import einops

from vqvae_muse import get_tokenizer_muse
from torch_vqvae_model import get_tokenizer
import matplotlib.pyplot as plt

from utils import read_image_to_tensor
import torch.nn as nn

# print(f"Devices available are: {torch.cuda.device_count()}")

device = "cuda:0"
"""
tokenizer = get_tokenizer_muse()
input_images = []
tokenizer.to(device)

# TODO: open ground images(base)
ground_truth_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/rigid_objects/im0/im_49.jpg")).unsqueeze(0)
ground_fake_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/rigid_objects/im0/output/output_image_20250308_161451.jpg")).unsqueeze(0)

# TODO: open compare images(very very similar to base, in the same scene+object layout but different object orientations)
compare_truth_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/rigid_objects/im4/im_46.jpg")).unsqueeze(0)
compare_fake_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/rigid_objects/im4/output/output_image_20250308_172806.jpg")).unsqueeze(0)

# TODO: open compare_medium images(slightly similar to base, in the same scene layout but different objects + orientations)
compare_medium_truth_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/soft_objects/im0/im_49.jpg")).unsqueeze(0)
compare_medium_fake_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/soft_objects/im0/output/output_image_20250308_192949.jpg")).unsqueeze(0)

# TODO: open compare_hard images(not anything like base images at all)
compare_hard_truth_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/bin_obj/im0/im_50.jpg")).unsqueeze(0)
compare_hard_fake_image = torch.tensor(read_image_to_tensor("/home/stevex2/data/vision-data/bridge_v2/bin_obj/im0/output/output_image_20250311_234204.jpg")).unsqueeze(0)
"""

model, preprocess = clip.load("ViT-B/32", device=device)
ground_truth_image = preprocess(Image.open("/home/stevex2/data/vision-data/bridge_v2/rigid_objects/im1/im_49.jpg")).unsqueeze(0).to(device)

"""
ground_truth_text_desc = clip.tokenize([
    "a robot arm positioned above a blue teddy bear, a brown teddy bear, a pink rabbit on the right", # TODO: soft_objects/im0/im_49.jpg
    "a robot arm positioned above a black teddy bear, a white teddy bear, a green rabbit on the right", # TODO: same as above, but wrong colors
    "a robot arm positioned above a blue cube, a yellow banana, and a green cucumber", # TODO: rigid_objects/im0/im_49.jpg
    "a robot arm positioned above a brown cube, an orange banana, and a black cucumber", # TODO: same as above, but wrong colors
    "a robot arm sweeping a piece of yellow cloth", # TODO: sweep/im0/im_29.jpg
    "a robot arm eating a piece of yellow cloth", # TODO: same as above but wrong action type
    "a picture of a kitchen scene", # TODO: any of these can be kitchen scenes
]).to(device)
"""


# TODO: taken from Bridge V2 Eval Slides, 
# TODO: first text description is VQA generated for an LVM output on rigid_objects: Move-cucumber
# TODO: second text description is VQA generated for rigid_objects: GT
# TODO: third text description is VQA generated for an LVM output on soft_objects: Move Green Bear Right
# TODO: second text description is VQA generated for soft_objects: GT
ground_truth_text_desc = clip.tokenize([
    "The robot arm has moved from its initial position in the first image to a new location above the wooden surface in the second image," +
    " indicating that it has been repositioned for interaction with the objects. A yellow banana-shaped object has been added to the scene" +
    " on the right side of the wooden surface in the second image. It is not present in the first image. A green cucumber-shaped object has" +
    " been placed on the wooden surface in the second image, near the center-right area. This object is also not present in the first image." +
    " The blue cube remains in the same position in both images, suggesting that it was not manipulated by the robot arm during the transition." + 
    " The other objects (yellow spoon-like object, white brush-like object, and the small blue and gray objects) appear to have remained stationary," + 
    " maintaining their positions relative to each other and the wooden surface. In summary, the primary changes involve the addition of the yellow" + 
    " banana-shaped and green cucumber-shaped objects, along with the repositioning of the robot arm. The remaining objects, including the blue cube," +
    " seem unchanged in terms of their placement.",
    
    "In the second image, the robot arm has moved from its initial position in the first image. It is now positioned over the green cucumber-like object" + 
    " on the wooden surface. The arm appears to be in the process of grasping or manipulating this object." + 
    " The green cucumber-like object has been slightly repositioned compared to its location in the first image. It seems to have been moved slightly" + 
    " towards the right and forward, closer to the edge of the wooden surface. All other objects on the wooden surface, including the yellow" +
    " banana-like object, the blue cube, the yellow brush with bristles, the yellow rectangular object, the gray sphere, and the small blue object," + 
    " remain in their original positions relative to the first image. There are no noticeable changes in their orientation or placement. The" + 
    " background elements, such as the wooden wall and the white structure on the right side, remain unchanged between the two images. The overall" + 
    " scene composition is consistent, with the primary difference being the movement of the robot arm and the slight repositioning of the green" + 
    " cucumber-like object.",
    
    "Stuffed Animals: The green stuffed animal with a red bow in the first image has been moved to the right side of the second image, and it" + 
    " appears to be lying on its back. The pink bunny in the first image has been flipped over and is now lying on its back in the second image," + 
    " with its head facing towards the bottom left corner. The leopard print stuffed animal in the first image has been moved slightly to the right" + 
    " and is now lying on its side in the second image. The blue stuffed animal in the first image has been moved slightly to the left and is now lying" + 
    " on its back in the second image. Robot Arm: The robot arm is present in both images but appears to have changed position. In the first image," + 
    " the arm is positioned above the stuffed animals, while in the second image, it is not directly above them, suggesting it has moved after manipulating the objects." + 
    " Overall Scene: The wooden surface and the background remain unchanged, indicating that the primary changes are the positions and orientations" + 
    " of the stuffed animals. In summary, the robot arm has manipulated the stuffed animals, moving them to different positions and changing their" + 
    " orientations. The green stuffed animal and the pink bunny have been flipped over, the leopard print stuffed animal has been moved slightly to" + 
    " the right, and the blue stuffed animal has been moved slightly to the left.",
    
    "The green stuffed toy with a red bow, which is shaped like a frog, has been moved from its original position in the first image to a new location" + 
    " near the pink rabbit-shaped stuffed toy in the second image. It appears to have been shifted slightly to the right and downward. The brown stuffed" + 
    " toy that resembles a bear or similar animal, located at the top of the wooden surface in the first image, has been repositioned closer to the blue" + 
    " stuffed toy resembling an elephant. This movement seems to be towards the center of the wooden surface. The robot arm's position has changed slightly" + 
    " between the two images. In the first image, it is positioned more towards the left side of the frame, while in the second image, it is more centered" + 
    " over the toys, indicating that it may have been involved in moving the objects. The other stuffed toys, including the leopard print one and the pink" + 
    " rabbit, remain in their original positions relative to each other, but their orientation might have been slightly altered due to the movements of the" +
    " other toys around them. Overall, the changes suggest that the robot arm was used to move the green frog-shaped toy and the brown bear-like toy to" + 
    " new locations on the wooden surface."

], context_length=)

with torch.no_grad():
    image_tok = model.encode_image(ground_truth_image)
    text_tok = model.encode_text(ground_truth_text_desc)
    
    logits_per_image, logits_per_text = model(ground_truth_image, ground_truth_text_desc)
    
    # print(f"Image logits: {logits_per_image} \n with shape {logits_per_image.shape}")
    # print(f"Text logits: {logits_per_text} \n with shape {logits_per_text.shape}")

    np.set_printoptions(suppress=True)
    image_probs = logits_per_image.softmax(dim=-1).cpu().numpy().squeeze(0)
    text_probs = logits_per_text.softmax(dim=0).cpu().numpy().squeeze(1)
    
print(f"Truth Image Probabilities: {image_probs}")
print(f"Truth Text Probabilities