import modules.scripts as scripts
import gradio as gr
import os
from modules import scripts


import modules.scripts as scripts
from PIL import Image , ImageFilter
import random
import re
import traceback
import math
import cv2
import gradio as gr
import numpy as np
import torch
from scipy.ndimage import binary_dilation


class EMScript(scripts.Script):
        # Extension title in menu UI

        def title(self):
                return "ExpandMask"

        # Decide to show menu in txt2img or img2img
        # - in "txt2img" -> is_img2img is `False`
        # - in "img2img" -> is_img2img is `True`
        #
        # below code always show extension menu
        def show(self, is_img2img):
                return scripts.AlwaysVisible
        
        

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion(f"Expanded Mask", open=False):
                        with gr.Row():
                                checkbox = gr.Checkbox(False,label="Enable")
                                new_expand = gr.Checkbox(False,label="New Expand")
                                expand_mask_blur = gr.Slider(label="Maskblur",minimum=0,maximum=100,step=2,value=10)
                                expand_sli = gr.Slider(label="Expand",minimum=5,maximum=100,step=5,value=10)
                        with gr.Row():
                                pcheckbox = gr.Checkbox(True,label="Padding Enable")
                                scalex = gr.Radio(["16","32","64","128","256"],label="Padding",value="128")

                # TODO: add more UI components (cf. https://gradio.app/docs/#components)
                return [expand_sli,expand_mask_blur,checkbox,scalex,pcheckbox,new_expand]

        # Extension main process
        # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
        # args is [StableDiffusionProcessing, UI1, UI2, ...]


        # def run(self,p,expand_sli,expand_mask_blur,checkbox):

                # def save_image_noget(x):
                #         # Assuming 'mask' is the key where the image data is stored in the dictionary
                #         image_data = x

                #         # Convert the image data to a PIL Image object
                #         pil_image = Image.fromarray(np.uint8(image_data))

                #         # Save the PIL Image to a file
                #         pil_image.save('D:\\Monser-sdee-ui\\Monser-sdee-ui\\temp.png')

                #         # Return the saved image path
                #         return pil_image

                # def expand_mask(sel_mask, expand_iteration=10):

                #         if not type(sel_mask) != "PIL.Image.Image":
                #                 new_sel_mask = sel_mask["mask"]
                #         else:
                #                 new_sel_mask = sel_mask


                #         expand_iteration = int(np.clip(expand_iteration, 1, 100))

                #         new_sel_mask = np.array(new_sel_mask, dtype=np.uint8)

                #         new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

                #         new_sel_mask = Image.fromarray(np.uint8(new_sel_mask))

                #         # save_image_noget(new_sel_mask)                      

                #         return new_sel_mask

                # print(checkbox)
                
                # proc = ""


                # if checkbox:
                #         p.image_mask = expand_mask(p.image_mask,expand_sli)
                #         img_mask = p.image_mask
                #         p.mask_blur = expand_mask_blur
                #         proc = process_images(p)

                #         proc.images.append(img_mask)

                # return proc


                #p.init_images[0]


                # TODO: add image edit process via Processed object proc
        def process(self,p,expand_sli,expand_mask_blur,checkbox,scalex,pcheckbox,new_expand):
                def expand_mask(sel_mask, expand_iteration=10):

                        if not type(sel_mask) != "PIL.Image.Image":
                                new_sel_mask = sel_mask["mask"]
                        else:
                                new_sel_mask = sel_mask


                        expand_iteration = int(np.clip(expand_iteration, 1, 100))

                        new_sel_mask = np.array(new_sel_mask, dtype=np.uint8)

                        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

                        new_sel_mask = Image.fromarray(np.uint8(new_sel_mask))
                        new_sel_mask = new_sel_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
                        # save_image_noget(new_sel_mask)   
                   

                        return new_sel_mask
                
                def dilate_mask(mask, dilation_amt):
                        mask = np.array(mask.convert("L"))
                        x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
                        center = dilation_amt // 2
                        dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
                        dilated_binary_img = binary_dilation(mask, dilation_kernel)
                        dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
                        return dilated_mask

                if checkbox:
                        p.mask_blur = int(math.ceil(expand_mask_blur))
                        if new_expand:
                                p.image_mask = dilate_mask(p.image_mask,expand_sli)
                        else:
                                p.image_mask = expand_mask(p.image_mask,expand_sli)
                        if pcheckbox:
                                p.inpaint_full_res_padding = int(scalex)
                        img_mask = p.image_mask


