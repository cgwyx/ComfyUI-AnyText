#from .AnyText.cldm.model import create_model, load_state_dict
from modelscope.pipelines import pipeline
import os
import folder_paths
import re
import torch
from modelscope.pipelines import pipeline
import cv2
import numpy as np
import node_helpers
from PIL import Image, ImageOps, ImageSequence
import hashlib
import cv2

# os.system("python app.py")
current_directory = os.path.dirname(os.path.abspath(__file__))
# img_save_folder = "temp"
# script_path = os.path.join(current_directory, "app.py")
# with open(script_path, "r", encoding='UTF-8') as f:
#         code = f.read()
# exec(code)
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# def tensor2np(tensor):
#     img = tensor.mul(255).byte()
#     img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
#     return img

# def create_canvas(w=512, h=512, c=3, line=5):
#     image = np.full((h, w, c), 200, dtype=np.uint8)
#     for i in range(h):
#         if i % (w//line) == 0:
#             image[i, :, :] = 150
#     for j in range(w):
#         if j % (w//line) == 0:
#             image[:, j, :] = 150
#     image[h//2-8:h//2+8, w//2-8:w//2+8, :] = [200, 0, 0]
#     return image

class AnyText:
  
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # paths = []
        # for search_path in folder_paths.get_folder_paths("checkpoints"):
        #     if os.path.exists(search_path):
        #         for root, subdir, files in os.walk(search_path, followlinks=True):
        #             if "configuration.json" in files:
        #                 paths.append(os.path.relpath(root, start=search_path))
        return {
            "required": {
                # "font_dir":(os.listdir(os.path.join(folder_paths.models_dir, "fonts")), 'utf-8', ),
                "prompt": ("STRING", {"default": "A raccoon stands in front of the blackboard with the words \"你好呀~Hello!\" written on it.", "multiline": True}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "mode": (['text-generation', 'text-editing'],{"default": 'text-generation'}),  
                "sort_radio": (["↕", "↔"],{"default": "↔"}), 
                "revise_pos": ("BOOLEAN", {"default": False}),
                "img_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "ddim_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "show_debug": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 9999, "min": -1, "max": 99999999}),
                # "width": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 16}),
                # "height": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 16}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "Random_Gen": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", {
                    "default": 1.00,
                    "min": -999999,
                    "max": 9999999,
                    "step": 0.01
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 9,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
            },
            "optional": {
                        "ori_image": ("STRING", {"forceInput": True}),
                        "pos_image": ("STRING", {"forceInput": True}),
                        },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "anytext_process"
    TITLE = "AnyText Geneation"

    def anytext_process(self,
        mode,
        # font_dir,
        pos_image,
        ori_image,
        sort_radio,
        revise_pos,
        Random_Gen,
        prompt, 
        show_debug, 
        img_count, 
        ddim_steps=20, 
        strength=1, 
        cfg_scale=9, 
        seed="", 
        eta=0.0, 
        a_prompt="", 
        n_prompt="", 
        width=512, 
        height=512,
    ):
        def check_overlap_polygon(rect_pts1, rect_pts2):
            poly1 = cv2.convexHull(rect_pts1)
            poly2 = cv2.convexHull(rect_pts2)
            rect1 = cv2.boundingRect(poly1)
            rect2 = cv2.boundingRect(poly2)
            if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
                return True
            return False
        
        def count_lines(prompt):
            prompt = prompt.replace('“', '"')
            prompt = prompt.replace('”', '"')
            p = '"(.*?)"'
            strs = re.findall(p, prompt)
            if len(strs) == 0:
                strs = [' ']
            return len(strs)
        
        def generate_rectangles(w, h, n, max_trys=200):
            img = np.zeros((h, w, 1), dtype=np.uint8)
            rectangles = []
            attempts = 0
            n_pass = 0
            low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
            while attempts < max_trys:
                rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
                ratio = np.random.uniform(4, 10)
                rect_h = max(low_edge, int(rect_w/ratio))
                rect_h = min(rect_h, int(h*0.8))
                # gen rotate angle
                rotation_angle = 0
                rand_value = np.random.rand()
                if rand_value < 0.7:
                    pass
                elif rand_value < 0.8:
                    rotation_angle = np.random.randint(0, 40)
                elif rand_value < 0.9:
                    rotation_angle = np.random.randint(140, 180)
                else:
                    rotation_angle = np.random.randint(85, 95)
                # rand position
                x = np.random.randint(0, w - rect_w)
                y = np.random.randint(0, h - rect_h)
                # get vertex
                rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
                rect_pts = np.int32(rect_pts)
                # move
                rect_pts += (x, y)
                # check boarder
                if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
                    attempts += 1
                    continue
                # check overlap
                if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles): # type: ignore
                    attempts += 1
                    continue
                n_pass += 1
                cv2.fillPoly(img, [rect_pts], 255)
                rectangles.append(rect_pts)
                if n_pass == n:
                    break
                print("attempts:", attempts)
            if len(rectangles) != n:
                raise Exception(f'Failed in auto generate positions after {attempts} attempts, try again!')
            return img
        
        # for search_path in folder_paths.get_folder_paths("checkpoints"):
        #     if os.path.exists(search_path):
        #         path = os.path.join(search_path, model_dir)
        #         if os.path.exists(path):
        #             model_path = path
        #             break
        path = f"{current_directory}\scripts"
        print("\033[93mBackend scripts location(后端脚本位置):\033[0m", path)
        pipe = pipeline('my-anytext-task', model=path, use_fp16=True, use_translator=False)
        n_lines = count_lines(prompt)
        print("\033[93mNumber of text-content to draw(需要生成的文本数量):\033[0m", n_lines)
        ref_image = ori_image
        print("\033[93mpos_imagg location(遮罩图位置):\033[0m", pos_image)
        print("\033[93mori_image location(原图位置):\033[0m", ori_image)
        if Random_Gen == True:
            pos_img = generate_rectangles(width, height, n_lines, max_trys=500)
        else:
            pos_img = pos_image
        
        if mode == "text-editing":
            params = {
            "mode": mode,
            "sort_priority": sort_radio,
            "revise_pos": False,
            "show_debug": show_debug,
            "image_count": img_count,
            "ddim_steps": ddim_steps - 1,
            "image_width": width,
            "image_height": height,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "eta": eta,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            }
            input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_img,
                "ori_image": ref_image,
                }
        else:
            params = {
            "mode": mode,
            "sort_priority": sort_radio,
            "revise_pos": revise_pos,
            "show_debug": show_debug,
            "image_count": img_count,
            "ddim_steps": ddim_steps - 1,
            "image_width": width,
            "image_height": height,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "eta": eta,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            }
            input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_img,
                }
        print("\033[93mDraw Order(文本生成优先位置):\033[0m", sort_radio)
        print("\033[93mrevise_pos set to(位置修正设置):\033[0m", revise_pos)
        x_samples, results, rtn_code, rtn_warning, debug_info = pipe(input_data, **params)
        if rtn_code < 0:
            raise Exception(f"Error in AnyText pipeline: {rtn_warning}")
        output = pil2tensor(x_samples)
        return(output)

class AnyText_Pose_IMG:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "image": (sorted(files), {"image_upload": True}),
                        },
                }

    CATEGORY = "ExtraModels/AnyText"
    RETURN_TYPES = (
        "INT", 
        "INT", 
        "STRING", 
        "STRING", 
        "STRING", 
        "IMAGE")
    RETURN_NAMES = (
        "width", 
        "height", 
        "ori_img", 
        "comfy_mask_pos_img", 
        "gr_mask_pose_img", 
        "mask_img")
    FUNCTION = "AnyText_Pose_IMG"
    TITLE = "AnyText Pose IMG"
    
    def AnyText_Pose_IMG(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        comfy_mask_pos_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_mask_pos_img.png")
        gr_mask_pose_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gr_mask_pos_imgs.png")
        img = node_helpers.pillow(Image.open, image_path)
        width = img.width
        height = img.height
        print("\033[93mInput Resolution(输入图像大小):\033[0m", width, "x", height)
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            # output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # output_image = output_images[0]
            output_mask = output_masks[0]
        invert_mask = 1.0 - output_mask
        inverted_mask_image = invert_mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        i = 255. * inverted_mask_image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save("custom_nodes\ComfyUI-AnyText\AnyText\comfy_mask_pos_img.png")

        return (
            width, 
            height, 
            image_path, 
            comfy_mask_pos_img_path, 
            gr_mask_pose_image_path, 
            inverted_mask_image)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText": AnyText,
    "AnyText_Pose_IMG": AnyText_Pose_IMG,
}
