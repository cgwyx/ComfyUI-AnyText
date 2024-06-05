from modelscope.pipelines import pipeline
import os
import folder_paths
import re
import torch
from modelscope.pipelines import pipeline
import cv2
import numpy as np
import cv2
from modelscope.hub.snapshot_download import snapshot_download

current_directory = os.path.dirname(os.path.abspath(__file__))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AnyText:
  
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AnyText_Loader": ("AnyText_Loader", {"forceInput": True}),
                "prompt": ("STRING", {"default": "A raccoon stands in front of the blackboard with the words \"你好呀~Hello!\" written on it.", "multiline": True}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "mode": (['text-generation', 'text-editing'],{"default": 'text-generation'}),  
                "sort_radio": (["↕", "↔"],{"default": "↔"}), 
                "revise_pos": ("BOOLEAN", {"default": False}),
                "img_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "ddim_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "show_debug": ("BOOLEAN", {"default": False}),
                "use_translator": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 9999, "min": -1, "max": 99999999}),
                "width": ("INT", {"default": 512, "min": 128, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1920, "step": 64}),
                # "width": ("INT", {"forceInput": True}),
                # "height": ("INT", {"forceInput": True}),
                "Random_Gen": ("BOOLEAN", {"default": False}),
                "fp16": ("BOOLEAN", {"default": True}),
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
                        "ori_image": ("ref", {"forceInput": True}),
                        "pos_image": ("pos", {"forceInput": True}),
                        },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "anytext_process"
    TITLE = "AnyText Geneation"

    def anytext_process(self,
        mode,
        AnyText_Loader,
        pos_image,
        ori_image,
        sort_radio,
        revise_pos,
        Random_Gen,
        prompt, 
        show_debug, 
        use_translator,
        img_count, 
        fp16,
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
        
        if width%64 == 0 and height%64 == 0:
            pass
        else:
            raise Exception(f"width and height must be multiple of 64(宽度和高度必须为64的倍数).")
        
        path = f"{current_directory}\scripts"
        
        loader_out = AnyText_Loader.split("|")
        
        if use_translator == True:#如果启用中译英，则提前判断本地是否存在翻译模型，没有则自动下载，以防跑半路报错。
            if os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en", "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                pass
            else:
                snapshot_download('damo/nlp_csanmt_translation_zh2en', revision='v1.0.1')
        pipe = pipeline('my-anytext-task', model=path, font_path=loader_out[0], ckpt_path=loader_out[1], clip_path=loader_out[2], use_fp16=fp16, translator_path=loader_out[3], use_translator=use_translator)
        n_lines = count_lines(prompt)
        if Random_Gen == True:
            pos_img = generate_rectangles(width, height, n_lines, max_trys=500)
        else:
            pos_img = pos_image
        if mode == "text-generation":
            ori_image = None
            revise_pos = revise_pos
        else:
            revise_pos = False
        # lora_path = r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\loras\ys艺术\sd15_mw_bpch_扁平风格插画v1d1.safetensors"
        # lora_ratio = 1
        # lora_path_ratio = str(lora_path)+ " " + str(lora_ratio)
        # print("\033[93m", lora_path_ratio, "\033[0m")
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
            # "lora_path_ratio": lora_path_ratio,
            }
        input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_img,
                "ori_image": ori_image,
                }
        print("\033[93mImg Resolution<=768x768 Recommended(图像分辨率,建议<=768x768):", width, "x", height, "\033[0m\n")
        if show_debug ==True:
            print("\033[93mloader from .util(从.util输入的loader):", AnyText_Loader, "\033[0m\n")
            print("\033[93mloader_out split form loader(分割loader得到4个参数):", loader_out, "\033[0m\n")
            print("\033[93mFont(字体)--loader_out[0],:", loader_out[0], "\033[0m\n")
            print("\033[93mAnyText Model(AnyText模型)--loader_out[1]:", loader_out[1], "\033[0m\n")
            print("\033[93mclip model(clip模型)--loader_out[2]:", loader_out[2], "\033[0m\n")
            print("\033[93mTranslator(翻译模型)--loader_out[3]:", loader_out[3], "\033[0m\n")
            print("\033[93mChinese2English translator(中译英):", use_translator, "\033[0m\n")
            print("\033[93mBackend scripts location(后端脚本位置):", path, "\033[0m\n")
            print("\033[93mNumber of text-content to generate(需要生成的文本数量):", n_lines, "\033[0m\n")
            print("\033[93mpos_image location(遮罩图位置):", pos_image, "\033[0m\n")
            print("\033[93mori_image location(原图位置):", ori_image, "\033[0m\n")
            print("\033[93mSort Position(文本生成位置排序):", sort_radio, "\033[0m\n")
            print("\033[93mEnable revise_pos(启用位置修正):", revise_pos, "\033[0m\n")
        x_samples, results, rtn_code, rtn_warning, debug_info = pipe(input_data, **params)
        if rtn_code < 0:
            raise Exception(f"Error in AnyText pipeline: {rtn_warning}")
        output = pil2tensor(x_samples)
        print("\n", debug_info)
        return(output)
        
# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText": AnyText,
}
