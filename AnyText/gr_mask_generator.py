import os
import cv2
import gradio as gr
import numpy as np
from gradio.components import Component
from scripts.util import check_channels, resize_image, save_images
import json


BBOX_MAX_NUM = 8
img_save_folder = 'SaveImages'

def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False


def draw_rects(width, height, rects):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for rect in rects:
        x1 = int(rect[0] * width)
        y1 = int(rect[1] * height)
        w = int(rect[2] * width)
        h = int(rect[3] * height)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img


def process(mode, pos_radio, show_debug, draw_img, ref_img, ori_img, w, h, *rect_list):
    # Text Generation
    if mode == 'gen':
        # create pos_imgs
        if pos_radio == 'Manual-draw(手绘)':
            if draw_img is not None:
                pos_imgs = 255 - draw_img['image']
                if 'mask' in draw_img:
                    pos_imgs = pos_imgs.astype(np.float32) + draw_img['mask'][..., 0:3].astype(np.float32)
                    pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
            else:
                pos_imgs = np.zeros((w, h, 1))
        elif pos_radio == 'Manual-rect(拖框)':
            rect_check = rect_list[:BBOX_MAX_NUM]
            rect_xywh = rect_list[BBOX_MAX_NUM:]
            checked_rects = []
            for idx, c in enumerate(rect_check):
                if c:
                    _xywh = rect_xywh[4*idx:4*(idx+1)]
                    checked_rects += [_xywh]
            pos_imgs = draw_rects(w, h, checked_rects)
        # print(pos_imgs)
    # Text Editing
    elif mode == 'edit':
        # revise_pos = False  # disable pos revise in edit mode
        if ref_img is None or ori_img is None:
            raise gr.Error('No reference image, please upload one for edit!')
        edit_image = ori_img.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=768)
        h, w = edit_image.shape[:2]
        if isinstance(ref_img, dict) and 'mask' in ref_img and ref_img['mask'].mean() > 0:
            pos_imgs = 255 - edit_image
            edit_mask = cv2.resize(ref_img['mask'][..., 0:3], (w, h))
            pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
            pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        else:
            if isinstance(ref_img, dict) and 'image' in ref_img:
                ref_img = ref_img['image']
            pos_imgs = 255 - ref_img  # example input ref_img is used as pos
        # print(pos_imgs)
    
    # print(pos_imgs)
    # save_images(pos_imgs,'./')
    # cv2.imwrite('pos_imgs.png', 255-pos_imgs[..., ::-1])
    results = cv2.imwrite('gr_mask_pos_imgs.png', 255-pos_imgs[..., ::-1])
    debug_info = results
    return results, gr.Markdown(debug_info, visible=show_debug)


def create_canvas(w=512, h=512, c=3, line=5):
    image = np.full((h, w, c), 200, dtype=np.uint8)
    for i in range(h):
        if i % (w//line) == 0:
            image[i, :, :] = 150
    for j in range(w):
        if j % (w//line) == 0:
            image[:, j, :] = 150
    image[h//2-8:h//2+8, w//2-8:w//2+8, :] = [200, 0, 0]
    return image


def resize_w(w, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (w, img1.shape[0])), cv2.resize(img2, (w, img2.shape[0]))]


def resize_h(h, img1, img2):
    if isinstance(img2, dict):
        img2 = img2['image']
    return [cv2.resize(img1, (img1.shape[1], h)), cv2.resize(img2, (img2.shape[1], h))]


is_t2i = 'true'
current_directory = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_directory, "javascript/style.css")
# block = gr.Blocks(css='javascript/style.css', theme=gr.themes.Soft()).queue()
block = gr.Blocks(css=css_path, theme=gr.themes.Soft()).queue()
js_path = os.path.join(current_directory, "javascript/bboxHint.js")
# with open('javascript/bboxHint.js', 'r', encoding='UTF-8') as file:
with open(js_path, 'r', encoding='UTF-8') as file:
    value = file.read()
escaped_value = json.dumps(value)

with block:
    block.load(fn=None,
               _js=f"""() => {{
               const script = document.createElement("script");
               const text =  document.createTextNode({escaped_value});
               script.appendChild(text);
               document.head.appendChild(script);
               }}""")
    with gr.Row(variant='compact'):
        with gr.Column(min_width=1600) as left_part:
            pass
        with gr.Column():
            pass
            # result_gallery = gr.Gallery(label='Result(结果)', show_label=True, preview=True, columns=2, allow_preview=True, height=600)
            # result_info = gr.Markdown('', visible=False)
        with left_part:
            with gr.Accordion('🛠Parameters(参数)', open=True):
                with gr.Row(variant='compact'):
                    image_width = gr.Slider(label="Image Width(宽度)", minimum=256, maximum=768, value=640, step=64)
                    image_height = gr.Slider(label="Image Height(高度)", minimum=256, maximum=768, value=640, step=64)
            with gr.Tabs() as tab_modes:
                        
                with gr.Tab("🖼Text Generation(文字生成)", elem_id='MD-tab-t2i') as mode_gen:
                    pos_radio = gr.Radio(["Manual-draw(手绘)", "Manual-rect(拖框)"], value='Manual-draw(手绘)', label="Pos-Method(位置方式)", info="choose a method to specify text positions(选择方法用于指定文字位置).")
                    with gr.Row(variant='compact'):
                        rect_cb_list: list[Component] = []
                        rect_xywh_list: list[Component] = []
                        for i in range(BBOX_MAX_NUM):
                            e = gr.Checkbox(label=f'{i}', value=False, visible=False, min_width='10')
                            x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-x', visible=False)
                            y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-y',  visible=False)
                            w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-w',  visible=False)
                            h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.0001, elem_id=f'MD-t2i-{i}-h',  visible=False)
                            x.change(fn=None, inputs=x, outputs=x, _js=f'v => onBoxChange({is_t2i}, {i}, "x", v)', show_progress=False, queue=False)
                            y.change(fn=None, inputs=y, outputs=y, _js=f'v => onBoxChange({is_t2i}, {i}, "y", v)', show_progress=False, queue=False)
                            w.change(fn=None, inputs=w, outputs=w, _js=f'v => onBoxChange({is_t2i}, {i}, "w", v)', show_progress=False, queue=False)
                            h.change(fn=None, inputs=h, outputs=h, _js=f'v => onBoxChange({is_t2i}, {i}, "h", v)', show_progress=False, queue=False)

                            e.change(fn=None, inputs=e, outputs=e, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)', queue=False)
                            rect_cb_list.extend([e])
                            rect_xywh_list.extend([x, y, w, h])

                    rect_img = gr.Image(value=create_canvas(), label="Rext Position(方框位置)", elem_id="MD-bbox-rect-t2i", show_label=False, visible=False)
                    draw_img = gr.Image(value=create_canvas(), label="Draw Position(绘制位置)", visible=True, tool='sketch', show_label=False, brush_radius=100)

                    def re_draw():
                        return [gr.Image(value=create_canvas(), tool='sketch'), gr.Slider(value=512), gr.Slider(value=512)]
                    draw_img.clear(re_draw, None, [draw_img, image_width, image_height])
                    image_width.release(resize_w, [image_width, rect_img, draw_img], [rect_img, draw_img])
                    image_height.release(resize_h, [image_height, rect_img, draw_img], [rect_img, draw_img])

                    def change_options(selected_option):
                        return [gr.Checkbox(visible=selected_option == 'Manual-rect(拖框)')] * BBOX_MAX_NUM + \
                                [gr.Image(visible=selected_option == 'Manual-rect(拖框)'),
                                 gr.Image(visible=selected_option == 'Manual-draw(手绘)')]
                    pos_radio.change(change_options, pos_radio, rect_cb_list + [rect_img, draw_img], show_progress=False, queue=False)
                    with gr.Row():
                        gr.Markdown("")
                        run_gen = gr.Button(value="Run(运行)!", scale=1, elem_classes='run')
                        gr.Markdown("")
                
                
                with gr.Tab("🎨Text Editing(文字编辑)") as mode_edit:
                    with gr.Row(variant='compact'):
                        ref_img = gr.Image(label='Ref(参考图)', source='upload', height=400)
                        ori_img = gr.Image(label='Ori(原图)', scale=1, height=400)

                    def upload_ref(x):
                        return [gr.Image(type="numpy", brush_radius=100, tool='sketch'),
                                gr.Image(value=x)]

                    def clear_ref(x):
                        return gr.Image(source='upload', tool=None)
                    ref_img.upload(upload_ref, ref_img, [ref_img, ori_img])
                    ref_img.clear(clear_ref, ref_img, ref_img)
                    with gr.Row():
                        gr.Markdown("")
                        run_edit = gr.Button(value="Run(运行)!", scale=1, elem_classes='run')
                        gr.Markdown("")

                
    ips = [pos_radio, draw_img, rect_img, ref_img, ori_img, image_width, image_height, *(rect_cb_list+rect_xywh_list)]
    # run_gen.click(fn=process, inputs=[gr.State('gen')] + ips, outputs=[result_gallery, result_info])
    # run_edit.click(fn=process, inputs=[gr.State('edit')] + ips, outputs=[result_gallery, result_info])
    run_gen.click(fn=process, inputs=[gr.State('gen')] + ips)
    run_edit.click(fn=process, inputs=[gr.State('edit')] + ips)


# block.launch(
#     server_name='0.0.0.0' if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1",
#     share=False,
#     root_path=f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv('GRADIO_PROXY_PATH') else ""
# )
block.launch(server_name='127.0.0.1', server_port=9999)
