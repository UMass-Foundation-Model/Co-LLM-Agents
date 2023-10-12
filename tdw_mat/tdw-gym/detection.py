from mmdet.apis import DetInferencer
from pycocotools import mask
import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
detector = None
class tdw_detection:
    def __init__(self):
        self.inferencer = DetInferencer(
            model = "detection_pipeline/config.py",
            weights = "detection_pipeline/epoch_1.pth",
            device = "cuda"
        )

        self.name_map = {
            0:    'b04_bowl_smooth',
            1:    'plate06',
            2:    'teatray',
            3:    'basket_18inx18inx12iin_plastic_lattice',
            4:    'basket_18inx18inx12iin_wicker',
            5:    'basket_18inx18inx12iin_wood_mesh',
            6:    'bread',
            7:    'b03_burger',
            8:    'b03_loafbread',
            9:    'apple',
            10:    'b04_banana',
            11:    'b04_orange_00',
            12:    'f10_apple_iphone_4',
            13:    'b05_executive_pen',
            14:    'key_brass',
            15:    'apple_ipod_touch_yellow_vray',
            16:    'b04_lighter',
            17:    'small_purse',
            18:    'b05_calculator',
            19:    'pencil_all',
            20:    'mouse_02_vray',
            21:    'bed',
        }

    def cls_to_name_map(self, cls_id):
        return self.name_map[cls_id]

    def __call__(self, img_path, decode = True, no_save_pred = True, out_dir = ''):
        # input can be a path or rgb image
        result = self.inferencer(img_path, no_save_pred = no_save_pred, out_dir = out_dir)
        if decode and result['predictions'][0]['masks'] != []:
            rle_format = result['predictions'][0]['masks']
            # print('rle:', rle_format)
            mask_format = mask.decode(rle_format)
            # print('mask:', mask_format)
            result['predictions'][0]['masks'] = mask_format
        return result

def init_detection():
    global detector
    if detector == None:
        detector = tdw_detection()
    return detector

from PIL import Image
def main():
    tdw = tdw_detection()
    result = tdw("demo/demo_images/img_0.jpg", decode = False, no_save_pred = False, out_dir = 'outputs')
    print(result)
    img = Image.open("demo/demo_images/img_0.jpg")
    img = np.array(img)[..., [2, 1, 0]]
    # need to change channel here.
    result = tdw(img, decode = True)
    print(result)
    print(result['predictions'][0]['masks'].shape)
    
if __name__ == '__main__':
    main()
