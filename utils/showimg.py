import cv2
import os
from utils.label_mapping import LABEL_MAPPING, BASIC_COLORS, NAME_ABBRE

problem_colors = {
    "FP" : (218,112,214),
    "FN" : (0,199,140),
    "ThreeBody" : (255,128,0)
}
def showimg(pic_num, noise_label_info, address, pic_prefix = '/mnt/intel/artifact_management/plusai_2d_dataset/revised/front/train/images/'):
    num = str(pic_num)
    if num in noise_label_info.keys():
        img_path = os.path.join(pic_prefix, noise_label_info[num]['img_id'])
        box_info = noise_label_info[num]['bboxes_unnormalized']
        wrong_idx = noise_label_info[num]['invalid_box_index']
        problems = noise_label_info[num]['problems']
        img = cv2.imread(img_path)
        for idx, pblm in zip(wrong_idx, problems):
            left_x, left_y, w, h = box_info[idx]
            right_x, right_y = left_x + w, left_y + h
            left_x, left_y, right_x, right_y = int(left_x), int(left_y), int(right_x), int(right_y)
            if pblm in problem_colors.keys():
                img = cv2.putText(img, pblm, (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, problem_colors[pblm], 1, cv2.LINE_AA)
                img = cv2.rectangle(img, (left_x, left_y), (right_x, right_y), problem_colors[pblm], 2)
            else:
                img = cv2.putText(img, pblm, (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,127,80), 1, cv2.LINE_AA)
                img = cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (255,127,80), 2)
        cv2.imwrite(f'/home/rui.cai/pic_code/{address}/{num}.jpg', img)
    else:
        print('not correct picture number')         

def show_groundtruth_img(gt_data, index, address='groundtruthimages',pic_prefix = '/mnt/intel/artifact_management/plusai_2d_dataset/revised/front/train/images/'):
    path = os.path.join(pic_prefix, gt_data[index]['filename'])
    image = cv2.imread(path)
    for anno in gt_data[index]['annotations']:
        if anno['class'] not in LABEL_MAPPING:
            continue
        x = anno['x']
        y = anno['y']
        w = anno['width']
        h = anno['height']
        top_x = int(x)
        top_y = int(y)
        buttom_x = int(x + w)
        buttom_y = int(y + h)
        color = BASIC_COLORS[LABEL_MAPPING[anno['class']]]
        image = cv2.putText(image, NAME_ABBRE[anno['class']], (top_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1, cv2.LINE_AA)
        image = cv2.rectangle(image, (top_x, top_y), (buttom_x, buttom_y), color, 2)
    cv2.imwrite(f'/home/rui.cai/pic_code/{address}/{index}.jpg', image)         

def show_pred_img(pred_data, pic_num, address='newpicsfrompred', pic_prefix = '/mnt/intel/artifact_management/plusai_2d_dataset/revised/front/train/images/'):
    img_path = os.path.join(pic_prefix, pred_data[pic_num]['img_id']) #get the whole address of each image
    img = cv2.imread(img_path)
    bbox_info = pred_data[pic_num]["bboxes_unnormalized"]#store the boxes in the 2-dimen matrix
    classes = pred_data[pic_num]['classes']
    for box,cls in zip(bbox_info, classes):
        c_x, c_y, w, h = box
        color = BASIC_COLORS[LABEL_MAPPING[cls]]
        left_x, left_y, right_x, right_y = int(c_x - w / 2), int(c_y - h / 2), int(c_x + w / 2), int(c_y + h / 2)
        img = cv2.putText(img, NAME_ABBRE[cls], (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1, cv2.LINE_AA)
        img = cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color, 2)
    cv2.imwrite(f'/home/rui.cai/pic_code/{address}/{pic_num}.jpg', img)

def generate_compare_images(pic_num:str, noise_label_info:dict, gt_data:dict, pred_data:dict, address:str, pic_prefix:str):
    img_path = os.path.join(pic_prefix, noise_label_info[pic_num]['img_id'])
    img = cv2.imread(img_path)

    if 'labeling' in gt_data.keys():
        gt_data = gt_data['labeling']

    #Label the image with ground truth annotation
    img_gt = cv2.imread(img_path)
    index = int(pic_num)
    for anno in gt_data[index]['annotations']:
        if anno['class'] not in LABEL_MAPPING:
            continue
        x = anno['x']
        y = anno['y']
        w = anno['width']
        h = anno['height']
        top_x = int(x)
        top_y = int(y)
        buttom_x = int(x + w)
        buttom_y = int(y + h)
        color = BASIC_COLORS[LABEL_MAPPING[anno['class']]]
        img_gt = cv2.putText(img_gt, NAME_ABBRE[anno['class']], (top_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,1, cv2.LINE_AA)
        img_gt = cv2.rectangle(img_gt, (top_x, top_y), (buttom_x, buttom_y), color, 2)

    #Label the image with prediction annotation
    img_pred = cv2.imread(img_path)
    bbox_info = pred_data[pic_num]["bboxes_unnormalized"]#store the boxes in the 2-dimen matrix
    classes = pred_data[pic_num]['classes']
    for box,cls in zip(bbox_info, classes):
        c_x, c_y, w, h = box
        color = BASIC_COLORS[LABEL_MAPPING[cls]]
        left_x, left_y, right_x, right_y = int(c_x - w / 2), int(c_y - h / 2), int(c_x + w / 2), int(c_y + h / 2)
        img_pred = cv2.putText(img_pred, NAME_ABBRE[cls], (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,1, cv2.LINE_AA)
        img_pred = cv2.rectangle(img_pred, (left_x, left_y), (right_x, right_y), color, 2)

    #Label the image with noisy samples information
    img_label = cv2.imread(img_path)
    box_info = noise_label_info[pic_num]['bboxes_unnormalized']
    wrong_idx = noise_label_info[pic_num]['invalid_box_index']
    problems = noise_label_info[pic_num]['problems']
    for idx, pblm in zip(wrong_idx, problems):
        left_x, left_y, w, h = box_info[idx]
        right_x, right_y = left_x + w, left_y + h
        left_x, left_y, right_x, right_y = int(left_x), int(left_y), int(right_x), int(right_y)
        if pblm in problem_colors.keys():
            img_label = cv2.putText(img_label, pblm, (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 1, problem_colors[pblm], 1, cv2.LINE_AA)
            img_label = cv2.rectangle(img_label, (left_x, left_y), (right_x, right_y), problem_colors[pblm], 2)
        else:
            img_label = cv2.putText(img_label, pblm, (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            img_label = cv2.rectangle(img_label, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)
    
    img = cv2.vconcat([img_gt, img_pred, img_label])
    cv2.imwrite(f'/home/rui.cai/pic_code/{address}/{pic_num}.jpg', img)
    
