import json
import os
import cv2
import numpy as np
from utils.box_iou import _box_iou, box_with_iou
from utils.label_mapping import LABEL_MAPPING_BLUR, LABEL_MAPPING, NAME_ABBRE

vehicle_class = [0,1,2,8,9]
moto_class = [3, 4, 5]

def compare_vehicles(pred_data : dict, gt_data : dict, args: list) -> tuple[dict, int]:
    sw, sh, mw, mh, thres_con, thres_iou = args
    count = 0
    noise_label_info = dict()
    if isinstance(gt_data, dict) and 'labeling' in gt_data.keys():
        gt_data = gt_data['labeling']
    for pic_num, gt_img in zip(pred_data.keys(), gt_data):
        invalid_index, problems, bboxs = [], [], []
        pred_box_info = pred_data[pic_num]["bboxes_unnormalized"]
        scores = pred_data[pic_num]["scores"]
        classes = pred_data[pic_num]['classes']
        img_path = pred_data[pic_num]["img_id"]
        visited = np.zeros(len(pred_box_info))
        for idx, anno in enumerate(gt_img["annotations"]):
            ##Finding the FP problem
            xg, yg, wg, hg = anno['x'], anno['y'], anno['width'], anno['height'] #read the gt_box location
            gt_box = [xg, yg, wg, hg]
            if anno['class'] not in LABEL_MAPPING or wg <= sw or hg <= sh:
                bboxs.append(gt_box)
                continue
            #Check if any center point of pred image is in b area, save in target points set
            target_points = []
            for pred_idx in range(len(pred_box_info)):
                pred_box = pred_box_info[pred_idx]
                cxp, cyp, wp, hp = pred_box
                if visited[pred_idx] == 0 and xg <= cxp <= xg + wg and yg <= cyp <= yg + hg:
                    target_points.append(pred_box)
            #If target points set is empty -> mark as over-annotated problem, continue
            if len(target_points) == 0:
                bboxs.append(gt_box)
                if anno['occluded'] == 'none' and anno['truncated_vehicle'] == 'no' and anno['clear']:
                    flag = True
                    for ano in gt_img["annotations"]:# Check if the car is located above a truck
                        x, y, wt, ht = ano['x'], ano['y'], ano['width'], ano['height']
                        if 'transporter' in ano.keys():
                            if (yg - 1.5 * hg <= y <= yg + 1.5 * hg) & (xg - 1.5 * wg <= x <= xg + 1.5 * wg) & ano['transporter']:
                                flag = False
                                break
                    if flag and LABEL_MAPPING[anno['class']] in vehicle_class:
                        invalid_index.append(idx)
                        problems.append('FP')
                continue
            #Compute the IoU between target points set and b, filter the points that satisfy threshold and choose the biggest IoU as matched point
            target_box_info = []
            for box in target_points:
                gt = [xg, yg, xg + wg, yg + hg]
                cx, cy, w, h = box
                pb = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                iou = _box_iou(pb, gt)
                if iou >= thres_iou:
                    target_box_info.append(box_with_iou(box, iou))
            #No such a matched point -> mark as over-annotated problem, continue
            if len(target_box_info) == 0:
                bboxs.append(gt_box)
                if anno['occluded'] == 'none' and anno['truncated_vehicle'] == 'no' and anno['clear']:
                    flag = True
                    for ano in gt_img["annotations"]:# Check if the car is located above a truck
                        x, y, wt, ht = ano['x'], ano['y'], ano['width'], ano['height']
                        if 'transporter' in ano.keys():
                            if (yg - 1.5 * hg <= y <= yg + 1.5 * hg) & (xg - 1.5 * wg <= x <= xg + 1.5 * wg) & ano['transporter']:
                                flag = False
                                break
                    if flag and LABEL_MAPPING[anno['class']] in vehicle_class:
                        invalid_index.append(idx)
                        problems.append('FP')
                continue
            target_box_info.sort(key = lambda x: x.iou, reverse = True)
            match_box = target_box_info[0].box
            match_box_index = pred_box_info.index(match_box)
            ## Check the label problem
            if LABEL_MAPPING[anno['class']] in vehicle_class or LABEL_MAPPING[classes[match_box_index]] in vehicle_class:
                if sw <= wg <= mw or sh <= hg <= mh:
                    if (LABEL_MAPPING[classes[match_box_index]] not in vehicle_class) and (LABEL_MAPPING[classes[match_box_index]] != 3):
                        invalid_index.append(idx)
                        problems.append('La_' + NAME_ABBRE[anno['class']]+'_' + NAME_ABBRE[classes[match_box_index]])#saved with a form of label_ground-truth label_prediction label
                elif wg > mw or hg > mh:
                    if LABEL_MAPPING_BLUR[classes[match_box_index]] != LABEL_MAPPING_BLUR[anno['class']] and (LABEL_MAPPING[classes[match_box_index]] != 3):
                        invalid_index.append(idx)
                        problems.append('La_' + NAME_ABBRE[anno['class']]+'_' + NAME_ABBRE[classes[match_box_index]])
            #Sign a state bit for the matched picture in pred image as visited
            visited[match_box_index] = 1
            bboxs.append(gt_box)
        ## Check the FN problem
        unvisited_index = np.where(visited == 0)[0]
        for uidx in unvisited_index:
            cx, cy, wp, hp = pred_box_info[uidx]
            if LABEL_MAPPING[classes[uidx]] not in vehicle_class or wp <= sw + 5 or hp <= sh + 5 or pred_data[pic_num]['occluded'][uidx][0] > 0.001:
                continue
            if scores[uidx] > thres_con:
                # surrounding = False
                # for anno in gt_img["annotations"]:
                #     xg, yg, wg, hg = anno['x'], anno['y'], anno['width'], anno['height'] #read the gt_box location
                #     if cx < xg
                saved_box_form = [cx - wp / 2, cy - hp / 2, wp, hp]
                bboxs.append(saved_box_form)
                invalid_index.append(len(bboxs) - 1)
                problems.append('FN')
        if len(invalid_index) != 0:
            noise_label_info[pic_num] = dict({'img_id':img_path, "bboxes_unnormalized":bboxs,"invalid_box_index":invalid_index, "problems":problems})
            count += 1
    return noise_label_info, count

def compare_moto_vehicles(pred_data:dict, gt_data:dict, args:list)->tuple[dict, int]:
    sw, sh, mw, mh, thres_con, thres_iou = args
    count = 0
    noise_label_info = dict()
    if isinstance(gt_data, dict) and 'labeling' in gt_data.keys():
        gt_data = gt_data['labeling']
    for pic_num, gt_img in zip(pred_data.keys(), gt_data):
        invalid_index, problems, bboxs = [], [], []
        pred_box_info = pred_data[pic_num]["bboxes_unnormalized"]
        scores = pred_data[pic_num]["scores"]
        classes = pred_data[pic_num]['classes']
        img_path = pred_data[pic_num]["img_id"]
        visited = np.zeros(len(pred_box_info))
        included_gt = np.zeros(len(gt_img["annotations"])) #To check whether a box of motorcycle is inside a hasRider box(ground truth anno)
        included_pred = np.zeros(len(pred_box_info))#find those box(prediction anno) is inside a hasRider box
        isoccluded = pred_data[pic_num]['occluded']

        #Detect whether a box inside a hasRider box
        for idx, anno in enumerate(gt_img["annotations"]):
            xg, yg, wg, hg = anno['x'], anno['y'], anno['width'], anno['height'] #read the gt_box location
            gt_box = [xg, yg, wg, hg]
            bboxs.append(gt_box)
            if 'hasRider' in anno.keys() and anno['hasRider'] == True and LABEL_MAPPING[anno['class']] in moto_class and wg > sw and hg > sh:
                JustBike, JustRider = False, False
                for idx_oth, anno_oth in enumerate(gt_img['annotations']):
                    if idx_oth == idx: continue;
                    lxo, lyo, wo, ho = anno_oth['x'], anno_oth['y'], anno_oth['width'], anno_oth['height']
                    cxo, cyo = lxo + wo / 2, lyo + ho / 2
                    if anno_oth['class'] == anno['class'] and xg <= cxo <= xg + wg + 5 and yg <= cyo <= yg + hg + 5:
                        JustBike = True
                        included_gt[idx_oth] = 1
                    elif (anno_oth['class'] == 'rider' or anno_oth == 'Rider') and xg <= cxo <= xg + wg:
                        JustRider = True
                        included_gt[idx_oth] = 1
                if not JustBike or not JustRider: #Missing boxs according to the annotation rules
                    invalid_index.append(idx)
                    problems.append('ThreeBody')
                for pred_idx in range(len(pred_box_info)):
                    pred_box = pred_box_info[pred_idx]
                    cxp, cyp, wp, hp = pred_box
                    if visited[pred_idx] == 0 and xg <= cxp <= xg + wg and yg <= cyp <= yg + hg:
                        included_pred[pred_idx] = 1
        
        for idx, anno in enumerate(gt_img["annotations"]):
            ##Finding the FP problem
            xg, yg, wg, hg = anno['x'], anno['y'], anno['width'], anno['height'] #read the gt_box location
            gt_box = [xg, yg, wg, hg]
            if anno['class'] not in LABEL_MAPPING or wg <= sw or hg <= sh or included_gt[idx]:
                continue
            #Check if any center point of pred image is in b area, save in target points set
            target_points = []
            for pred_idx in range(len(pred_box_info)):
                pred_box = pred_box_info[pred_idx]
                cxp, cyp, wp, hp = pred_box
                if visited[pred_idx] == 0 and xg <= cxp <= xg + wg and yg <= cyp <= yg + hg:
                    target_points.append(pred_box)
            #If target points set is empty -> mark as over-annotated problem, continue
            if len(target_points) == 0:
                if LABEL_MAPPING[anno['class']] in moto_class and LABEL_MAPPING[anno['class']] != 4 and anno['occluded'] == 'none':
                    invalid_index.append(idx)
                    problems.append('FP')
                continue
            #Compute the IoU between target points set and b, filter the points that satisfy threshold and choose the biggest IoU as matched point
            target_box_info = []
            for box in target_points:
                gt = [xg, yg, xg + wg, yg + hg]
                cx, cy, w, h = box
                pb = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                iou = _box_iou(pb, gt)
                if iou >= thres_iou:
                    target_box_info.append(box_with_iou(box, iou))
            #No such a matched point -> mark as over-annotated problem, continue
            if len(target_box_info) == 0:
                if LABEL_MAPPING[anno['class']] in moto_class and LABEL_MAPPING[anno['class']] != 4 and anno['occluded'] == 'none': #because of too much undetected pedestrian 
                    invalid_index.append(idx)
                    problems.append('FP')
                continue
            target_box_info.sort(key = lambda x: x.iou, reverse = True)
            match_box = target_box_info[0].box
            match_box_index = pred_box_info.index(match_box)
            # Check the label problem
            if LABEL_MAPPING[anno['class']] in moto_class or LABEL_MAPPING_BLUR[classes[match_box_index]] in moto_class:
                if LABEL_MAPPING_BLUR[anno['class']] != LABEL_MAPPING_BLUR[classes[match_box_index]]:
                    surrounding = False
                    for box_iou in target_box_info:
                        surround_index = pred_box_info.index(box_iou.box)
                        if LABEL_MAPPING[classes[surround_index]] == LABEL_MAPPING[anno['class']]:
                            surrounding = True
                            break
                    if not surrounding:
                        if anno['class'] != 'rider' or classes[match_box_index] != 'moto':
                            invalid_index.append(idx)
                            problems.append('La_' + NAME_ABBRE[anno['class']]+'_' + NAME_ABBRE[classes[match_box_index]])
            #Sign a state bit for the matched picture in pred image as visited
            visited[match_box_index] = 1
        ## Check the FN problem
        unvisited_index = np.where(visited == 0)[0]
        for uidx in unvisited_index:
            cx, cy, wp, hp = pred_box_info[uidx]
            if LABEL_MAPPING[classes[uidx]] not in moto_class or wp <= sw + 3 or hp <= sh + 3 or included_pred[uidx] or isoccluded[uidx][0] > 0.001:
                continue
            if scores[uidx] > thres_con:
                saved_box_form = [cx - wp / 2, cy - hp / 2, wp, hp]
                bboxs.append(saved_box_form)
                invalid_index.append(len(bboxs) - 1)
                problems.append('FN')
        if len(invalid_index) != 0:
            noise_label_info[pic_num] = dict({'img_id':img_path, "bboxes_unnormalized":bboxs,"invalid_box_index":invalid_index, "problems":problems})
            count += 1
    return noise_label_info, count