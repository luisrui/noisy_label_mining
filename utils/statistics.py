from .label_mapping import LABEL_MAPPING
import matplotlib.pyplot as plt
import numpy as np

def classify(noise_label_info : dict) -> tuple[int, int, int]:#do statistics on the number of different noisy problems
    fp, fn, lab = 0, 0, 0
    for pic_num in noise_label_info.keys():
        for pblm in noise_label_info[pic_num]['problems']:
            if pblm == 'FP': fp += 1
            elif pblm == 'FN': fn += 1
            if 'La' in pblm:
                lab += 1
    return fp, fn, lab

def merge_noisy_info_files(*json_files): #Merge different json file with different classes together
    '''
        json_files include the noisy_label_info within different categories(about three files)
    '''
    max_number = max([int(list(json_file.keys())[-1]) for json_file in json_files]) #take the largest number of pic number
    noisy_label_info_whole = dict()

    for id in range(max_number):
        for noisy_label_file in json_files:
            if str(id) in noisy_label_file.keys() and str(id) not in noisy_label_info_whole.keys():#Initialization stage
                noisy_label_info_whole.update({str(id):noisy_label_file[str(id)]})
            if str(id) in noisy_label_file.keys() and str(id) in noisy_label_info_whole.keys():#Update stage
                new_boxes = noisy_label_file[str(id)]['bboxes_unnormalized']
                for i in range(len(new_boxes)):
                    if new_boxes[i] in noisy_label_info_whole[str(id)]['bboxes_unnormalized']: #new box exist in current boxes
                        current_index = noisy_label_info_whole[str(id)]['bboxes_unnormalized'].index(new_boxes[i])
                        if i in noisy_label_file[str(id)]['invalid_box_index'] and current_index not in noisy_label_info_whole[str(id)]['invalid_box_index']:
                            noisy_label_info_whole[str(id)]['invalid_box_index'].append(current_index)
                            problem_index = noisy_label_file[str(id)]['invalid_box_index'].index(i)
                            new_problem = noisy_label_file[str(id)]['problems'][problem_index]
                            noisy_label_info_whole[str(id)]['problems'].append(new_problem)
                    else:# new box no exist in current boxes
                        noisy_label_info_whole[str(id)]['bboxes_unnormalized'].append(new_boxes[i])
                        noisy_label_info_whole[str(id)]['invalid_index'].append(len(noisy_label_info_whole[str(id)]['bboxes_unnormalized'])-1)
                        problem_index = noisy_label_file[str(id)]['invalid_box_index'].index(i)
                        new_problem = noisy_label_file[str(id)]['problems'][problem_index]
                        noisy_label_info_whole[str(id)]['problems'].append(new_problem)
    return noisy_label_info_whole

def analysis_label_noise(noise_label_info: dict):
    # vehicle_class = [0,1,2,8,9]
    # moto_class = [3, 4, 5]
    # cone_class = [6, 7]
    def label_in_problems(problems):
        for pblm in problems:
            if 'La' in pblm:
                return True
        return False
    label_noise = dict(filter(lambda p: label_in_problems(p[1]['problems']), noise_label_info.items()))
    label_wrong_class = []
    #mapping_class = []
    for pic_num in list(label_noise.keys()):
        for pblm in label_noise[pic_num]['problems']:
            if pblm != 'FN' and pblm != 'FP' and pblm != 'ThreeBody':
                label = pblm[3:]
                _ = label.index('_')
                gt_la, pd_la = label[:_], label[_+1:]
                label_wrong_class.append(gt_la+'->'+pd_la)
                #mapping_class.append(str(LABEL_MAPPING[gt_la]) + '_'+ str(LABEL_MAPPING[pd_la]))
    single_mistake = np.unique(label_wrong_class)
    num_mistake = []
    for mistake in single_mistake:
        num_mistake.append(label_wrong_class.count(mistake))
    fig, ax = plt.subplots(figsize =(16, 9))
    ax.barh(single_mistake, num_mistake)
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    
    # Show top values
    ax.invert_yaxis()
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    # Show Plot
    plt.show()
    return label_noise
