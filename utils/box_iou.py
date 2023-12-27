class box_with_iou():
    def __init__(self, bb, iou):
        self.box = bb
        self.iou = iou

def _box_iou(bb, gt):
    '''
    bb:[x1,y1,x2,y2]
    gt:[x1,y1,x2,y2]
    '''
    bi = [max(bb[0], gt[0]), max(bb[1], gt[1]),
            min(bb[2], gt[2]), min(bb[3], gt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    ov = 0
    if iw > 0 and ih > 0:
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) - iw * ih
        ov = iw * ih / ua
    return ov