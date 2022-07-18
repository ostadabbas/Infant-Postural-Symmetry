import json
import matplotlib.pyplot as plt
import numpy as np
from sympy import Point, Line
import csv


# define angle threshold
thres = 42

def mirrorImage( a, b, c, x1, y1):
    temp = -2 * (a * x1 + b * y1 + c) /(a * a + b * b)
    x = temp * a + x1
    y = temp * b + y1 
    return [x, y]

def midpoint(x1, y1, x2, y2):
    return [(x1 + x2)/2, (y1 + y2)/2]

# load 2d pose pred for syrip real images

with open('./data/SyRIP_2d_pred/keypoints_validate_infant_results_0.json', 'r') as f1:
    pred = json.load(f1)


AngleM2D = np.empty([700, 4])
RatorM2D = np.empty([700, 8])

for i in range(len(pred)):
    img_id = pred[i]['image_id']
  
    kpts = pred[i]['keypoints']

    # 2d symmetry measurement
    # upper body basis
    p1, p2 = Point(kpts[6 * 3 + 0], kpts[6 * 3 + 1]), Point(kpts[5 * 3 + 0], kpts[5 * 3 + 1])
    if p1[0] == p2[0] and p1[1] == p2[1]:
        p1 = Point(p1[0]+0.1, p1[1]+0.1)

    l1 = Line(p1, p2)

    mid_p = midpoint(kpts[6 * 3 + 0], kpts[6 * 3 + 1], kpts[5 * 3 + 0], kpts[5 * 3 + 1])
    p3 = Point(mid_p[0], mid_p[1])

    l2 = l1.perpendicular_line(p3)
    a, b, c = l2.coefficients

    # shoulder    mirror right to left
    m_sh = mirrorImage(a, b, c, kpts[6 * 3 + 0], kpts[6 * 3 + 1])
    m_elb = mirrorImage(a, b, c, kpts[8 * 3 + 0], kpts[8 * 3 + 1])
    m_arm1 = np.asarray([a_i - b_i for a_i, b_i in zip(m_elb, m_sh)]).astype('float')

    arm1 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[(7 * 3 + 0): (7 * 3 + 2)], kpts[(5 * 3 + 0): (5 * 3 + 2)])]).astype('float')

    m_arm1_ = m_arm1 / np.linalg.norm(m_arm1)
    arm1_ = arm1 / np.linalg.norm(arm1)
    dot_product = np.dot(m_arm1_, arm1_)
    arm1_ang = np.rad2deg(np.arccos(dot_product))
            
    AngleM2D[i,0] = arm1_ang
    if arm1_ang <= thres:
        RatorM2D[i,0] = 1
    else:
        RatorM2D[i,0] = 0

    if arm1_ang < 30:
        RatorM2D[i,4] = 1
    elif arm1_ang >= 30 and arm1_ang < 60:
        RatorM2D[i,4] = 2
    elif arm1_ang >= 60:
        RatorM2D[i,4] = 3
    else:
        RatorM2D[i,4] = 0

    # elbow
    m_wrs = mirrorImage(a, b, c, kpts[10 * 3 + 0], kpts[10 * 3 + 1])
    m_arm2 = np.asarray([a_i - b_i for a_i, b_i in zip(m_wrs, m_elb)]).astype('float')

    arm2 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[(9 * 3 + 0): (9 * 3 + 2)], kpts[(7 * 3 + 0): (7 * 3 + 2)])]).astype('float')

    m_arm2_ = m_arm2 / np.linalg.norm(m_arm2)
    arm2_ = arm2 / np.linalg.norm(arm2)
    dot_product = np.dot(m_arm2_, arm2_)
    arm2_ang = np.rad2deg(np.arccos(dot_product))


    AngleM2D[i,1] = arm2_ang
    if arm2_ang <= thres:
        RatorM2D[i,1] = 1
    else:
        RatorM2D[i,1] = 0

    if arm2_ang < 30:
        RatorM2D[i,5] = 1
    elif arm2_ang >= 30 and arm2_ang < 60:
        RatorM2D[i,5] = 2
    elif arm2_ang >= 60:
        RatorM2D[i,5] = 3
    else:
        RatorM2D[i,5] = 0

    # lower body basis
    p1, p2 = Point(kpts[12 * 3 + 0], kpts[12 * 3 + 1]), Point(kpts[11 * 3 + 0], kpts[11 * 3 + 1])

    if p1[0] == p2[0] and p1[1] == p2[1]:
        p1 = Point(p1[0]+0.1, p1[1]+0.1)

    l1 = Line(p1, p2)

    mid_p = midpoint(kpts[12 * 3 + 0], kpts[12 * 3 + 1], kpts[11 * 3 + 0], kpts[11 * 3 + 1])
    p3 = Point(mid_p[0], mid_p[1])

    l2 = l1.perpendicular_line(p3)
    a, b, c = l2.coefficients

    # hip
    m_hip = mirrorImage(a, b, c, kpts[12 * 3 + 0], kpts[12 * 3 + 1])
    m_kne = mirrorImage(a, b, c, kpts[14 * 3 + 0], kpts[14 * 3 + 1])
    m_leg1 = np.asarray([a_i - b_i for a_i, b_i in zip(m_kne, m_hip)]).astype('float')

    leg1 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[(13 * 3 + 0): (13 * 3 + 2)], kpts[(11 * 3 + 0): (11 * 3 + 2)])]).astype('float')

    m_leg1_ = m_leg1 / np.linalg.norm(m_leg1)
    leg1_ = leg1 / np.linalg.norm(leg1)
    dot_product = np.dot(m_leg1_, leg1_)
    leg1_ang = np.rad2deg(np.arccos(dot_product))

    AngleM2D[i,2] = leg1_ang
    if leg1_ang <= thres:
        RatorM2D[i,2] = 1
    else:
        RatorM2D[i,2] = 0

    if leg1_ang < 30:
        RatorM2D[i,6] = 1
    elif leg1_ang >= 30 and leg1_ang < 60:
        RatorM2D[i,6] = 2
    elif leg1_ang >= 60:
        RatorM2D[i,6] = 3
    else:
        RatorM2D[i,6] = 0

    # knee
    m_ank = mirrorImage(a, b, c, kpts[16 * 3 + 0], kpts[16 * 3 + 1])
    m_leg2 =np.asarray([a_i - b_i for a_i, b_i in zip(m_ank, m_kne)]).astype('float')

    leg2 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[(15 * 3 + 0): (15 * 3 + 2)], kpts[(13 * 3 + 0): (13 * 3 + 2)])]).astype('float')
    if np.linalg.norm(leg2) == 0.0:
        leg2[0] = leg2[0]+0.1
    m_leg2_ = m_leg2 / np.linalg.norm(m_leg2)
    leg2_ = leg2 / np.linalg.norm(leg2)
    dot_product = np.dot(m_leg2_, leg2_)
    leg2_ang = np.rad2deg(np.arccos(dot_product))


    AngleM2D[i,3] = leg2_ang
    if leg1_ang <= thres:
        RatorM2D[i,3] = 1
    else:
        RatorM2D[i,3] = 0

    if leg2_ang < 30:
        RatorM2D[i,7] = 1
    elif leg2_ang >= 30 and leg2_ang < 60:
        RatorM2D[i,7] = 2
    elif leg2_ang >= 60:
        RatorM2D[i,7] = 3
    else:
        RatorM2D[i,7] = 0


header = ['UA_ang', 'LA_ang', 'UL_ang', 'LL_ang']
with open('./outputs/angles_syrip_2d_pred.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(AngleM2D)

header = ['UA_sym', 'LA_sym', 'UL_sym', 'LL_sym', 'UA_ang', 'LA_ang', 'UL_ang', 'LL_ang']
with open('./outputs/rating_syrip_2d_pred_thres42.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(RatorM2D)






