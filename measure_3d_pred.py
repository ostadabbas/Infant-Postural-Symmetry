import json
import matplotlib.pyplot as plt
import numpy as np
from sympy import Point3D, Line
import csv
import matplotlib.pyplot as plt


# define angle threshold
thres = 28

# 3d prediction order: [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist

# For SPIN predicted 14 joint order: r_ankle, r_knee, r_hip, l_hip, l_knee, l_ankle, r_wrist, r_elbow, r_shoulder, l_shoulder, l_elbow, l_wrist, neck, head

kpts_pair = [[9, 9, 10, 8, 7, 3, 3, 4, 2, 1], [8, 10, 11, 7, 6, 2, 4, 5, 1, 0]]


def mirrorImage(a, b, c, d, x1, y1, z1):
      
    k =(-a * x1-b * y1-c * z1-d)/float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2-x1
    y3 = 2 * y2-y1
    z3 = 2 * z2-z1
    return [x3, y3, z3]


imgs = np.load('./data/SyRIP_3d_pred/output_imgnames.npy')


pred = np.load('./data/SyRIP_3d_pred/output_pose_3D.npy')


# load image names mapping 'new_names' and 'ori_names'
name_map = np.load('./data/SyRIP_data/img_name700_map.npy')


AngleM3D = np.empty([700, 4])
RaterM3D = np.empty([700, 8])
for i in range(100): #name_map.shape[0]):
    new_name = 'GB/' + name_map[i+200, 0]
    
    idx, = np.where(imgs == new_name)
    
    kpts = pred[idx[0], :, :]
    
    # 3d symmetry measurement
    # upper body basis
    p1, p2 = [kpts[9][0], kpts[9][1], kpts[9][2]], [kpts[8][0], kpts[8][1], kpts[8][2]]

    p3 = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3) 
    n_vec = (p1 - p2) / np.sqrt(np.sum((p1 - p2)**2))  
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    d = -a * p3[0] - b * p3[1] - c * p3[2]

    # shoulder    mirror right to left
    m_sh = mirrorImage(a, b, c, d, kpts[8][0], kpts[8][1], kpts[8][2])
    m_elb = mirrorImage(a, b, c, d, kpts[7][0], kpts[7][1], kpts[7][2])
    m_arm1 = np.asarray([a_i - b_i for a_i, b_i in zip(m_elb, m_sh)]).astype('float')

    arm1 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[10], kpts[9])]).astype('float')

    m_arm1_ = m_arm1 / np.linalg.norm(m_arm1)
    arm1_ = arm1 / np.linalg.norm(arm1)
    dot_product = np.dot(m_arm1_, arm1_)
    arm1_ang = np.rad2deg(np.arccos(dot_product))

    AngleM3D[i,0] = arm1_ang
    if arm1_ang <= thres:
        RaterM3D[i,0] = 1
    else:
        RaterM3D[i,0] = 0

    if arm1_ang < 30:
        RaterM3D[i,4] = 1
    elif arm1_ang >= 30 and arm1_ang < 60:
        RaterM3D[i,4] = 2
    elif arm1_ang >= 60:
        RaterM3D[i,4] = 3
    else:
        RaterM3D[i,4] = 0

    # elbow
    m_wrs = mirrorImage(a, b, c, d, kpts[6][0], kpts[6][1], kpts[6][2])
    m_arm2 = np.asarray([a_i - b_i for a_i, b_i in zip(m_wrs, m_elb)]).astype('float')

    arm2 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[11], kpts[10])]).astype('float')

    m_arm2_ = m_arm2 / np.linalg.norm(m_arm2)
    arm2_ = arm2 / np.linalg.norm(arm2)
    dot_product = np.dot(m_arm2_, arm2_)
    arm2_ang = np.rad2deg(np.arccos(dot_product))

    AngleM3D[i,1] = arm2_ang
    if arm2_ang <= thres:
        RaterM3D[i,1] = 1
    else:
        RaterM3D[i,1] = 0

    if arm2_ang < 30:
        RaterM3D[i,5] = 1
    elif arm2_ang >= 30 and arm2_ang < 60:
        RaterM3D[i,5] = 2
    elif arm2_ang >= 60:
        RaterM3D[i,5] = 3
    else:
        RaterM3D[i,5] = 0


    # lower body basis
    p1, p2 = [kpts[3][0], kpts[3][1], kpts[3][2]], [kpts[2][0], kpts[2][1], kpts[2][2]]

    p3 = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3) 
    n_vec = (p1 - p2) / np.sqrt(np.sum((p1 - p2)**2))  
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    d = -a * p3[0] - b * p3[1] - c * p3[2]

    # hip
    m_hip = mirrorImage(a, b, c, d, kpts[2][0], kpts[2][1], kpts[2][2])
    m_kne = mirrorImage(a, b, c, d, kpts[1][0], kpts[1][1], kpts[1][2])
    m_leg1 = np.asarray([a_i - b_i for a_i, b_i in zip(m_kne, m_hip)]).astype('float')

    leg1 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[4], kpts[3])]).astype('float')

    m_leg1_ = m_leg1 / np.linalg.norm(m_leg1)
    leg1_ = leg1 / np.linalg.norm(leg1)
    dot_product = np.dot(m_leg1_, leg1_)
    leg1_ang = np.rad2deg(np.arccos(dot_product))

    AngleM3D[i,2] = leg1_ang
    if leg1_ang <= thres:
        RaterM3D[i,2] = 1
    else:
        RaterM3D[i,2] = 0

    if leg1_ang < 30:
        RaterM3D[i,6] = 1
    elif leg1_ang >= 30 and leg1_ang < 60:
        RaterM3D[i,6] = 2
    elif leg1_ang >= 60:
        RaterM3D[i,6] = 3
    else:
        RaterM3D[i,6] = 0

    # knee
    m_ank = mirrorImage(a, b, c, d, kpts[0][0], kpts[0][1], kpts[0][2])
    m_leg2 =np.asarray([a_i - b_i for a_i, b_i in zip(m_ank, m_kne)]).astype('float')

    leg2 = np.asarray([a_i - b_i for a_i, b_i in zip(kpts[5], kpts[4])]).astype('float')

    m_leg2_ = m_leg2 / np.linalg.norm(m_leg2)
    leg2_ = leg2 / np.linalg.norm(leg2)
    dot_product = np.dot(m_leg2_, leg2_)
    leg2_ang = np.rad2deg(np.arccos(dot_product))
    
    AngleM3D[i,3] = leg2_ang
    if leg1_ang <= thres:
        RaterM3D[i,3] = 1
    else:
        RaterM3D[i,3] = 0

    if leg2_ang < 30:
        RaterM3D[i,7] = 1
    elif leg2_ang >= 30 and leg2_ang < 60:
        RaterM3D[i,7] = 2
    elif leg2_ang >= 60:
        RaterM3D[i,7] = 3
    else:
        RaterM3D[i,7] = 0


header = ['UA_ang', 'LA_ang', 'UL_ang', 'LL_ang']
with open('./output/angles_syrip_3d_pred.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(AngleM3D)

header = ['UA_sym', 'LA_sym', 'UL_sym', 'LL_sym', 'UA_ang', 'LA_ang', 'UL_ang', 'LL_ang']
with open('./outputs/rating_syrip_3d_pred_thres28.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(RaterM3D)

