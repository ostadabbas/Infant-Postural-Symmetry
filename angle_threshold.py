import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

rater_dict = {i: 'Rater ' + str(i) for i in range(1, 11)}
rater_dict.update({11: 'Human Vote', 12: 'Bayesian Aggregate', 13: '2D Ground Truth', 14: '2D Infant Pose Estimate', 
                   15: '3D Pose Ground Truth', 16:'3D Adult Pose Estimate'})
num_raters = len(rater_dict)

print(rater_dict)

data_dir = './data/'
survey_dir = './data/survey_data/asymmetryanalysisAll.xlsx'

twod_gt_dir = './outputs/angles_syrip_2d_gt.csv'
twod_pred_dir = './outputs/angles_syrip_2d_pred.csv'

threed_gt_dir = './outputs/angles_syrip_3d_correction.csv'
threed_pred_dir = './outputs/angles_syrip_3d_pred.csv'

bayesian_angle_dir = './outputs/angle_estimated_annotations.csv'
bayesian_sym_dir = './outputs/symmetry_estimated_annotations.csv'

# Create dataframe with one row per rating, by manually
# reshaping and labelling imported data.

# Import and reshape data.
data = pd.read_excel(survey_dir)
survey = pd.DataFrame(data, columns=['Rater1Symmetry', 'Rater1AngelRange', 'Rater2Symmetry', 'Rater2AngelRange',
                                 'Rater3Symmetry', 'Rater1Ange3Range', 'Rater4Symmetry', 'Rater4AngelRange',
                                 'Rater5Symmetry', 'Rater1Ange5Range', 'Rater6Symmetry', 'Rater6AngelRange',
                                 'Rater7Symmetry', 'Rater1Ange7Range', 'Rater8Symmetry', 'Rater8AngelRange',
                                 'Rater9Symmetry', 'Rater1Ange9Range', 'Rater10Symmetry', 'Rater1Angel0Range']).to_numpy().reshape((-1, 2))

# Create labels.
symmetry = survey[ : , 0]
angle = survey[ : , 1]
image = np.arange(1, 701).repeat(4 * 10)
rater = np.tile(np.arange(1, 11), 2800)
part = np.tile(np.array(['upper-arm', 'lower-arm', 'upper-leg', 'lower-leg']).repeat(10), 700)
test = [0] * 2 * 4000 + [1] * 4000 + [0] * 4 * 4000


# Create dataframe.
df = pd.DataFrame({'image': image, 'rater': rater, 'part': part, 'symmetry': symmetry, 'angle': angle, 'test': test})

# Define median rater.

median_df = df.groupby(['image', 'part'], as_index = False).median()
median_df[['symmetry', 'angle']] = median_df[['symmetry', 'angle']].apply(np.ceil)
median_df['rater'] = [11] * 2800

# Import Bayesian rater.


# Import and reshape data.
bayesian_sym = pd.read_csv(bayesian_sym_dir).to_numpy()[ : , 1].astype('int')
bayesian_angles = pd.read_csv(bayesian_angle_dir).to_numpy()[ : , 4].astype('int')

# Create labels.
image = np.arange(1, 701).repeat(4)
rater = [12] * 2800
part = np.tile(np.array(['upper-arm', 'lower-arm', 'upper-leg', 'lower-leg']), 700)
test = [0] * 2 * 400 + [1] * 400 + [0] * 4 * 400

# Create dataframe.
bayesian_df = pd.DataFrame({'image': image, 'rater': rater, 'part': part,
                            'angle': bayesian_angles, 'symmetry': bayesian_sym,
                            'test': test})


# Import and process pose-based 2D and 3D angles into dataframe with same structure.


# Import and reshape data.
twod_gt = pd.read_csv(twod_gt_dir).to_numpy().reshape((-1))
twod_pred = pd.read_csv(twod_pred_dir).to_numpy().reshape((-1))

threed_gt = pd.read_csv(threed_gt_dir).to_numpy().reshape((-1))
threed_pred = pd.read_csv(threed_pred_dir).to_numpy().reshape((-1))

# Create labels.
image = np.arange(1, 701).repeat(4)

# Create dataframe.
twod_gt_df = pd.DataFrame({'image': image, 'rater': [13] * 2800, 'test': test, 'degree': twod_gt})
twod_pred_df = pd.DataFrame({'image': image, 'rater': [14] * 2800, 'test': test, 'degree': twod_pred})

threed_gt_df = pd.DataFrame({'image': image, 'rater': [16] * 2800, 'test': test, 'degree': threed_gt})
threed_pred_df = pd.DataFrame({'image': image, 'rater': [18] * 2800, 'test': test, 'degree': threed_pred})

# Add angle categories.
def angle_category(theta, thresholds = [30, 60]):
    for i, threshold in enumerate(thresholds):
        if theta < threshold:
            return i + 1
    return len(thresholds) + 1


# twod_gt_df['angle'] = twod_gt_df['degree'].apply(angle_category)
# twod_gt_df['symmetry'] = twod_gt_df['degree'].apply(lambda x: 2 - angle_category(x, [39]))

# twod_pred_df['angle'] = twod_pred_df['degree'].apply(angle_category)
# twod_pred_df['symmetry'] = twod_pred_df['degree'].apply(lambda x: 2 - angle_category(x, [42]))

# twodad_pred_df['angle'] = twodad_pred_df['degree'].apply(angle_category)
# twodad_pred_df['symmetry'] = twodad_pred_df['degree'].apply(lambda x: 2 - angle_category(x, [44]))

# threed_gt_df['angle'] = threed_gt_df['degree'].apply(angle_category)
# threed_gt_df['symmetry'] = threed_gt_df['degree'].apply(lambda x: 2 - angle_category(x, [28]))

# threed_pred_df['angle'] = threed_pred_df['degree'].apply(angle_category)
# threed_pred_df['symmetry'] = threed_pred_df['degree'].apply(lambda x: 2 - angle_category(x, [28]))

# threedft_pred_df['angle'] = threedft_pred_df['degree'].apply(angle_category)
# threedft_pred_df['symmetry'] = threedft_pred_df['degree'].apply(lambda x: 2 - angle_category(x, [23]))

# threedad_pred_df['angle'] = threedad_pred_df['degree'].apply(angle_category)
# threedad_pred_df['symmetry'] = threedad_pred_df['degree'].apply(lambda x: 2 - angle_category(x, [18]))

pose_df = pd.concat([twod_gt_df, twod_pred_df, threed_gt_df, threed_pred_df], ignore_index = True)

# Rater-means of angle and symmetry, per image and part.


rater_means = df.groupby(['image']) \
    .agg({'symmetry': np.mean, 'angle': np.mean, 'test': np.mean}).reset_index()

rater_stds = df.groupby(['image', 'part']) \
    .agg({'symmetry': np.std, 'angle': np.std, 'test': np.mean}).reset_index()


# Combine raters, median rater, and two-dimensional ground truth.

joint_df = pd.concat([df, median_df, bayesian_df, pose_df.drop(columns = ['degree'])],
                      ignore_index = True, sort=False)

joint_df = pd.concat([df, median_df, bayesian_df, pose_df.drop(columns = ['degree'])],
                      ignore_index = True, sort=False)

joint_angle_df = joint_df.drop(columns = ['symmetry'])
joint_symmetry_df = joint_df.drop(columns = ['angle'])


# Combined graph for publication. Highly inefficient. 

max_angle, steps_per_degree = 90, 10

median_irr = np.empty((4, max_angle * steps_per_degree))
mean_irr = np.empty((4, max_angle * steps_per_degree))
bayesian_irr = np.empty((4, max_angle * steps_per_degree))

median_symmetry = median_df['symmetry'].astype('int')
bayesian_symmetry = bayesian_df['symmetry'].astype('int')

sym_dic = {
    0: [twod_gt_df, '2D Ground Truth'],
    1: [twod_pred_df, '2D Infant Pose Prediction'],
    2: [threed_gt_df, '3D Ground Truth'],
    3: [threed_pred_df, '3D Infant Pose Prediction'],
}

fig, ax = plt.subplots(1, 4, figsize=(10, 4))

for i in range(4):
    for threshold in range(max_angle * steps_per_degree):
        symmetry = sym_dic[i][0]['degree'].apply(lambda x: 2 - angle_category(x, [threshold / steps_per_degree]))
        median_irr[i, threshold] = cohen_kappa_score(median_symmetry.to_numpy(), symmetry.to_numpy())
        bayesian_irr[i, threshold] = cohen_kappa_score(bayesian_symmetry.to_numpy(), symmetry.to_numpy())

        individual_irrs = np.empty(10)
        for j in range(10):
            individual_irrs[j] = cohen_kappa_score(df[df['rater'] == j + 1]['symmetry'].to_numpy(), 
                                                   symmetry.to_numpy())

        mean_irr[i, threshold] = np.mean(individual_irrs)
    
    print('Optimal mean, median, and Bayesian angle for ' + sym_dic[i][1] + ':', \
            np.argmax(mean_irr[i]) / steps_per_degree, \
            np.argmax(median_irr[i]) / steps_per_degree, \
            np.argmax(bayesian_irr[i]) / steps_per_degree)

    sns.lineplot(ax = ax[i], x = range(0, max_angle), y = median_irr[i, ::steps_per_degree], label = 'Median rater')
    sns.lineplot(ax = ax[i], x = range(0, max_angle), y = mean_irr[i , ::steps_per_degree], label = 'Mean across raters')
    sns.lineplot(ax = ax[i], x = range(0, max_angle), y = bayesian_irr[i, ::steps_per_degree], label = 'Bayesian rater')
    ax[i].get_legend().remove()
    ax[i].set_xlabel('Threshold Angle\n' + sym_dic[i][1])
    ax[i].set_ylim([-0.02, 0.4])

ax[0].set_ylabel('Cohen kappa')


handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol = 3, loc = 'lower center') #, bbox_to_anchor = (0.35, -0.05))
fig.subplots_adjust(bottom = 0.25)
plt.suptitle('Pose Prediction Agreement with Human Assessment, by Threshold Angle')
plt.show()