import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = pd.read_csv('features.csv')
sample_submission = pd.read_csv('submission_sample.csv')
region_info = pd.read_csv('region_info.csv')

def find_closest_point(points, target_point):
    points = np.array(points)
    target_point = np.array(target_point)
    distances = np.linalg.norm(points - target_point, axis=1)
    closest_index = np.argmin(distances)
    return closest_index

def preprocess_train_test(df):
    df['features_id'] = df[['lat', 'lon']].apply(lambda x: find_closest_point(features[['lat', 'lon']], [x['lat'], x['lon']]), axis=1)
    df = df.merge(features.drop(columns=['lat', 'lon']), left_on='features_id', right_index=True, how='left')
    df = df.merge(region_info, how='left')
    return df

train = preprocess_train_test(train)
test = preprocess_train_test(test)
train.drop(columns=['region','features_id','population', 'median_reg_score', 'max_reg_score', 'min_reg_score'], inplace=True)
train.dropna(inplace=True)
test.drop(columns=['region','features_id','population', 'median_reg_score', 'max_reg_score', 'min_reg_score'], inplace=True)

opt_features = ['region_population', '42', '86', '178', '197', '272', '317',  '91', '93', '96', '97', '98', '100', '185', '188', '189', '190', '190', '191', '192', '193', '194', '195', '196', '198', '199', '200',  '204', '205', '206', '207', '208', '209', '210', '211', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225',  '226', '227', '228', '229',  '230', '231', '232', '233', '234']
X_train = train[opt_features]
y_train = train['score']
X_test = test[opt_features]
scl = StandardScaler()
X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)
model = joblib.load('model.pkl') 
pred = model.predict(X_test)

sample_submission['score'] = pred
sample_submission.to_csv('submission.csv', index=False)