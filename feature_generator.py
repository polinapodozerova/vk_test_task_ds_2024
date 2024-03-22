from dadata import Dadata
import pandas as pd
import numpy as np
import re

token = "cc95c6db64a799c3f46c74b4c34f98ef5662b0e7"
dadata = Dadata(token)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = pd.read_csv('features.csv')
coord_and_regions = pd.read_csv('coordinates_to_regions.csv')
population = pd.read_csv('population.csv', names=['year', 'region', 'population'])
population = population[population['year'] == 2019]

missing_region_coord = []
def coord_to_region(lat, lon):
    try:
        region_name = dadata.geolocate(name="address", lat=lat, lon=lon)[0]['data']['region_with_type']
    except:
        missing_region_coord.append([lat, lon])
        region_name = ''
    return region_name

def clear_region_name(name):
    replacements = {
                    'Чувашская республика - ' : '',
                    'Кемеровская область - Кузбасс' : 'Кемеровская область',
                    'Ханты-Мансийский Автономный округ - Югра' : 'Ханты-Мансийский автономный округ',
                    'Удмуртская Респ' : 'Удмуртия',
                    'Респ Дагестан' : 'Дагестан',
                    'Респ Татарстан' : 'Татарстан',
                    'Респ Башкортостан' : 'Башкортостан',
                    'г ' : '',
                    'Респ' : 'Республика', 
                    ' обл' : ' область'}
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    try:
        return pattern.sub(lambda match: replacements[match.group(0)], name)
    except:
        print(name)

def add_reg_information(df):
    df['region'] = df[['lat', 'lon']].apply(lambda x: coord_to_region(x['lat'], x['lon']), axis=1)
    return df

def change_reg_information(df):
    df['region'] = df['region'].apply(lambda x: clear_region_name(x))
    return df

def extract_population(x):
    t = population[population['region']==x]['population'].values
    if len(t) > 0:
        return int(t[0])
    else: return int(population['population'].median())

def add_population_information(df):
    df['region_population'] = df['region'].apply(func=lambda x:extract_population(x))
    return df

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
    

# all_coordinates = pd.concat([train[['lat', 'lon']], test[['lat', 'lon']]], axis=0)
# coord_and_regions = add_reg_information(all_coordinates)

# coord_and_regions.loc[225, 'region'] = 'Саратовская область'
# coord_and_regions.loc[1884, 'region'] = 'Московская область'
# coord_and_regions.to_csv('coordinates_to_regions.csv', index=False)

region_info = add_population_information(change_reg_information(coord_and_regions))
temp_info = train.merge(region_info, how='left').groupby(by='region', as_index=False).agg(
        population=('region_population', 'median'),
        median_reg_score=('score', 'median'),
        max_reg_score=('score', 'max'),
        min_reg_score=('score', 'min')
    ).drop(0, axis=0)

region_info = region_info.merge(temp_info, how='left')
region_info.to_csv('region_info.csv', index=False)