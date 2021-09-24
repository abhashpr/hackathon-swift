import concurrent
from functools import partial
import numpy as np
import pandas as pd

from bisect import bisect_left
from geopy.distance import geodesic
from collections import OrderedDict
from operator import getitem
import numpy as np
import pickle5 as pickle
import time
from faker import Faker
import json


ADD_GEO_MAP = json.load(open('add_to_geo_map.json' , 'r'))
FOREX = json.load(open("forex.json", "rb"))

# ADD_GEO_MAP = pickle.load(open('add_to_geo_map.pkl', 'rb'))
# FOREX = pickle.load(open("exchange_rates.pkl", "rb"))

# senders_count_dict = OrderedDict(
#                         sorted(pickle.load(open('models/senders_counts.pkl', 'rb')).items(), 
#                                                     key = lambda x: x[1]))
# receivers_count_dict = OrderedDict(
#                         sorted(pickle.load(open('models/receivers_counts.pkl', 'rb')).items(),
#                                                     key = lambda x : x[1]))


def geo_loc(point = None, address = None):
    
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="map.app")
    
    if address is None:
        try:
            location = geolocator.reverse(point)
            return (location.address, (location.latitude, location.longitude), 
                    location.raw)
        except:
            return 'Invalid geo-coordinates !!'
    else:
        try:
            location = geolocator.geocode(address)
            return (location.address, (location.latitude, location.longitude), 
                    location.raw)
        except:
            return 'Invalid Address !!'

def get_lat_long(address):
    try:
        point_ = ADD_GEO_MAP[address.strip()]
        return point_['latitude'], point_['longitude']
    except:
        point_ = geo_loc(address = address.strip())
        return point_[2]['lat'], point_[2]['lon']
    else:
        return 0, 0
    

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def get_octa(point):
    if float(point['latitude']) < 0:
        if float(point['longitude']) < 0:
            return int(float(point['longitude']) / -90)
        else:
            return int(float(point['longitude']) / 90) + 2
    else:
        if float(point['longitude']) < 0:
            return int(float(point['longitude']) / -90) + 4
        else:
            return int(float(point['longitude']) / 90) + 6


def reorganize_geocodes(geocodes_dictionary):
    from collections import OrderedDict
    coordinates = OrderedDict()
    
    coordinates['address'] = np.array(list(geocodes_dictionary.keys()))
    coordinates['latitude'] = np.array([row['latitude'] for name, row in geocodes_dictionary.items()])
    coordinates['longitude'] = np.array([row['longitude'] for name, row in geocodes_dictionary.items()])
    coordinates['category'] = np.array([row['category'] for name, row in geocodes_dictionary.items()])
    coordinates['distance'] = np.array([row['distance'] for name, row in geocodes_dictionary.items()])
    return coordinates

def augment_geocode_dict(geocodes):
    
    from geopy.distance import geodesic
    from collections import OrderedDict
    from operator import getitem
    
    for key in geocodes.keys():
        geocodes[key]['category'] = get_octa(geocodes[key])
        geocodes[key]['distance'] = float(geodesic((0, 0), (geocodes[key]['latitude'], geocodes[key]['longitude'])).km)
    geocodes = OrderedDict(sorted(geocodes.items(), key=lambda x: getitem(x[1], 'distance')))
    return geocodes


COORDS = reorganize_geocodes(augment_geocode_dict(ADD_GEO_MAP))


def get_nearest_loc(point, geocodes = None):
    from geopy.distance import geodesic
    if geocodes is None:
        return geo_loc(point)
    else:
        try:
            point_category = get_octa(point)
            geolist = geocodes['distance'][geocodes['category'] == point_category]
            distance = float(geodesic((0, 0), (point['latitude'], point['longitude'])).km)
            closest = np.where(geocodes['distance'] == take_closest(geolist, distance))

            closest_category = geocodes['category'][closest][0]
            closest_distance = geocodes['distance'][closest][0]
            # print(closest_category, point_category, abs(closest_distance - distance))
            if (closest_category != point_category) or abs(closest_distance - distance) > 50:
                return geo_loc(point)
            else:
                # return point_category, closest_category, distance, closest_distance, geocodes['address'][closest[0][0]]
                return geocodes['address'][closest[0][0]]

        except:
            return ''

def get_geocodes(address):
    try:
        # return coords_dict[address]
        location = COORDS[address]
        return [location['latitude'], location['longitude']]
    except:
        return ''

    
def xfm_data_for_modelling(df):
    data = pd.DataFrame()
    data['33b_cur'] = df['33b_cur']
    data['usd_amt'] = df[['33b_orig_ord_amt', '33b_cur']].apply(lambda x: x[0] * FOREX[x[1]], axis = 1)
    df['usd_amt'] = df[['33b_orig_ord_amt', '33b_cur']].apply(lambda x: x[0] * FOREX[x[1]], axis = 1)
    conditions = [
                    ~df['50a_payor_lon'].isnull(), 
                    ~df['50f_payor_add_ln_2'].isnull(), 
                    ~df['50k_payor_add_ln_2'].isnull()]
    choices = ['A', 'F', 'K']
    data['src_xfrr_type'] = np.select(conditions, choices, default = "")
    df['src_xfrr_type'] = np.select(conditions, choices, default = "")
    conditions = [
                    ~df['50f_payor_add_lon'].isnull(),
                    ~df["50k_payor_add_lon"].isnull()
                 ]
    choices = [df["50f_payor_add_lon"], df["50k_payor_add_lon"]]
    data['src_lon'] = np.select(conditions, choices, default=0)
    df['src_lon'] = np.select(conditions, choices, default=0)
    
    conditions = [
                    ~df["50f_payor_add_lat"].isnull(),
                    ~df["50k_payor_add_lat"].isnull()]
    choices = [df["50f_payor_add_lat"], df["50k_payor_add_lat"]]
    
    # data['52a_sender'] = df['52a_sender']
    # data['57a_receiver'] = df['57a_receiver']
    data['send_rec_pair'] = df['52a_sender'] + df['57a_receiver']
    
    data["src_lat"] = np.select(conditions, choices, default=0)
    df["src_lat"] = np.select(conditions, choices, default=0)
    data[["target_lat", "target_lon"]] = df.loc[:, ["59f_ben_add_lat", "59f_ben_add_lon"]]
    df[["target_lat", "target_lon"]] = df.loc[:, ["59f_ben_add_lat", "59f_ben_add_lon"]]
    data["charge_dtls"] = df.loc[:, "71A_chg_dtls"]
    df["charge_dtls"] = df.loc[:, "71A_chg_dtls"]
    data[["charge_dtls_cur", "charge_dtls_amt"]] = df.loc[:, ["71f_chg_dtls_cur", "71f_chg_dtls_amt"]]
    df[["charge_dtls_cur", "charge_dtls_amt"]] = df.loc[:, ["71f_chg_dtls_cur", "71f_chg_dtls_amt"]]
    print(f"Saving source data prepared for modeling..")
    df.to_csv('data/source_data_for_analysis_v1.csv', index=False)
        
    data["charge_dtls_cur"] = data["charge_dtls_cur"].replace(np.nan, "999", regex=True)
    data["charge_dtls_amt"] = data["charge_dtls_amt"].replace(np.nan, 0, regex=True)

    data["target_lat"] = data["target_lat"].replace(np.nan, 0, regex=True)
    data["target_lon"] = data["target_lon"].replace(np.nan, 0, regex=True)
    data["src_lon"] = data["src_lon"].replace(np.nan, 0, regex=True)
    data["src_lat"] = data["src_lat"].replace(np.nan, 0, regex=True)
    return df, data

# simple wrapper code around serial_calc to parallelize the work
def parallel_calc(df, func, n_core, col):
    futs = []
    df_split = np.array_split(df, n_core)
    # pool = concurrent.futures.ThreadPoolExecutor(max_workers = n_core)
    pool = concurrent.futures.ProcessPoolExecutor(max_workers = n_core)
    apply_partial = partial(func, col=col)
    return pd.concat(pool.map(apply_partial, df_split))

##------------------------------------------------------------------

def _50f_payor_add_ln_2(row, col):
    return get_nearest_loc({'latitude' : row[col][1], 'longitude': row[col][2]}, geocodes = COORDS) if row[col][0] == 'F' else ''

def proc_50f_payor_add_ln_2(df, col):
    apply_partial = partial(_50f_payor_add_ln_2, col=col)
    # df['50f_payor_add_ln_2'] = df.apply(apply_partial, axis=1)
    # return df
    return df.apply(apply_partial, axis=1)

##------------------------------------------------------------------

def _50k_payor_add_ln_2(row, col):
    return get_nearest_loc({'latitude' : row[col][1], 'longitude': row[col][2]}, geocodes = COORDS) if row[col][0] == 'K' else ''

def proc_50k_payor_add_ln_2(df, col):
    apply_partial = partial(_50k_payor_add_ln_2, col=col)
    # df['50k_payor_add_ln_2'] = df.apply(apply_partial, axis=1)
    # return df
    return df.apply(apply_partial, axis=1)

##------------------------------------------------------------------

def _59f_ben_add_ln_2(row, col):
    return get_nearest_loc({'latitude' : row[col][1], 'longitude': row[col][2]}, geocodes = COORDS)

def proc_59f_ben_add_ln_2(df, col):
    apply_partial = partial(_59f_ben_add_ln_2, col=col)
    # df['59f_ben_add_ln_2'] = df.apply(apply_partial, axis=1)
    # return df
    return df.apply(apply_partial, axis=1)


def get_time_elasped_on_col(col, start, end):
    print(f"Time elasped for {col} = {end - start}")

def reverse_samples_for_analysis(samples, ver = 1, sfx = '_fk', model_type = 'gan'):
    
    start = time.perf_counter()
    samples['50f_payor_add_ln_2'] = ''
    get_time_elasped_on_col('50f_payor_add_ln_2', start, time.perf_counter())
        
    start = time.perf_counter()
    samples.loc[samples['src_xfrr_type'] == 'F', '50f_payor_add_ln_2'] = \
                                    parallel_calc(samples[samples['src_xfrr_type'] == 'F'], 
                                    proc_50f_payor_add_ln_2, 8, 
                                    ['src_xfrr_type', 'src_lat', 'src_lon'])
    get_time_elasped_on_col('src_xfrr_type', start, time.perf_counter())
    
    
    start = time.perf_counter()
    condition = [samples['src_xfrr_type'] == 'F', samples['src_xfrr_type'] != 'F']
    choice = [samples['src_lat'], 0]
    samples['50f_payor_add_lat'] = np.select(condition, choice, default = 0)
    get_time_elasped_on_col('50f_payor_add_lat', start, time.perf_counter())
    
    start = time.perf_counter()    
    condition = [samples['src_xfrr_type'] == 'F']
    choice = [samples['src_lon']]
    samples['50f_payor_add_lon'] = np.select(condition, choice, default = 0)
    get_time_elasped_on_col('50f_payor_add_lon', start, time.perf_counter())
    
    start = time.perf_counter()
    condition = [samples['src_xfrr_type'] == 'K']
    choice = [samples['src_lat']]
    samples['50k_payor_add_lat'] = np.select(condition, choice, default = 0)
    get_time_elasped_on_col('50k_payor_add_lat', start, time.perf_counter())
    
    start = time.perf_counter()
    condition = [samples['src_xfrr_type'] == 'K']
    choice = [samples['src_lon']]
    samples['50k_payor_add_lon'] = np.select(condition, choice, default = 0)
    get_time_elasped_on_col('50k_payor_add_lon', start, time.perf_counter())
    
    start = time.perf_counter()
    samples['50k_payor_add_ln_2'] = ''
    get_time_elasped_on_col('50k_payor_add_ln_2', start, time.perf_counter())
    
    start = time.perf_counter()
    samples.loc[samples['src_xfrr_type'] == 'K', '50k_payor_add_ln_2'] = \
                                parallel_calc(samples[samples['src_xfrr_type'] == 'K'], 
                                proc_50k_payor_add_ln_2, 8, 
                                ['src_xfrr_type', 'src_lat', 'src_lon'])
    get_time_elasped_on_col('50k_payor_add_ln_2', start, time.perf_counter())
    
    start = time.perf_counter()
    samples['59f_ben_add_ln_2'] = parallel_calc(samples, 
                                proc_59f_ben_add_ln_2, 8, 
                                ['src_xfrr_type', 'target_lat', 'target_lon'])
    get_time_elasped_on_col('59f_ben_add_ln_2', start, time.perf_counter())
    
    start = time.perf_counter()
    samples['59f_ben_add_lat'] = samples['target_lat']
    get_time_elasped_on_col('59f_ben_add_lat', start, time.perf_counter())
    
    start = time.perf_counter()
    samples['59f_ben_add_lon'] = samples['target_lon']
    get_time_elasped_on_col('59f_ben_add_lon', start, time.perf_counter())
    
    start = time.perf_counter()
    samples['71A_chg_dtls'] = samples['charge_dtls']
    get_time_elasped_on_col('71A_chg_dtls', start, time.perf_counter())
    
    # start = time.perf_counter()
    samples['71f_chg_dtls_cur'] = samples['charge_dtls_cur']
    get_time_elasped_on_col('71f_chg_dtls_cur', start, time.perf_counter())
    
    start = time.perf_counter()
    # samples['71f_chg_dtls_amt'] = samples['charge_dtls_amt']
    samples.loc[samples['charge_dtls_amt'] < 0, 'charge_dtls_amt'] = 0
    samples['71f_chg_dtls_amt'] = samples['charge_dtls_amt']
    get_time_elasped_on_col('71f_chg_dtls_amt', start, time.perf_counter())
    
    ver = ver
    samples_data_file = f"data/samples_{model_type}_for_analysis_{sfx}_{ver}.csv"
    samples.to_csv(samples_data_file, index=False)
    return samples


fake = Faker()

# def get_user(num, dict_ = senders_count_dict):
#     key_list = list(senders_count_dict.keys())
#     val_list = list(senders_count_dict.values())
#     val = take_closest(val_list, num)
#     return key_list[val_list.index(val)]

def convert_to_mt103(df, final):  
    final['src_xfrr_type'] = df['src_xfrr_type']
#     final[':52A:'] =  df['52a_sender'].apply(lambda x: get_user(x)[:-3])
#     final[':57A:'] = df['57a_receiver'].apply(lambda x: get_user(x, dict_ = receivers_count_dict))
    final[':52A:'] = df['send_rec_pair'].str[:11]
    final[':57A:'] = df['send_rec_pair'].str[11:]
    final[':56A:'] = "None"
    final['Sender'] = final[':52A:'].str[:-3]                        
    final['Receiver'] = final[':57A:'].str[:-3]
    final[':20:'] = 'FIN001-NO-FX-679'
    final[':23B:'] = 'CRED'
    final[':33B:'] = df['33b_cur'] + \
                        df[['usd_amt', '33b_cur']].apply(lambda x: str(x[0] / FOREX[x[1]]), axis = 1)

    final[':32A:'] = pd.Timestamp("today").strftime("%y%m%d") + final[':33B:']
    final[':36:'] = 'None'
    
    print("Generating fake list for 50A")
    list_ = ['/00000000000000000000000000000' + str(int(np.random.rand() * 4)) + '\n' + final[':52A:'] 
            for i in range(df.shape[0])]
    
    final[':50A:'] = list_
    final[final['src_xfrr_type'] != 'A'][':50A:'] = ''

                                                                                   
    print("Generating fake list for 50F")
    list_ = ['DRLC/US/VA/000000' +
                str(int(np.random.rand() * 1000)) +
            '\n 1/' +
            fake.name() + 
            '\n 2/' + 
            fake.address().split('\n')[0] +
            '\n 3/' +
            df['50f_payor_add_ln_2'][i].replace(' ', '/')
            for i in range(df.shape[0])]

    final[':50F:'] = list_
    final[final['src_xfrr_type'] != 'F'][':50F:'] = ''

    print("Generating fake list for 50k")
    list_ = ['/000000000000000000000000000000' + 
             str(int(np.random.rand() * 1000)) +
            '\n 1/' +
            fake.name() + 
            '\n 2/' + 
            fake.address().split('\n')[0] +
            '\n 3/' +
            df['50k_payor_add_ln_2'][i].replace(' ', '/')
            for i in range(df.shape[0])]

    final[':50K:'] = list_
    final[final['src_xfrr_type'] != 'K'][':50K:'] = ''
    
    print("Generating fake list for 59F")
    list_ = ['/000000000000000000000000000000' + 
             str(int(np.random.rand() * 1000)) +
            '\n 1/' +
            fake.name() + 
            '\n 2/' + 
            fake.address().split('\n')[0] +
            '\n 3/' +
            df['59f_ben_add_ln_2'][i].replace(' ', '/')
            for i in range(df.shape[0])]

    final[':59F:'] = list_
    
    final[':71A:'] = df['71A_chg_dtls']
    final[':71F:'] = df['71f_chg_dtls_cur'] + df['71f_chg_dtls_amt'].astype('str')

    final[':71G:'] = ""
    final[':79:'] = ""
    return final[["Sender", "Receiver", ":20:", ":23B:", ":32A:", ":33B:", 
               ":36:", ":50A:", ":50F:", ":50K:", ":52A:", ":56A:", 
               ":57A:", ":59F:", ":71A:", ":71F:"]]