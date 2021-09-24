from bisect import bisect_left
from geopy.distance import geodesic
from collections import OrderedDict
from operator import getitem
import numpy as np


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
    
    coordinates['address'] = list(geocodes_dictionary.keys())
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


def get_nearest_loc(point, geocodes = None):
    from geopy.distance import geodesic
    if geocodes is None:
        return geo_loc(point)
    else:
        point_category = get_octa(point)
        geolist = geocodes['distance'][geocodes['category'] == point_category]
        distance = float(geodesic((0, 0), (point['latitude'], point['longitude'])).km)
        closest = np.where(geocodes['distance'] == take_closest(geolist, distance))
        # return distance, geocodes[closest]
        return closest, distance, geocodes['distance'][closest]