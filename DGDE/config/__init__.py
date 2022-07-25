from .defaults import _C as cfg

# TYPE_ID_CONVERSION = {
#     'Car': 0,
#     'Pedestrian': 1,
#     'Cyclist': 2,
#     'Van': -4,
#     'Truck': -4,
#     'Person_sitting': -2,
#     'Tram': -99,
#     'Misc': -99,
#     'DontCare': -1,
# }

TYPE_ID_CONVERSION = {
    'car': 0,
    'pedestrian': 1,
    'bicycle': 2,
    'motorcycle': 3,
    'barrier': 4,
    'bus': 5,
    'construction_vehicle':6,
    'traffic_cone':7,
    'trailer':8,
    'truck':9,
    'DontCare': 10,
}