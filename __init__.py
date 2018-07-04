import vg.draw

EXT_IMG = '.jpg'

INPUTS = {
    "EDGE_HIGH"     : {'min': 100,  'max': 1000,'val': 500,         'clean': int},
    "EDGE_LOW"      : {'min': 10,   'max': 500, 'val': 100,         'clean': int},
    "THRESH_BLOCK"  : {'min': 3,    'max': 21,  'val': 11,          'clean': lambda x: 2*int(float(x)/2)+1},
    "THRESH_C"      : {'min': 1,    'max': 5,   'val': 2,           'clean': int},
    "RECT_SCALE"    : {'min': 1,    'max': 100, 'val': 20,          'clean': int},
    "CORNERS_NUM"   : {'min': 100,  'max': 1000,'val': 500,         'clean': int},
    "CORNERS_LOW"   : {'min': 0.001,'max': 0.1, 'val': 0.01,        'clean': float},
    "CORNERS_HIGH"  : {'min': 0.1,  'max': 50,  'val': 10,          'clean': float},
    "HATCH_DENSITY" : {'min': 0,    'max': 5000000, 'val':2500000,  'clean': int},
    "SIN_COUNT"     : {'min': 10,   'max': 1000,'val':50,           'clean': int},
    "SIN_HEIGHT"    : {'min': 10,   'max': 50,  'val':20,           'clean': int},
    "SIN_LENGTH"    : {'min': 2,    'max': 30,  'val':10,           'clean': int}
}
