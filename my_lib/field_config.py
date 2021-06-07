# from .constants import code_order

from my_lib.encoding import load_data_encoder

# ds_suffix = "vfdata"



PRE_DATE_ORDER = []
DATE_ORDER = ['td_sc', 'month', 'day', 'dow']
POST_DATE_ORDER = ['tcode_num', 'log_amount_sc']

DATA_KEY_ORDER = PRE_DATE_ORDER + DATE_ORDER + POST_DATE_ORDER # sorted(data[0].keys())

CLOCK_FIELDS = {"day": 31,
           "dow": 7,
           "month": 12,}


print("DATA_KEY_ORDER is", DATA_KEY_ORDER)
print("If this is not correct, edit my_lib/field_config.py and re-run notebook")


def get_field_info(ds_suffix):
    data_encoder = load_data_encoder(ds_suffix)

    ONE_HOT_DIMS = {"tcode_num": data_encoder.n_tcodes}

    FIELD_DIMS = {}
    for k in DATA_KEY_ORDER:

        if k in ONE_HOT_DIMS:
            depth = ONE_HOT_DIMS[k]
        elif k in CLOCK_FIELDS:
            depth = 2
        else:
            depth = 1

        FIELD_DIMS[k] = depth



    FIELD_STARTS = {}
    start = 0
    for k in DATA_KEY_ORDER:

        FIELD_STARTS[k] = start
        start += FIELD_DIMS[k]




    FIELD_DIMS_TAR = {}
    for k in DATA_KEY_ORDER:

        if k in ONE_HOT_DIMS:
            depth = 1
        elif k in CLOCK_FIELDS:
            depth = 1
        else:
            depth = 1

        FIELD_DIMS_TAR[k] = depth



    FIELD_STARTS_TAR = {}
    start = 0
    for k in DATA_KEY_ORDER:

        FIELD_STARTS_TAR[k] = start
        start += FIELD_DIMS_TAR[k]
        
        
    return ONE_HOT_DIMS, FIELD_DIMS, FIELD_STARTS, FIELD_DIMS_TAR, FIELD_STARTS_TAR

    
    
    
