###########################################################
# Author: J. Pastorino, Ashis K. Biswas
#
# Description: Auxiliary functions.

import pickle
import json                  # For reading config. / and managing objects

##########################################################################################
def save_obj(obj, filename ):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except:
        return False
    else:
        return True



##########################################################################################
def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

    

##########################################################################################
def read_config(CONFIG_FILE): 
    ## Returns a dictionary with the configuration parameters. 
    with open(CONFIG_FILE, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    return data
