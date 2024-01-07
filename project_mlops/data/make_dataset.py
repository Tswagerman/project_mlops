import numpy as np
import pandas as pd 
import os

def get_data():
    local_dvc_repo = '<path-to-local-dvc-repo>'
    os.system(f'dvc pull -r myremote')

    pass

if __name__ == '__main__':
    get_data()
    pass['remote "storage"']
    url = gdrive://1X2P4EfkFSkOlSrUg8ugygr5blUvGojsE