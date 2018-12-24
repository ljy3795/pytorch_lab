# packages for loading movielens dataset

"""Download and extract the MovieLens dataset from GroupLens website.
"""

import os
import sys
import zipfile

from six.moves import urllib  # pylint: disable=redefined-builtin

DATASETS = ['ml-1m','ml-20m', 'ml-20m', 'ml-100k', 'ml-latest-small']
RATINGS_FILE = ["ratings.dat","ratings.csv"]
MOVIES_FILE = ["movies.dat","movies.csv"]
USERS_FILE = ["users.dat","users.csv"]

# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/"


def download_unzip(dataset_nm, data_dir, force=False):
    """Download Movielens data set from GroupLens website to 'data_dir'
      Args:
          dataset_nm = ml-1m / ml-20m / 'ml-100k' / 'ml-latest-small' (default = ml-1m)
          data_dir = the directory to download dataset from GroupLens (Default : ./downloaded)
          force = Force to download regardless of pre-downloaded file on 'data_dir) (Default : False)
      """

    if dataset_nm not in DATASETS:
        raise ValueError("dataset {} is not in {{{}}}".format(dataset_nm, ",".join(DATASETS)))

    data_subdir = os.path.join(os.path.abspath(data_dir), dataset_nm)
    if not os.path.exists(data_subdir):
        os.makedirs(data_subdir)
    
    expected_files = ["{}.zip".format(dataset_nm)]
    expected_files.extend(RATINGS_FILE)
    expected_files.extend(MOVIES_FILE)
    expected_files.extend(USERS_FILE)

    if force == False:
        file_list = os.listdir(data_subdir)
        if set(file_list).intersection(expected_files) == set(expected_files):
            return
        
    url = "{}{}.zip".format(_DATA_URL, dataset_nm)    
    zip_path = os.path.join(data_subdir, "{}.zip".format(dataset_nm))
    zip_path, _ = urllib.request.urlretrieve(url, zip_path)
    statinfo = os.stat(zip_path)
    print('downloaded filesize is {0:.2f} mb'.format(int(statinfo.st_size)/1024/1024))

    zipfile.ZipFile(zip_path, "r").extractall(os.path.join(data_subdir,".."))
    file_list = os.listdir(data_subdir)
    file_list = set(file_list).intersection(expected_files)
    print("downloaded files : {}".format(file_list))