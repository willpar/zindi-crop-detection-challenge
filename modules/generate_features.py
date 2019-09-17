import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn import base


### LOCATION FEATURES

class GenerateLocationFeatures(base.BaseEstimator, base.TransformerMixin):

    '''

    Generates location zone features

    -----------

    - Accepts pandas dataframe, returns pandas dataframe.

    - Generates location-based zone features for each field, based on a k-means clustering algorithm

    -----------

    Methods:

    __init__(self, zone_sizes=[10, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000])
    fit(self, data)
    transform(self, data, path = 'location_features')

    Arguments:

    zone_sizes = number of zones to cluster fields into (default=[10, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000])

    data = data to create features from
    save = specify whether to pickle the resulting file (default = True)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, zone_sizes=[10, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000]):

        self.zone_sizes = zone_sizes

    def fit(self, data):

        from sklearn.cluster import KMeans

        locations = data[['x_loc', 'y_loc']]

        self.clst_dict = {}

        for n_clusters in self.zone_sizes:
            clst = KMeans(n_clusters=n_clusters)
            clst.fit(locations)

            self.clst_dict[str(n_clusters)] = clst

        return self

    def transform(self, data, save=False, path='location_features'):

        locations = data[['x_loc', 'y_loc']]

        new_features = []
        new_features.append(data['Field_Id'])

        for n_clusters in self.zone_sizes:
            clst = self.clst_dict[str(n_clusters)]

            clusters = clst.predict(locations)

            features = pd.DataFrame({'location_' + str(n_clusters): clusters})

            new_features.append(features)

        new_features = pd.concat(new_features, axis=1)

        if save == True:
            new_features.to_pickle(path + '.pkl')

        return new_features


### VEGINDEX FEATURES

class GenerateVegIndexFeatures(base.BaseEstimator, base.TransformerMixin):

    '''

    Generate vegetative index features

    -----------

    - Accepts pandas dataframe, returns pandas dataframe.

    - Generates various vegetative index (VI) arrays for each image.

    - Features include: mean, std, max, min. Calculated for each timeframe.

    -----------

    Methods:

    __init__(self, dates = None, drop=[], dropna = False, fillna = True)
    fit(self) - NOTE: does not do anything
    transform(self, data, save = False, path = '/vegindex_features')

    Arguments:

    dates = which dates to calculate VI for (default = all dates in data)
    drop = keywords to drop ie. date '0804' (default = [])
    dropna = removes any nan values in arrays after processing (eg. those arising by divide-by-zero),
            arrays are left flattened (default = False)
    fillna = fills nan values with zeros, only works if dropna = False (default = True)

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, dates=None, drop=[], dropna=False, fillna=True):

        self.dates = dates
        self.drop = drop
        self.dropna = dropna
        self.fillna = fillna

    def fit(self):
        return self

    def transform(self, data, save=False, path='/vegindex_features'):

        def ndvi(x, RED_col, NIR_col):

            NDVI = (x[NIR_col].astype(np.int16) - x[RED_col].astype(np.int16)) / (
                        x[NIR_col].astype(np.int16) + x[RED_col].astype(np.int16))

            NDVI = np.where((NDVI == 0) & (x[RED_col] != 0) & (x[NIR_col] != 0), 0.0000001, NDVI)

            return NDVI

        def dvi(x, RED_col, NIR_col):

            DVI = np.subtract(x[NIR_col].astype(np.int16), x[RED_col].astype(np.int16))

            DVI = np.where((DVI == 0) & (x[RED_col] != 0) & (x[NIR_col] != 0), 0.0000001, DVI)

            return DVI

        def rvi(x, RED_col, NIR_col):

            RVI = np.divide(x[NIR_col].astype(np.int16), x[RED_col].astype(np.int16))

            RVI = np.where((RVI == 0) & (x[RED_col] != 0) & (x[NIR_col] != 0), 0.0000001, RVI)

            return RVI

        def ipvi(x, RED_col, NIR_col):

            IPVI = x[NIR_col].astype(np.int16) / (x[NIR_col].astype(np.int16) + x[RED_col].astype(np.int16))

            IPVI = np.where((IPVI == 0) & (x[RED_col] != 0) & (x[NIR_col] != 0), 0.0000001, IPVI)

            return IPVI

        def savi(x, RED_col, NIR_col):

            SAVI = (x[NIR_col].astype(np.int16) - x[RED_col].astype(np.int16)) / (
                        2 * (x[NIR_col].astype(np.int16) + x[RED_col].astype(np.int16) + 1))

            SAVI = np.where((SAVI == 0) & (x[RED_col] != 0) & (x[NIR_col] != 0), 0.0000001, SAVI)

            return SAVI

        def arvi(x, BLUE_col, RED_col, NIR_col):

            RB = (2 * x[RED_col].astype(np.int16)) - x[BLUE_col].astype(np.int16)

            ARVI = (x[NIR_col].astype(np.int16) - RB) / (x[NIR_col].astype(np.int16) + RB)

            ARVI = np.where((ARVI == 0) & (x[RED_col] != 0) & (x[BLUE_col] != 0) & (x[NIR_col] != 0), 0.0000001, ARVI)

            return ARVI

        if self.dates == None:
            img_cols = [col for col in data.columns if col not in
                        ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

            self.dates = list(set([img.split('_')[0] for img in img_cols]))

        for d in self.drop:
            self.dates = [date for date in self.dates if d not in date]

        new_features = []
        new_features.append(data['Field_Id'])

        for date in self.dates:

            BLUE_col = date + '_B02'
            RED_col = date + '_B04'
            NIR_col = date + '_B08'

            NDVI = data.apply(lambda x: ndvi(x, RED_col, NIR_col), axis=1)
            DVI = data.apply(lambda x: dvi(x, RED_col, NIR_col), axis=1)
            RVI = data.apply(lambda x: rvi(x, RED_col, NIR_col), axis=1)
            IPVI = data.apply(lambda x: ipvi(x, RED_col, NIR_col), axis=1)
            # SAVI = data.apply(lambda x: savi(x, RED_col, NIR_col), axis = 1)
            ARVI = data.apply(lambda x: arvi(x, BLUE_col, RED_col, NIR_col), axis=1)

            if self.dropna == True:
                NDVI = NDVI.apply(lambda x: x[~np.isnan(x)])
                DVI = DVI.apply(lambda x: x[~np.isnan(x)])
                RVI = RVI.apply(lambda x: x[~np.isnan(x)])
                IPVI = IPVI.apply(lambda x: x[~np.isnan(x)])
                # SAVI = SAVI.apply(lambda x: x[~np.isnan(x)])
                ARVI = ARVI.apply(lambda x: x[~np.isnan(x)])

            if self.fillna == True:
                NDVI = NDVI.apply(lambda x: np.nan_to_num(x, nan=0))
                DVI = DVI.apply(lambda x: np.nan_to_num(x, nan=0))
                RVI = RVI.apply(lambda x: np.nan_to_num(x, nan=0))
                IPVI = IPVI.apply(lambda x: np.nan_to_num(x, nan=0))
                # SAVI = SAVI.apply(lambda x: np.nan_to_num(x, nan=0))
                ARVI = ARVI.apply(lambda x: np.nan_to_num(x, nan=0))

            features = pd.DataFrame({date + '_NDVI': NDVI, date + '_DVI': DVI,
                                     date + '_RVI': RVI, date + '_IPVI': IPVI,
                                     date + '_ARVI': ARVI})

            new_features.append(features)

        new_features = pd.concat(new_features, axis=1)

        if save == True:
            new_features.to_pickle(path + '.pkl')

        return new_features


### TIMEDIFF FEATURES

class GenerateTimeDiffFeatures(base.BaseEstimator, base.TransformerMixin):
    '''

    Generate time difference features

    -----------

    - Accepts pandas dataframe, returns pandas dataframe

    - For each layer, calculates the fractional difference between the image over different timescales

    - Timescales include: Winter-Summer (01Jan-10Jul), Winter-Spring (01Jan-22Mar),
                            Spring-Summer (22Mar-20Jun), Summer-Autumn(?) (20Jun-19Aug)

    -----------

    Methods:

    __init__(self, layers = None, drop = [], dropna = False)
    fit(self) - NOTE: does not do anything
    transform(self, data, save = False, path = '/timediff_features')

    Arguments:

    layers = which layers to calculate timediff for (default = all dates in data)
    drop = keywords to drop ie. layer 'B01' (default = [])
    dropna = removes any nan values in arrays after processing (eg. those arising by divide-by-zero),
            arrays are left flattened (default = False)
    fillna = fills nan values with zeros, only works if dropna = False (default = True)

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used

    '''

    def __init__(self, layers=None, drop=[], dropna=False, fillna=True):

        self.layers = layers
        self.drop = drop
        self.dropna = dropna
        self.fillna = fillna

    def fit(self):
        return self

    def transform(self, data, save=False, path='/timediff_features'):

        def change_calc(x, new_col, old_col):

            change = (x[new_col].astype(np.float32) - x[old_col].astype(np.float32)) / x[old_col].astype(np.float32)

            change = np.where((change == 0) & (x[new_col] != 0) & (x[old_col] != 0), 0.0000001, change)

            return change

        if self.layers == None:
            img_cols = [col for col in data.columns if col not in
                        ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

            self.layers = list(set([img.split('_')[1] for img in img_cols]))

        for d in self.drop:
            self.layers = [layer for layer in self.layers if d not in layer]

        new_features = []
        new_features.append(data['Field_Id'])

        for layer in self.layers:

            JAN01_col = '0101_' + layer
            MAR22_col = '0322_' + layer
            JUL10_col = '0710_' + layer
            JUN20_col = '0620_' + layer
            AUG19_col = '0819_' + layer

            time_diff_WINSUM = data.apply(lambda x: change_calc(x, JUL10_col, JAN01_col), axis=1)
            time_diff_WINSPR = data.apply(lambda x: change_calc(x, MAR22_col, JAN01_col), axis=1)
            time_diff_SPRSUM = data.apply(lambda x: change_calc(x, JUN20_col, MAR22_col), axis=1)
            time_diff_SUMAUT = data.apply(lambda x: change_calc(x, AUG19_col, JUN20_col), axis=1)

            if self.dropna == True:
                time_diff_WINSUM = time_diff_WINSUM.apply(lambda x: x[~np.isnan(x)])
                time_diff_WINSPR = time_diff_WINSPR.apply(lambda x: x[~np.isnan(x)])
                time_diff_SPRSUM = time_diff_SPRSUM.apply(lambda x: x[~np.isnan(x)])
                time_diff_SUMAUT = time_diff_SUMAUT.apply(lambda x: x[~np.isnan(x)])

            if self.fillna == True:
                time_diff_WINSUM = time_diff_WINSUM.apply(lambda x: np.nan_to_num(x, nan=0))
                time_diff_WINSPR = time_diff_WINSPR.apply(lambda x: np.nan_to_num(x, nan=0))
                time_diff_SPRSUM = time_diff_SPRSUM.apply(lambda x: np.nan_to_num(x, nan=0))
                time_diff_SUMAUT = time_diff_SUMAUT.apply(lambda x: np.nan_to_num(x, nan=0))

            features = pd.DataFrame({layer + '_time_diff_WINSUM': time_diff_WINSUM,
                                     layer + '_time_diff_WINSPR': time_diff_WINSPR,
                                     layer + '_time_diff_SPRSUM': time_diff_SPRSUM,
                                     layer + '_time_diff_SUMAUT': time_diff_SUMAUT,
                                     })

            new_features.append(features)

        new_features = pd.concat(new_features, axis=1)

        if save == True:
            new_features.to_pickle(path + '.pkl')

        return new_features


### STATISTICAL FEATURES

class GenerateStatFeatures(base.BaseEstimator, base.TransformerMixin):
    '''

    Generate statistical features

    -----------

    - Accepts pandas dataframe, returns pandas dataframe.

    - Generates statistical features for each image, excluding any zero-valued pixels.

    - Features include: mean, std, max, min. Calculated for each layer and timeframe in cols.

    -----------

    Methods:

    __init__(self, cols=None, drop = [], ignorena = True)
    fit(self) - NOTE: does not do anything
    transform(self, data, save = False, path = 'stat_features')

    Arguments:

    cols = which columns to calculate stats for (default = all image cols in data)
    drop = keywords to drop ie. layer 'B01' or date '0804' (default = [])
    ignorena = converts nan values to zero, such that they are ignored in calculating stats

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, cols=None, drop=[], ignorena=True):

        self.cols = cols
        self.drop = drop
        self.ignorena = ignorena

    def fit(self):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, data, save=False, path='stat_features'):

        if self.cols == None:
            self.cols = [col for col in data.columns if col not in
                         ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

        for d in self.drop:
            self.cols = [col for col in self.cols if d not in col]

        new_features = []
        new_features.append(data['Field_Id'])

        def max_arr(x):
            try:
                return x[x != 0].max()
            except:
                return 0

        def min_arr(x):
            try:
                return x[x != 0].min()
            except:
                return 0

        for col in self.cols:

            if self.ignorena == True:
                data[col].apply(lambda x: np.nan_to_num(x, nan=0))

            mean_img = data[col].apply(lambda x: x[x != 0].mean())
            std_img = data[col].apply(lambda x: np.std(x[x != 0]))
            max_img = data[col].apply(lambda x: max_arr(x))
            min_img = data[col].apply(lambda x: min_arr(x))

            features = pd.DataFrame({col + '_mean': mean_img, col + '_std': std_img,
                                     col + '_max': max_img, col + '_min': min_img})

            new_features.append(features)

        new_features = pd.concat(new_features, axis=1)

        if save == True:
            new_features.to_pickle(path + '.pkl')

        return new_features


### MEAN DIFFERENCE FEATURES

class GenerateMeanDiffFeatures(base.BaseEstimator, base.TransformerMixin):
    '''

    Generate mean difference features

    -----------

    - Accepts pandas dataframe, returns pandas dataframe

    - For each column, calculates the difference between the image and the mean image for each crop type,
        excluding any zero-valued pixels.

    - NOTE: Warning! Takes a long time to run and generates n_cols*9 features

    -----------

    Methods:

    __init__(self, cols=None, drop = [])
    fit(self, data)
    transform(self, data, save = False, path = 'meandiff_features')

    Arguments:

    cols = which columns to calculate stats for (default = all image cols in data)
    drop = keywords to drop ie. layer 'B01' or date '0804' (default = [])

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, cols=None, drop=[]):

        self.cols = cols
        self.drop = drop

    def fit(self, data):

        def get_mean(x):

            arr_list = x.values.tolist()
            arr = [item[item != 0] for item in arr_list]
            arr = np.concatenate(arr)

            return np.mean(arr)

        if self.cols == None:
            self.cols = [col for col in data.columns if col not in
                         ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

        for d in self.drop:
            self.cols = [col for col in self.cols if d not in col]

        self.grouped_means_dict = {}

        for col in self.cols:
            grouped = data.groupby('Crop_Id_Ne')[col].apply(lambda x: get_mean(x))
            grouped_means = pd.DataFrame(grouped).reset_index()

            self.grouped_means_dict[col] = grouped_means

        return self

    def transform(self, data, save=False, path='meandiff_features'):

        def get_meandiff(crop_id, col, x):

            mean_crop = grouped_means[grouped_means['Crop_Id_Ne'] == crop_id][col].values
            mean_img = x[x != 0].mean()
            meandiff = (mean_img - mean_crop) / mean_crop

            return meandiff[0]

        if self.cols == None:
            self.cols = [col for col in data.columns if col not in
                         ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

        for d in self.drop:
            self.cols = [col for col in self.cols if d not in col]

        new_features = []
        new_features.append(data['Field_Id'])

        for col in self.cols:
            grouped_means = self.grouped_means_dict[col]

            mean_diff_crop_1 = data[col].apply(lambda x: get_meandiff('1', col, x))
            mean_diff_crop_2 = data[col].apply(lambda x: get_meandiff('2', col, x))
            mean_diff_crop_3 = data[col].apply(lambda x: get_meandiff('3', col, x))
            mean_diff_crop_4 = data[col].apply(lambda x: get_meandiff('4', col, x))
            mean_diff_crop_5 = data[col].apply(lambda x: get_meandiff('5', col, x))
            mean_diff_crop_6 = data[col].apply(lambda x: get_meandiff('6', col, x))
            mean_diff_crop_7 = data[col].apply(lambda x: get_meandiff('7', col, x))
            mean_diff_crop_8 = data[col].apply(lambda x: get_meandiff('8', col, x))
            mean_diff_crop_9 = data[col].apply(lambda x: get_meandiff('9', col, x))

            features = pd.DataFrame({col + '_mean_diff_crop_1': mean_diff_crop_1,
                                     col + '_mean_diff_crop_2': mean_diff_crop_2,
                                     col + '_mean_diff_crop_3': mean_diff_crop_3,
                                     col + '_mean_diff_crop_4': mean_diff_crop_4,
                                     col + '_mean_diff_crop_5': mean_diff_crop_5,
                                     col + '_mean_diff_crop_6': mean_diff_crop_6,
                                     col + '_mean_diff_crop_7': mean_diff_crop_7,
                                     col + '_mean_diff_crop_8': mean_diff_crop_8,
                                     col + '_mean_diff_crop_9': mean_diff_crop_9
                                     })

            new_features.append(features)

        new_features = pd.concat(new_features, axis=1)

        if save == True:
            new_features.to_pickle(path + '.pkl')

        return new_features


### RESIZED IMAGES

class GenerateResizedImages(base.BaseEstimator, base.TransformerMixin):
    '''

    Generate resized images

    -----------

    - Accepts pandas dataframe, returns numpy array of dimension (len(data), new_size)

    - Generates uniform array of either upsampled or downsampled images depending on their original size

    - !!! NOTE: concerned that min-max scaling is wrong. Perhaps should try simply x/x.max() !!!

    -----------

    Methods:

    __init__(self, cols=None, drop = [], new_size = (1,32,32), scale_factor = None)
    fit(self, data)
    transform(self, data, save = False, path = 'resized_images')

    Arguments:

    cols = which columns to calculate stats for (default = all image cols in data)
    drop = keywords to drop ie. layer 'B01' or date '0804' (default = [])
    new_size = dimensions to resize images to (default = (1,32,32))
    scale_factor = min-max scale each array and multiply by the scale_factor eg. 255 (default = None)

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, cols=None, drop=[], new_size=(1, 32, 32), scale_factor=None):

        self.cols = cols
        self.drop = drop
        self.new_size = new_size
        self.scale_factor = scale_factor

    def fit(self, data):
        return self

    def transform(self, data, save=False, path='resized_images'):

        from skimage.transform import resize

        if self.cols == None:
            self.cols = [col for col in data.columns if col not in
                         ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

        for d in self.drop:
            self.cols = [col for col in self.cols if d not in col]

        new_images = []

        for col in self.cols:

            if self.scale_factor != None:
                data[col] = data[col].apply(lambda x: self.scale_factor * (x - np.min(x)) / np.ptp(x).astype(int))

            resized_images = np.array([resize(image, self.new_size, mode='constant') for image in data[col]])

            new_images.append(resized_images)

        new_images = np.hstack(new_images)

        if save == True:
            np.save(path + '.npy', new_images)

        return new_images


### OBJECT TO PIXEL TRANSFORMATION

class ObjectToPixels(base.BaseEstimator, base.TransformerMixin):
    '''


    Expand images to individual pixels

    -----------

    - Accepts pandas dataframe, returns expanded pandas dataframe

    - Expands image arrays such that each row represents a single pixel

    - NOTE: all images have to be the same size,
            hence only B02, B03, B04, B08 layers (along with VI images) may be expanded

    -----------

    Methods:

    __init__(self, cols=None, drop = [])
    fit(self, data)
    transform(self, data, save = False, path = 'expanded_pixels')

    Arguments:

    cols = which columns to calculate stats for (default = all image cols in data)
    drop = keywords to drop ie. layer 'B01' or date '0804' (default = [])

    data = data to create features from
    save = specify whether to pickle the resulting file (default = False)
    path = path and filename to save to if save = True, set directory to 'train' or 'test' depending on set used


    '''

    def __init__(self, cols=None, drop=[]):

        self.cols = cols
        self.drop = drop

    def fit(self, data):
        return self

    def transform(self, data, save=False, path='expanded_pixels'):

        # Drop dodgy field
        # data = data[data['Field_Id'] != 2402]

        # Select columns to perform transformation on
        if self.cols == None:
            self.cols = [col for col in data.columns if col not in
                         ['Field_Id', 'Area', 'Subregion', 'Crop_Id_Ne', 'geometry', 'x_loc', 'y_loc']]

        # Drop by default layers which have inconsistent sizes
        default_drop = ['B01', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B10', 'B11', 'B12', 'TCI']

        self.drop.extend(default_drop)

        for d in self.drop:
            self.cols = [col for col in self.cols if d not in col]

        # Remove zeros and flatten each image

        def remove_zeros(x):

            x_new = x[x != 0]

            if len(x_new) == 0:
                x_new = np.array([np.nan])

            return x_new


        for col in self.cols:
            #data[col] = data[col].apply(lambda x: x[x != 0])
            data[col] = data[col].apply(lambda x: remove_zeros(x))

        # Calculate lengths of each image
        lengths = [len(item) for item in data[self.cols[0]]]

        # Initialise new feature object
        all_pixels = []
        all_pixels.append(pd.DataFrame({"Field_Id": np.repeat(data['Field_Id'].values, lengths)}))

        try:
            all_pixels.append(pd.DataFrame({"Crop_Id_Ne": np.repeat(data['Crop_Id_Ne'].values, lengths)}))
        except:
            pass

        # Expand images into one row per pixel
        for col in self.cols:
            all_pixels.append(pd.DataFrame({col: np.concatenate(data[col].values)}))

        # Convert to dataframe
        new_pixels = pd.concat(all_pixels, axis=1)

        if save == True:
            new_pixels.to_pickle(path + '.pkl')

        return new_pixels

