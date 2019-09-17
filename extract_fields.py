
#Import packages

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent, show
from rasterio.mask import mask
import geopandas as gpd
import shapefile
import pandas as pd
import glob
import datetime

filenames_A = glob.glob('sat_img/*.jp2')
filenames_B = glob.glob('sat_img_2/*.jp2')


#Import shapefiles as dataframe (NOTE: these have been reprojected into the same CRS at the raster files)

train_data = pd.DataFrame(gpd.read_file('train_new/train_new.shp')) 
test_data = pd.DataFrame(gpd.read_file('test_new/test_new.shp'))


#Parses filename into key string

def name_splitter(name):
    
    name = name.split("/")[1]
    name = name.split(".")[0]
    
    return name

#Extracts field from satellite image raster

def get_shape(idx, row, file_A, file_B):
    
    try:
        with rasterio.open(file_A) as src:
            
            out_image, out_transform = rasterio.mask.mask(src, [row.geometry], crop=True, all_touched=False)

            return out_image, out_transform
        
    except:

        try:
            with rasterio.open(file_B) as src:
                
                out_image, out_transform = rasterio.mask.mask(src, [row.geometry], crop=True, all_touched=False)
                
                return out_image, out_transform

        except Exception as e:
            
                print('Error - ', e, idx)
                return np.array([np.nan])
            


#Saves extracted field image features for each band and date in turn

def extract_images(data, path):
    
    i = 1

    for file_A, file_B in zip(filenames_A, filenames_B):

        img_array = []
        
        name = name_splitter(file_A)

        for idx, row in data.iterrows():
            
            field_id = row['Field_Id']

            out_image, out_transform = get_shape(idx, row, file_A, file_B)
            
            img_array.append((field_id, out_image))
            
            
        pd.DataFrame(img_array, columns = ['Field_Id', name]).to_pickle(path + name + '.pkl')
        
        out_string = name + ' | ' + str(i) + '/' + str(len(filenames_A)) + ' files completed | ' + str(len(img_array)) + ' images saved | ' + str(datetime.datetime.now())
        print(out_string)
        
        i = i + 1
        

#Run extraction

print('\nSTART TRAIN ' + str(datetime.datetime.now()) + '\n')

extract_images(train_data, 'extracted_data/train/')

print('\nEND TRAIN ' + str(datetime.datetime.now())  + '\n')



print('\nSTART TEST ' + str(datetime.datetime.now()) + '\n')

extract_images(test_data, 'extracted_data/test/')

print('\nEND TEST ' + str(datetime.datetime.now())  + '\n')


#Load image arrays

train_image_files = glob.glob('extracted_data/train/*.pkl')
test_image_files = glob.glob('extracted_data/test/*.pkl')

layer_names = [item.split('/')[2].split('.')[0] for item in train_image_files]


#Merges all bands and dates into a single dataframe of image arrays

def merger(data, image_files):
    for file in image_files:
        new_frame = pd.read_pickle(file)

        data = pd.merge(data, new_frame, on='Field_Id')

    return data

train_data = merger(train_data, train_image_files)
test_data = merger(test_data, test_image_files)

#Save

train_data.to_pickle('extracted_data/all_train_data.pkl')
test_data.to_pickle('extracted_data/all_test_data.pkl')