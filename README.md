### Zindi FarmPin Crop Detection Challenge

---

### Introduction

This repo outlines my approach to the Zindi FarmPin Crop Detection Challenge:

https://zindi.africa/competitions/farm-pin-crop-detection-challenge
        
The competition demands a model that is capable of classifying fields by crop type using Sentinel-2 satellite imagery of an area of the Orange River, South Africa.

The dataset is comprised of satellite images across 13 spectral bands, and taken at 11 discrete time intervals during 2017. 

The training data is supplied in the form of 2497 shape files which define the perimeter of each field, the test data consists of 1074 fields for which predictions must be made. The predictions take the form of probabilities of each field belonging to one of 9 classes:

1. Cotton
2. Dates
3. Grass
4. Lucern
5. Maize
6. Pecan
7. Vacant/No Crop
8. Vineyard
9. Vineyard & Pecan Intercrop

The performance of the classifier in making these predictions is scored using a log-loss error function.

---

### Description of repository contents

```
root

|-- raw_data                                 (UNPOPULATED)
        |-- sat_img                          (Sentinel-2 data part 1)
        |-- sat_img_2                        (Sentinel-2 data part 2)
        |-- test_new                         (train data shapefiles, reprojected into new CRS)
        |-- train_new                        (test data shapefiles, reprojected into new CRS)
        
|-- extracted_data                           (UNPOPULATED)
        |-- train                            (train fields extracted from each satellite image band and date)
        |-- test                             (test fields extracted from each satellite image band and date)
        |-- train_data.pkl                   (train field images as a single dataframe)
        |-- test_data.pkl                    (test field images as a single dataframe)
        
|-- processed_data                           (UNPOPULATED)
        |-- train                            (same contents as test below)
        |-- test               
            |-- expanded_pixels.pkl          (generated features: field images converted to individual pixels)
            |-- location_features.pkl        (generated features: location of fields as zone clusters)
            |-- resized_images.pkl           (generated features: field images scaled to the same dimensions)
            |-- stat_features.pkl            (generated features: statistical summary features for each field)
            |-- timediff_features.pkl        (generated features: seasonal differences in intensity)
            |-- vegindex_features.pkl        (generated features: calculated vegetative indices)
            
|-- modules
        |-- generate_features.py             (helper classes and functions to create features)
        |-- process_data.py                  (helper classes and functions to pre-process data)
        |-- run_models.py                    (helper classes and functions to run models and ensembles)
        |-- metaclassifiers.py               (custom estimators for ensembling models)
        
|-- extract_fields.py                        (extracts fields from satellite images using shapefiles)
|-- generate_features.ipynb                  (executes feature generator modules)
|-- explore_data.ipynb                       (very basic visualisation of data and features)
|-- run_object_ensemble.ipynb                (solution 1: model ensemble using field-based features)
|-- run_pixel_ensemble.ipynb                 (solution 2: model ensemble using pixel-based features)
|-- run_CNN.ipynb                            (solution 3: CNN using field images)
|-- requirements.txt
|-- README.md

```

To run the project:

1. Download satellite imagery and train/test shapefiles from Zindi competition website to `raw_data` folder (too large to upload to GitHub).
2. Reproject train/test shapefiles into desired co-ordinate reference system (Sentinel-2 uses UTM/WGS84).
3. Run `extract_fields.py` to extract the individual fields from each satellite image (one for each band and date, 143 in total). This uses the `mask` module provided by the `rasterio` package.
4. Run the transformer classes contained in `modules > generate_features.py` to generate features from the extracted field images, an example is given in `generate_features.ipynb`.
5. EDA may be performed, a brief summary of which is presented in `explore_data.ipynb`.
6. Run classifier models, detailed below, examples of which are presented in `run_object_ensemble.ipynb`, `run_pixel_ensemble.ipynb` and `run_CNN.ipynb`.

---

### Methodology

This approach uses a number of methods to achieve a log-loss score of 0.581 (17th place) on the Zindi private leaderboard. Unfortunately due to time constraints, the full potential of these methods was not explored and it is believed that a much better score may be obtained without much further work.

Current research into machine learning for crop type classification from satellite imagery (eg. [1-2]) focuses on a number of common techniques. These may be split into object-based methods -- treating each field as a whole -- or pixel-based methods -- training and classifying based upon each individual pixel that makes up a field. The use of neural networks is a natural fit for this kind of problem: fully-connected, CNN and LTSM RNN have all been demonstrated as capable models. Additionally, the potential of Random Forest and Support Vector Machine classifiers has also been demonstrated in the literature.

[1] *A high-performance and in-season classification system of field-level crop types using time-series Landsat data and a machine learning approach*, Y. Cai, et. al, Remote Sensing of Environment, 210 (2018) 35-47

[2] *Deep Learning Classification of Land Cover and Crop Types Using Remote Sensing Data*, N. Kussul, et. al, IEEE Geoscience and Remote Sensing Letters, 14 (5) (2017) 778-782

#### Features

A number of features are generated using the custom transformer classes in the `modules` directory:

* **Location :** the latitude and longitude of each field in the training set is used to fit a KMeans clustering algorithm which divides the location of the fields into 'zones'. This fitted model then acts as a transformer to label each field with a zone number -- with the number of clusters specifying the number of zones (features generated for between 10-2000 zones in this example). <br>
  <br>
* **Vegetative Index :** certain transformations of spectral bands may be used to accentuate the spectral response of plants such that the characteristics of the vegetation may be measured, the background soil signal, as well as atmospheric and topographic effects may be discounted. A selection of these are calculated for each field.<br>
  <br>
* **Time Difference :** the change in the appearance of a field over time is likely to be closely related to the crop type, as a result of seeding and growth cycles throughout the year. Intensity difference features are calculated between the seasons of the year in an attempt to highlight this time-series pattern in the absence of several years of data.<br>
  <br>
* **Statistical Features :** the mean, std, max and min pixel intensities are calculated for each image of each field.<br>
  <br>
* **Mean Difference Features :** the deviation of a field from the 'typical' image for each crop may be a useful indicator of their similarity. The mean pixel value for each crop type in the training set is calculated, the difference between each pixel in an image and this mean is then determined for each field.<br>
  <br>
* **Resized Images :** each field is a different size and shape, with the image represented by a numpy array of pixel intensities padded by zeros. In order to standardise these images for NN training, they are either downsampled or upsampled using interpolation to common dimensions (32 x 32 in this example).<br>
  <br>
* **Transform To Pixels :** for training the pixel-based models, the dataset is transformed from a set of image arrays to a set of pixel values.<br>

#### Helper Classes and Functions

Alongside the feature generators, several classes and functions are written to simplify transforming the data and building models. These consist of a number of processing steps, which perform actions on specific columns of the data based upon subtractive keyword rules (see docstring for explanation):

`SelectFeatures`,
`Scale`,
`OneHot`

A custom ensembling module, `ModelEnsemble`, is created that allows the predictions of multiple classifiers to be stacked using one or more metaclassifiers. The module is initialised with dicts specifying the classifiers and metaclassifers to be used, along with their parameters. When the fit method is called two-fold training of the classifiers is performed, and their out-of-fold predictions combined to train the metaclassifiers. When predict is called, the classifiers are trained using the full data available and predictions are made using these as well as the pre-trained metaclassifiers. (Note: if time had allowed, a further module would be made to allow numerous `ModelEnsemble` instances trained on different feature datasets to be stacked.)

For the pixel-based ensemble models, an additional module `PixelToObject` is called on the fitted `ModelEnsemble` instance. This transforms the outputs from individual pixel predictions to whole field predictions by taking the mean pixel prediction for each object. (Note: An attempt was made to heuristically improve this method of aggregating individual pixel predictions, both by boosting the class probability that is most confidently 'voted-for' as well as applying Laplace Smoothing to those class probability distributions for fields where no confident prediction exists.)

#### Models

Three distinct techniques were tested:

1. **Object-based ensemble method :** This method makes use of statistical and location features calculated for each field using an ensemble of linear (LogisticRegression), nearest-neighbours (KNN) and tree-based methods (RandomForest, ExtraTrees, XGBoost). See `run_object_ensemble.ipynb`.
2. **Pixel-based ensemble method :** This method makes use of individual pixel features (spectral band intensities, vegetative index and time difference values) to train an ensemble of linear (LogisticRegression) and tree-based methods (RandomForest, ExtraTrees, XGBoost). See `run_pixel_ensemble.ipynb`.
3. **CNN method :** This method trains convolutional neural networks using individual pixel feature values (1D CNN) or resized field images (2D CNN). See `run_CNN.ipynb`.<br> 

---

### Outcomes

The brief tests that were carried out indicated that pixel-based methods were the best performing. The highest leaderboard score (\~0.58) was achieved using a single XGBoost model using pixel features including the intensities of the B02, B03, B04 and B08 spectral bands (R,G,B,NIR) in combination with vegetative indices. A slightly poorer score (\~0.64) was achieved using a much faster ensemble of object-based models, trained on the statistical features. The CNN models achieved only (\~0.70). 

However, it should be noted that this was an incomplete effort limited by time and it should be expected that these methods may be improved in numerous ways.

A selection of other techniques were explored:

* Data augmentation using SMOTE in order to balance the training dataset, which is heavily weighted towards classes 8 and 4. This did not improve the log-loss leaderboard score (which was the aim of the competition) as the test set most likely included these class biases too. However it only slightly impacted the validation score, which in a real-world scenario may be desirable as the model will generalise better.

* Use of a 'weighted average' metaclassifier, where model prediction weights are selected by Nelder-Mean minimisation. This led to some overfitting.

* Algorithmically aggregating the pixel predictions (as explained above). For example, applying a multiplying factor to the most confident pixel predictions led to a 0.01 improvement in log-loss.

* 3D CNN where the additional tensor dimension was represented by the time-series images. Performed similarly to 2D CNN without further optimisation.

* Automatic outlier detection using an Isolation Forest algorithm. Degraded performance.

* Feature selection both using univariate and model-based selection (`SelectKBest` on mutual information and `SelectFromModel` on Random Forest from scikit-learn). These were both shown to improve validation performance but not submitted to leaderboard.


Future improvements could be made in the following areas:

* Substantial improvement of ensembling methods and metaclassifiers, along with ensembling models using different feature sets eg. pixel-based, object-based and NN methods could all be ensembled.

* Gridsearch of model hyperparameters.

* Improvement to CNN architecture and tuning. Use larger resized 2D images. Explore the use of transfer learning and available pre-trained nets, albeit this may not be optimal for this task.

* Improve treatment of outliers, such as clouds and field boundaries - which may include fences, roads and gates.

* Explore PCA of the multispectral images.



---

W. Parfitt, 17th September 2019
