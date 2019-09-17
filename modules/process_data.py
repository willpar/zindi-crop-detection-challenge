from sklearn import base


### SELECT FEATURES

class SelectFeatures(base.BaseEstimator, base.TransformerMixin):

    """

    Selects specific columns from the input datasets based on subtractive keyword rules

    ----------

    - Accepts pandas dataframe, returns pandas dataframe.

    1) Finds all columns in input dataset dataframes
    2) Selects only those columns that match keywords in 'keep_cols'
    3) Of these columns, removes any columns that match keywords in 'drop_cols'

    eg. Input dataframe may contain columns ['Field_Id',  'NDVI_0131', 'NDVI_0819']
    and only column ['NDVI_0131'] is to be selected:

    keep_cols = ['NDVI'] --> will match ['NDVI_0131', 'NDVI_0819']
    followed by:
    drop_cols = ['0819'] --> will result in ['NDVI_0131']

    ----------

    Methods:

        __init__(self, keep_cols=[], drop_cols=[])

        fit(self)

        transform(self, datasets=[])


    Arguments:

    keep_cols = subset of all columns to keep
    drop_cols = subset of kept columns to drop
    datasets = list of feature datasets to select from


    """

    def __init__(self, keep_cols=None, drop_cols=[]):

        self.keep_cols = keep_cols
        self.drop_cols = drop_cols


    def fit(self):

        return self


    def transform(self, datasets=[]):

        import pandas as pd

        self.cols = []

        data_subset = [datasets[0]['Field_Id']]

        for data in datasets:

            cols = []

            if self.keep_cols == None:
                cols.extend(data.columns)
            else:
                for keep in self.keep_cols:
                    cols.extend([col for col in data.columns if keep in col])

            for drop in self.drop_cols:
                cols = [col for col in cols if drop not in col]

            self.cols.extend(cols)

            data_subset.append(data[cols])

        print('Selected ', str(len(self.cols)), ' columns \nUse .cols attribute to see all columns\n')

        data_subset = pd.concat(data_subset, axis=1)

        return data_subset


### SCALE FEATURES

class Scale(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, keep_cols=None, drop_cols=[], scaler_type='minmax'):

        self.keep_cols = keep_cols
        self.drop_cols = drop_cols
        self.scaler_type = scaler_type

    def fit(self, data):

        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        cols = []

        if self.keep_cols == None:
            cols = [col for col in data.columns]
        else:
            for keep in self.keep_cols:
                cols.extend([col for col in data.columns if keep in str(col)])

        for drop in self.drop_cols:
            cols = [col for col in cols if drop not in col]

        self.cols = cols

        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler().fit(data[self.cols])

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler().fit(data[self.cols])

        return self

    def transform(self, data):

        scaler = self.scaler

        data[self.cols] = scaler.transform(data[self.cols])

        print('Scaling ', str(len(self.cols)), ' columns \nUse .cols attribute to see all columns\n')

        return data


### ONEHOT ENCODE FEATURES

class OneHot(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, keep_cols=None, drop_cols=[]):

        self.keep_cols = keep_cols
        self.drop_cols = drop_cols

    def fit(self, data):

        from sklearn.preprocessing import OneHotEncoder

        onehot_cols = []

        if self.keep_cols == None:
            onehot_cols = [col for col in data.columns]
        else:
            for keep in self.keep_cols:
                onehot_cols.extend([col for col in data.columns if keep in str(col)])

        for drop in self.drop_cols:
            onehot_cols = [col for col in onehot_cols if drop not in col]

        self.cols = onehot_cols #For consistency with other methods
        self.onehot_cols = onehot_cols
        self.non_onehot_cols = list(set(data.columns) - set(onehot_cols))

        self.encoder_dict = {}

        for col in self.onehot_cols:
            encoder = OneHotEncoder(categories='auto', sparse=False)
            encoder.fit(data[col].values.reshape(-1, 1))

            self.encoder_dict[col] = encoder

        return self

    def transform(self, data):

        import pandas as pd

        new_features = []
        new_features.append(data[self.non_onehot_cols])

        for col in self.onehot_cols:
            encoder = self.encoder_dict[col]

            onehots = encoder.transform(data[col].values.reshape(-1, 1))

            features = pd.DataFrame(onehots)

            features = features.add_prefix(col + '_')

            new_features.append(features)

        print('Encoding ', str(len(self.cols)), ' columns \nUse .cols attribute to see all columns\n')

        new_features = pd.concat(new_features, axis=1)

        return new_features
