from sklearn import base

### ENSEMBLE MODELS

class ModelEnsemble(base.BaseEstimator, base.TransformerMixin):
    '''

    Fits and makes predictions for several classifiers and stacks these using one or more metaclassifiers.

    -----------

    1) A series of (level 1) classifiers and (level 2) metaclassifiers are specified by the user.

    2) When .fit is called, the 'fit' data (labelled training data: X_fit, y_fit) is split equally into two folds.

    3) Both folds are used in turn to train and make class probability predictions using each (level 1) classifier.

        --> The trained classifier models (.fit_clfs), their out-of-fold scores (.fit_clf_scores) and
        predictions (.fit_clf_predictions) are stored in their respective instance attributes.

    4) The resulting predictions, which account for all observations in the fit data, are combined and used to
        train the (level 2) metaclassifiers.

        --> The trained metaclassifier models (.fit_mclfs), their training scores (.fit_mclf_scores) and
        predictions (.fit_mclf_predictions) are stored in their respective instance attributes.

    5) When .predict is called, the fit data is used again to re-train all (level 1) classifiers - this time
        using the full dataset.

    6) Predictions from each (level 1) classifier are generated using the 'predict' data
        (unlabelled test data: X_predict).

        --> The trained classifier models (.predict_clfs) and their predictions (.predict_clf_predictions)
        are stored in their respective instance attributes.

    7) The predictions from the (level 1) classifiers are combined and used as features to make ensemble
        predictions with the (level 2) metaclassifiers that were trained in the .fit method.

        --> The metaclassifier predictions (.predict_mclf_predictions) are stored in the instance
        attribute and returned.

    ----------

    Methods:

        __init__(self, clfs=[], mclfs=[])

        fit(self, X_fit, y_fit = train_data)

        predict(self, X_predict)


    Arguments:

        clfs = (level 1) classifiers; dict {name, model}.
                Eg. {'RF':RandomForestClassifier(), 'LogReg': LogisticRegressor(solver='lbfgs')}
                Note: special behavior invoked for XGBoost models and must be named 'XGB'

        mclfs = (level 2) metaclassifiers; dict {name, model}.

        X_fit = training data features for fitting; dataframe
        y_fit = training labels for fitting; dataframe
                Note: can be any dataframe, as long as it contains 'Field_Id' and 'Crop_Id_Ne' columns

        X_predict = test data features for prediction; dataframe


    Instance Attributes:

        .fit_clfs
        .fit_mclfs
        .predict_clfs

        .X_fit
        .y_fit
        .fit_clf_scores
        .fit_clf_predictions
        .fit_mclf_scores
        .fit_mclf_predictions

        .X_predict
        .predict_clf_predictions
        .predict_mclf_predictions


    '''

    def __init__(self, clfs=[], mclfs=[]):

        # Initialise classifier and metaclassifier models
        self.fit_clfs = clfs.copy()
        self.fit_mclfs = mclfs
        self.predict_clfs = clfs.copy()

    def fit(self, X_fit, y_fit=None):

        import numpy as np
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        # Stores full fit data ready for predict method to be called
        self.X_fit = X_fit
        self.y_fit = y_fit

        # Split fit data into two folds
        all_fields = X_fit['Field_Id'].unique().tolist()

        fold_1_fields, fold_2_fields = train_test_split(all_fields, test_size=0.5, random_state=42)

        X_fold_1 = X_fit[X_fit['Field_Id'].isin(fold_1_fields)].drop('Field_Id', axis=1)
        X_fold_2 = X_fit[X_fit['Field_Id'].isin(fold_2_fields)].drop('Field_Id', axis=1)

        y_fold_1 = y_fit['Crop_Id_Ne'][y_fit['Field_Id'].isin(fold_1_fields)]
        y_fold_2 = y_fit['Crop_Id_Ne'][y_fit['Field_Id'].isin(fold_2_fields)]

        # Define dicts to store classifier scores and predictions

        self.fit_clf_scores = dict()
        self.fit_clf_predictions = dict()

        self.fit_mclf_scores = dict()
        self.fit_mclf_predictions = dict()

        print("Fitting classifiers... \n")
        print("Classifier %-*s  Fold 1 Score %-*s  Fold 2 Score" % (15, '', 10, ''))
        print("---------------------------------------------------------------------")

        # For each classifier, fit, make predictions and score model on each fold respectively

        for name, model in self.fit_clfs.items():

            clf_score = []
            clf_prediction = []

            for X_train, X_test, y_train, y_test in [(X_fold_1, X_fold_2, y_fold_1, y_fold_2),
                                                     (X_fold_2, X_fold_1, y_fold_2, y_fold_1)]:

                if name == 'XGB':

                    eval_set = [(X_train, y_train), (X_test, y_test)]
                    eval_metric = ["mlogloss"]
                    model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set,
                              verbose=False, early_stopping_rounds=10)

                else:
                    model.fit(X_train, y_train)

                fold_prediction = model.predict_proba(X_test)

                clf_score.append(log_loss(y_test, fold_prediction))
                clf_prediction.append(fold_prediction)

            print("%-*s  %-*.3f  %.3f" % (30, name, 23, clf_score[0], clf_score[1]))

            self.fit_clf_scores[name] = clf_score
            self.fit_clf_predictions[name] = np.vstack(clf_prediction)

        # For each metaclassifier, fit, make predictions and score using all classifier predictions

        print("\n\nFitting metaclassifiers... \n")
        print("Meta-Classifier %-*s  Score" % (14, ''))
        print("----------------------------------------------")

        clf_predictions_combined = np.hstack([value for value in self.fit_clf_predictions.values()])

        for name, model in self.fit_mclfs.items():
            model.fit(clf_predictions_combined, self.y_fit['Crop_Id_Ne'])

            mclf_prediction = model.predict_proba(clf_predictions_combined)
            mclf_score = log_loss(self.y_fit['Crop_Id_Ne'], mclf_prediction)

            print("%-*s  %-*.3f" % (30, name, 23, mclf_score))

            self.fit_mclf_scores[name] = mclf_score
            self.fit_mclf_predictions[name] = mclf_prediction

        print("\n\nComplete!")

        return self

    def predict(self, X_predict):

        import numpy as np

        self.X_predict = X_predict

        X_train = self.X_fit.drop('Field_Id', axis=1)
        y_train = self.y_fit['Crop_Id_Ne']

        X_test = self.X_predict.drop('Field_Id', axis=1)

        # Define dicts to store classifier and metaclassifier predictions

        self.predict_clf_predictions = dict()
        self.predict_mclf_predictions = dict()

        # For each classifier fit using full 'fit' dataset and predict using 'predict' dataset

        print("Re-fitting and predicting classifiers... \n")
        print("Classifier %-*s  Status" % (18, ''))
        print("----------------------------------------------")

        for name, model in self.predict_clfs.items():

            if name == 'XGB':

                eval_set = [(X_train, y_train)]
                eval_metric = ["mlogloss"]
                model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set,
                          verbose=False, early_stopping_rounds=10)

            else:
                model.fit(X_train, y_train)

            clf_prediction = model.predict_proba(X_test)

            self.predict_clf_predictions[name] = clf_prediction

            print("%-*s  %s" % (30, name, 'Complete'))

        # For each metaclassifier, make predictions using 'predict' dataset classifier predictions

        print("\n\nPredicting metaclassifiers... \n")
        print("Meta-Classifier %-*s  Status" % (14, ''))
        print("----------------------------------------------")

        clf_predictions_combined = np.hstack([value for value in self.predict_clf_predictions.values()])

        for name, model in self.fit_mclfs.items():
            mclf_prediction = model.predict_proba(clf_predictions_combined)

            self.predict_mclf_predictions[name] = mclf_prediction

            print("%-*s  %s" % (30, name, 'Complete'))

        print("\n\nComplete!")

        return self.predict_mclf_predictions


### MAKE SUBMISSIONS

def make_submission(predictions, test_data, tag = 'Submission'):

    '''

    Takes ensemble predictions as dict {model_name, predictions} and generates .csv files for submission

    -----------

    Arguments:

        predictions = ensemble predictions as dict {model_name, predictions}
        tag = optional file identifier (default = 'Submission')


    '''

    import pandas as pd

    for name, preds in predictions.items():
        submission = pd.DataFrame(preds, index=test_data['Field_Id'],
                                  columns=['crop_id_1', 'crop_id_2', 'crop_id_3', 'crop_id_4',
                                           'crop_id_5', 'crop_id_6', 'crop_id_7', 'crop_id_8', 'crop_id_9'])

        submission.to_csv('submissions/' + tag + '_' + name + '_' + datetime.now().strftime('_%m-%d_%H%M') + '.csv')

    print(len(predictions), 'submissions generated')


### PIXEL TO OBJECT

class PixelToObject():

    '''

    Used for pixel-based models, transforms predictions from individual pixel predictions to
    object (field) predictions by taking the mean pixel prediction for that object

    (NOTE: this method may be modified to be more complex ie. apply Laplace smoothing to predictions
    for objects where the pixel predictions are contradictory)

    -----------

    - Accepts ModelEnsemble object which has had .fit (and optionally .predict) methods applied

    - Returns an analog to the ModelEnsemble object with the same instance attributes,
        albeit transformed to object predictions and scores as opposed to those for pixels

    -----------

    Methods:

        __init__(self)

        fit(self)

        transform(self, ensemble, train_data)


    Arguments:

        ensemble = input ModelEnsemble object which has had .fit (and optionally .predict) applied

        train_data = dataset containing training set label 'Crop_Id_Ne' column, used for scoring


    Instance Attributes:

        .X_fit
        .y_fit
        .fit_clf_scores
        .fit_clf_predictions
        .fit_mclf_scores
        .fit_mclf_predictions

        .X_predict
        .predict_clf_predictions
        .predict_mclf_predictions

    '''

    def __init__(self):

        self.ensemble = None

    def _average_field_preds(self, pixel_preds, field_id):

        import pandas as pd

        pixel_preds = pd.DataFrame(pixel_preds)
        pixel_preds['Field_Id'] = field_id
        field_preds = pixel_preds.groupby(['Field_Id']).mean()

        return field_preds.values

    def fit(self):

        return self

    def transform(self, ensemble, train_data):

        from sklearn.metrics import log_loss
        import pandas as pd

        self.X_fit = ensemble.X_fit
        self.y_fit = ensemble.y_fit

        field_id = self.X_fit['Field_Id']

        self.fit_clf_scores = dict()
        self.fit_clf_predictions = dict()

        self.fit_mclf_scores = dict()
        self.fit_mclf_predictions = dict()

        self.predict_clf_predictions = dict()
        self.predict_mclf_predictions = dict()

        print("Transforming 'fit' classifier pixel predictions to object predictions... \n")
        print("Classifier %-*s  Object Score" % (19, ''))
        print("----------------------------------------------")

        for name, pixel_predictions in ensemble.fit_clf_predictions.items():
            prediction = self._average_field_preds(pixel_predictions, field_id)
            score = log_loss(train_data['Crop_Id_Ne'], prediction)

            print("%-*s  %-*.3f" % (30, name, 23, score))

            self.fit_clf_scores[name] = score
            self.fit_clf_predictions[name] = prediction

        print("\n\nTransforming 'fit' metaclassifier pixel predictions to object predictions... \n")
        print("Meta-Classifier %-*s  Object Score" % (14, ''))
        print("----------------------------------------------")

        for name, pixel_predictions in ensemble.fit_mclf_predictions.items():
            prediction = self._average_field_preds(pixel_predictions, field_id)
            score = log_loss(train_data['Crop_Id_Ne'], prediction)

            print("%-*s  %-*.3f" % (30, name, 23, score))

            self.fit_mclf_predictions[name] = prediction
            self.fit_mclf_scores[name] = score

        # If .predict method has also been called on input ensemble transform these too
        try:

            self.X_predict = ensemble.X_predict

            print("\n\nTransforming 'predict' classifier pixel predictions to object predictions... \n")
            print("Classifier %-*s  Status" % (19, ''))
            print("----------------------------------------------")

            for name, pixel_predictions in ensemble.predict_clf_predictions.items():
                prediction = self._average_field_preds(pixel_predictions, field_id)

                print("%-*s  %s" % (30, name, 'Complete'))

                self.predict_clf_predictions[name] = prediction

            print("\n\nTransforming 'predict' metaclassifier pixel predictions to object predictions... \n")
            print("Meta-Classifier %-*s  Status" % (14, ''))
            print("----------------------------------------------")

            for name, pixel_predictions in ensemble.predict_mclf_predictions.items():
                prediction = self._average_field_preds(pixel_predictions, field_id)

                print("%-*s  %s" % (30, name, 'Complete'))

                self.predict_mclf_predictions[name] = prediction

            return self.predict_mclf_predictions

        except:

            return self.predict_clf_predictions
