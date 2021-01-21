import pandas as pd
import numpy as np
import re
from tqdm.notebook import tqdm
import random
import sklearn.metrics
from sklearn.pipeline import Pipeline

# For XGBoost Regression and Classification
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, f1_score, r2_score, mean_absolute_error

import catboost

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import lightgbm

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor



class ModelsParameters:
    def __init__(self, dictionary_params):
        self.dictionary_params = dictionary_params

    """ I need this function for having all keys for the coming functions """
## Functions for creating a dictionary by simply inputting values of the params for
## each type of estimator

    # Create dictionary with all params of RandomForest
    def random_forest_params(
        n_estimators=[100],  # The number of trees in the forest.
        criterion=['mse'],  # {“mse”, “mae”}, default=”mse”. The function to measure the quality of a split.
        max_depth=[None],  # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
        min_samples_split=[2],  # The minimum number of samples required to split an internal node
        min_samples_leaf=[1],  # The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf=[0.0],  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
        max_features=['auto'],  # {“auto”, “sqrt”, “log2”}, int or float, default=”auto” The number of features to consider when looking for the best split. If auto, == n_features [if not so many]
        max_leaf_nodes=[None],  # pruning? [FIX]
        min_impurity_decrease=[0.0],
        min_impurity_split=[None],
        bootstrap=[True],
        oob_score=[False],  # whether to use out-of-bag samples to estimate the R^2 on unseen data [should be true? FIX but then it will be less comparable]
        n_jobs=[None],
        random_state=[None],
        verbose=[0],
        warm_start=[False],
        ccp_alpha=[0.0],  # Complexity parameter used for Minimal Cost-Complexity Pruning [FIX]. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
        max_samples=[None],  # bootstrap is True, the number of samples to draw from X to train each base estimator. If None, == to X.shape[0]
    ):

        params_dict = {
            'n_estimators': [n_estimators][0],
            'criterion': [criterion][0],
            'max_depth': [max_depth][0],
            'min_samples_split': [min_samples_split][0],
            'min_samples_leaf': [min_samples_leaf][0],
            'min_weight_fraction_leaf': [min_weight_fraction_leaf][0],
            'max_features': [max_features][0],
            'max_leaf_nodes': [max_leaf_nodes][0],
            'min_impurity_decrease': [min_impurity_decrease][0],
            'min_impurity_split': [min_impurity_split][0],
            'bootstrap': [bootstrap][0],
            'oob_score': [oob_score][0],
            'n_jobs': [n_jobs][0],
            'random_state': [random_state][0],
            'verbose': [verbose][0],
            'warm_start': [warm_start][0],
            'ccp_alpha': [ccp_alpha][0],
            'max_samples': [max_samples][0],
        }

        return params_dict



    def rf_params_pipeline(self, existing_prefix='', prefix_to_add='rf__'):
        params_dict = {
        prefix_to_add+'n_estimators': self.dictionary_params[existing_prefix+'n_estimators'],
        prefix_to_add+'criterion': self.dictionary_params[existing_prefix+'criterion'],
        prefix_to_add+'max_depth': self.dictionary_params[existing_prefix+'max_depth'],
        prefix_to_add+'min_samples_split': self.dictionary_params[existing_prefix+'min_samples_split'],
        prefix_to_add+'min_samples_leaf': self.dictionary_params[existing_prefix+'min_samples_leaf'],
        prefix_to_add+'min_weight_fraction_leaf': self.dictionary_params[existing_prefix+'min_weight_fraction_leaf'],
        prefix_to_add+'max_features': self.dictionary_params[existing_prefix+'max_features'],
        prefix_to_add+'max_leaf_nodes': self.dictionary_params[existing_prefix+'max_leaf_nodes'],
        prefix_to_add+'min_impurity_decrease': self.dictionary_params[existing_prefix+'min_impurity_decrease'],
        prefix_to_add+'min_impurity_split': self.dictionary_params[existing_prefix+'min_impurity_split'],
        prefix_to_add+'bootstrap': self.dictionary_params[existing_prefix+'bootstrap'],
        prefix_to_add+'oob_score': self.dictionary_params[existing_prefix+'oob_score'],
        prefix_to_add+'n_jobs': self.dictionary_params[existing_prefix+'n_jobs'],
        prefix_to_add+'random_state': self.dictionary_params[existing_prefix+'random_state'],
        prefix_to_add+'verbose': self.dictionary_params[existing_prefix+'verbose'],
        prefix_to_add+'warm_start': self.dictionary_params[existing_prefix+'warm_start'],
        prefix_to_add+'ccp_alpha': self.dictionary_params[existing_prefix+'ccp_alpha'],
        prefix_to_add+'max_samples': self.dictionary_params[existing_prefix+'max_samples'],
        }
        return params_dict



    def adaboost_params(
        base_estimator=[None],
        n_estimators=[50],
        learning_rate=[1.0],
        loss=['linear'],
        random_state=[None]
    ):

        params_dict = {
            'base_estimator': [base_estimator][0],
            'n_estimators': [n_estimators][0],
            'learning_rate': [learning_rate][0],
            'loss': [loss][0],
            'random_state': [random_state][0]
        }

        return params_dict

    def ab_params_pipeline(self, existing_prefix='', prefix_to_add='ab__'):
        params_dict = {
        prefix_to_add+'base_estimator': self.dictionary_params[existing_prefix+'base_estimator'],
        prefix_to_add+'n_estimators': self.dictionary_params[existing_prefix+'n_estimators'],
        prefix_to_add+'learning_rate': self.dictionary_params[existing_prefix+'learning_rate'],
        prefix_to_add+'loss': self.dictionary_params[existing_prefix+'loss'],
        prefix_to_add+'random_state': self.dictionary_params[existing_prefix+'random_state'],
        }
        return params_dict


    def gradientboost_params(
        loss=['ls'],
        learning_rate=[0.1],
        n_estimators=[100],
        subsample=[1.0],
        criterion=['friedman_mse'],
        min_samples_split=[2],
        min_samples_leaf=[1],
        min_weight_fraction_leaf=[0.0],
        max_depth=[3],
        min_impurity_decrease=[0.0],
        # min_impurity_split=[None], # deprecated FIX
        init=[None],
        random_state=[None],
        max_features=[None],
        alpha=[0.9],
        verbose=[0],
        max_leaf_nodes=[None],
        warm_start=[False],
        presort=['deprecated'],
        validation_fraction=[0.1],
        n_iter_no_change=[None],
        tol=[0.0001],
        ccp_alpha=[0.0],
    ):

        params_dict = {
            'loss': [loss][0],
            'learning_rate': [learning_rate][0],
            'n_estimators': [n_estimators][0],
            'subsample': [subsample][0],
            'criterion': [criterion][0],
            'min_samples_split': [min_samples_split][0],
            'min_samples_leaf': [min_samples_leaf][0],
            'min_weight_fraction_leaf': [min_weight_fraction_leaf][0],
            'max_depth': [max_depth][0],
            'min_impurity_decrease': [min_impurity_decrease][0],
            # 'min_impurity_split': [min_impurity_split][0],
            'init': [init][0],
            'random_state': [random_state][0],
            'max_features': [max_features][0],
            'alpha': [alpha][0],
            'verbose': [verbose][0],
            'max_leaf_nodes': [max_leaf_nodes][0],
            'warm_start': [warm_start][0],
            'presort': [presort][0],
            'validation_fraction': [validation_fraction][0],
            'n_iter_no_change': [n_iter_no_change][0],
            'tol': [tol][0],
            'ccp_alpha': [ccp_alpha][0],
        }

        return params_dict


    def gb_params_pipeline(self, existing_prefix='', prefix_to_add='gb__'):
        params_dict = {
        prefix_to_add+'loss': self.dictionary_params[existing_prefix+'loss'],
        prefix_to_add+'learning_rate': self.dictionary_params[existing_prefix+'learning_rate'],
        prefix_to_add+'n_estimators': self.dictionary_params[existing_prefix+'n_estimators'],
        prefix_to_add+'subsample': self.dictionary_params[existing_prefix+'subsample'],
        prefix_to_add+'criterion': self.dictionary_params[existing_prefix+'criterion'],
        prefix_to_add+'min_samples_split': self.dictionary_params[existing_prefix+'min_samples_split'],
        prefix_to_add+'min_samples_leaf': self.dictionary_params[existing_prefix+'min_samples_leaf'],
        prefix_to_add+'min_weight_fraction_leaf': self.dictionary_params[existing_prefix+'min_weight_fraction_leaf'],
        prefix_to_add+'max_depth': self.dictionary_params[existing_prefix+'max_depth'],
        prefix_to_add+'min_impurity_decrease': self.dictionary_params[existing_prefix+'min_impurity_decrease'],
        # prefix_to_add+'min_impurity_split': self.dictionary_params[existing_prefix+'min_impurity_split'],
        prefix_to_add+'init': self.dictionary_params[existing_prefix+'init'],
        prefix_to_add+'random_state': self.dictionary_params[existing_prefix+'random_state'],
        prefix_to_add+'max_features': self.dictionary_params[existing_prefix+'max_features'],
        prefix_to_add+'alpha': self.dictionary_params[existing_prefix+'alpha'],
        prefix_to_add+'verbose': self.dictionary_params[existing_prefix+'verbose'],
        prefix_to_add+'max_leaf_nodes': self.dictionary_params[existing_prefix+'max_leaf_nodes'],
        prefix_to_add+'warm_start': self.dictionary_params[existing_prefix+'warm_start'],
        prefix_to_add+'presort': self.dictionary_params[existing_prefix+'presort'],
        prefix_to_add+'validation_fraction': self.dictionary_params[existing_prefix+'validation_fraction'],
        prefix_to_add+'n_iter_no_change': self.dictionary_params[existing_prefix+'n_iter_no_change'],
        prefix_to_add+'tol': self.dictionary_params[existing_prefix+'tol'],
        prefix_to_add+'ccp_alpha': self.dictionary_params[existing_prefix+'ccp_alpha'],
        }
        return params_dict


    # XGBoost
    def xgb_params(
        objective=['reg:squarederror'],
        n_estimators=[100],
        max_depth=[10],
        learning_rate=[0.3],
        verbosity=[0],
        booster=[None],  # 'gbtree'
        tree_method=['auto'],
        n_jobs=[1],
        gamma=[0],
        min_child_weight=[None],
        max_delta_step=[None],
        subsample=[None],
        colsample_bytree=[None],
        colsample_bylevel=[None],
        colsample_bynode=[None],
        reg_alpha=[0],
        reg_lambda=[0],
        scale_pos_weight=[None],
        base_score=[None],
        random_state=[random.randint(0, 500)],
        missing=[np.nan],
        num_parallel_tree=[None],
        monotone_constraints=[None],
        interaction_constraints=[None],
        importance_type=['gain']
    ):
        params_dict = {
            'objective': [objective][0],
            'n_estimators': [n_estimators][0],
            'max_depth': [max_depth][0],
            'learning_rate': [learning_rate][0],
            'verbosity': [verbosity][0],
            'booster': [booster][0],
            'tree_method': [tree_method][0],
            'n_jobs': [n_jobs][0],
            'gamma': [gamma][0],
            'min_child_weight': [min_child_weight][0],
            'max_delta_step': [max_delta_step][0],
            'subsample': [subsample][0],
            'colsample_bytree': [colsample_bytree][0],
            'colsample_bylevel': [colsample_bylevel][0],
            'colsample_bynode': [colsample_bynode][0],
            'reg_alpha': [reg_alpha][0],
            'reg_lambda': [reg_lambda][0],
            'scale_pos_weight': [scale_pos_weight][0],
            'base_score': [base_score][0],
            'random_state': [random_state][0],
            'missing': [missing][0],
            'num_parallel_tree': [num_parallel_tree][0],
            'monotone_constraints': [monotone_constraints][0],
            'interaction_constraints': [interaction_constraints][0],
            'importance_type': [importance_type][0]
        }

        return params_dict


    def xgb_params_pipeline(self, existing_prefix='', prefix_to_add='xgb__'):
        params_dict = {
        prefix_to_add+'objective': self.dictionary_params[existing_prefix+'objective'],
        prefix_to_add+'n_estimators': self.dictionary_params[existing_prefix+'n_estimators'],
        prefix_to_add+'max_depth': self.dictionary_params[existing_prefix+'max_depth'],
        prefix_to_add+'learning_rate': self.dictionary_params[existing_prefix+'learning_rate'],
        prefix_to_add+'verbosity': self.dictionary_params[existing_prefix+'verbosity'],
        prefix_to_add+'booster': self.dictionary_params[existing_prefix+'booster'],
        prefix_to_add+'tree_method': self.dictionary_params[existing_prefix+'tree_method'],
        prefix_to_add+'n_jobs': self.dictionary_params[existing_prefix+'n_jobs'],
        prefix_to_add+'gamma': self.dictionary_params[existing_prefix+'gamma'],
        prefix_to_add+'min_child_weight': self.dictionary_params[existing_prefix+'min_child_weight'],
        prefix_to_add+'max_delta_step': self.dictionary_params[existing_prefix+'max_delta_step'],
        prefix_to_add+'subsample': self.dictionary_params[existing_prefix+'subsample'],
        prefix_to_add+'colsample_bytree': self.dictionary_params[existing_prefix+'colsample_bytree'],
        prefix_to_add+'colsample_bylevel': self.dictionary_params[existing_prefix+'colsample_bylevel'],
        prefix_to_add+'colsample_bynode': self.dictionary_params[existing_prefix+'colsample_bynode'],
        prefix_to_add+'reg_alpha': self.dictionary_params[existing_prefix+'reg_alpha'],
        prefix_to_add+'reg_lambda': self.dictionary_params[existing_prefix+'reg_lambda'],
        prefix_to_add+'scale_pos_weight': self.dictionary_params[existing_prefix+'scale_pos_weight'],
        prefix_to_add+'base_score': self.dictionary_params[existing_prefix+'base_score'],
        prefix_to_add+'random_state': self.dictionary_params[existing_prefix+'random_state'],
        prefix_to_add+'missing': self.dictionary_params[existing_prefix+'missing'],
        prefix_to_add+'num_parallel_tree': self.dictionary_params[existing_prefix+'num_parallel_tree'],
        prefix_to_add+'monotone_constraints': self.dictionary_params[existing_prefix+'monotone_constraints'],
        prefix_to_add+'interaction_constraints': self.dictionary_params[existing_prefix+'interaction_constraints'],
        prefix_to_add+'importance_type': self.dictionary_params[existing_prefix+'importance_type'],
        }
        return params_dict


    # Greedy search?
    def create_spaces(self, prefix_pipeline, estimator_name):
        df = pd.DataFrame(data=[self.dictionary_params])
        params_range = {}

        for col in df.columns:
            number = 0
            string = 0      # not needed so far
            nones = 0
            trees = 0
            string_key = str(col)

            for i in df[col][0]:
                type_i = type(i)
                if (type_i == int) | (type_i == float):
                    number += 1
                elif type_i == str:  # not needed
                    string += 1
                elif i == None:  # not needed?
                    nones += 1
                elif (type_i == DecisionTreeRegressor):
                    trees += 1

            # Ranges for simple numeric values - FIX check upon them
            if (number == len(df)) & (col != prefix_pipeline+'verbose') & \
                    (col != (prefix_pipeline+'random_state')) & (col != (prefix_pipeline+'verbosity')) \
                    & (col != (prefix_pipeline+'n_jobs')) & (trees == 0) & + (col != (prefix_pipeline+'n_iter_no_change')) & \
                    (col != (prefix_pipeline+'missing')) & (col != (prefix_pipeline+'validation_fraction')):
                output = df[col][0][0]

                if estimator_name == 'RandomForest':
                    range_output, lower_output, upper_output = ModelsParameters.rf_ranges(self, col, prefix_pipeline, output)

                elif estimator_name == 'AdaBoost':
                    range_output, lower_output, upper_output = ModelsParameters.ab_ranges(self, col, prefix_pipeline, output, trees)

                elif estimator_name == 'GradientBoosting':
                    range_output, lower_output, upper_output = ModelsParameters.gb_ranges(self, col, prefix_pipeline, output)

                elif estimator_name == 'XGBoost':
                    range_output, lower_output, upper_output = ModelsParameters.xgb_ranges(self, col, prefix_pipeline, output)

                # Further Conditions on the allowed output range and append
                data_to_append = ModelsParameters.create_outputs(self, output, range_output, string_key, lower_output, upper_output)
                params_range.update(data_to_append)

            # Special Range for AdaBoost trees' max_depth
            elif (trees > 0):
                data_to_append = ModelsParameters.range_ab_decision_tree(df, self.dictionary_params, col, prefix_pipeline)
                params_range.update(data_to_append)

            # Else cases - just repeat the same value
            else:
                data_to_append = {string_key: [i]}
                params_range.update(data_to_append)

        return params_range


    def rf_ranges(self, col, prefix_pipeline, output):
        if col == prefix_pipeline+'n_estimators':
            range_output = 5
        elif col == prefix_pipeline+'max_depth':
            range_output = 3
        elif col == prefix_pipeline+'min_samples_split':
            range_output = 2
        elif col == prefix_pipeline+'min_samples_leaf':
            range_output = 1
        elif col == prefix_pipeline+'min_weight_fraction_leaf':
            range_output = 0.05
        elif col == prefix_pipeline+'max_features':
            range_output = 0
        elif col == prefix_pipeline+'max_leaf_nodes':
            range_output = 0
        elif col == prefix_pipeline+'min_impurity_decrease':
            range_output = 0.2
        elif col == prefix_pipeline+'ccp_alpha':
            range_output = 0.2
        elif col == prefix_pipeline+'max_samples':
            range_output = 0
        lower_output = output - range_output
        upper_output = output + range_output
        return range_output, lower_output, upper_output


    def ab_ranges(self, col, prefix_pipeline, output, trees):

        # FIX later: for not needed, thinking of merging with the estimator for tree
        if trees == 0:
            if col == prefix_pipeline+'n_estimators':
                range_output = 5
            elif col == prefix_pipeline+'learning_rate':
                range_output = 0.01 # FIX: is learning rate max == 1?

        else:
            pass

        lower_output = output - range_output
        upper_output = output + range_output

        return range_output, lower_output, upper_output


    def range_ab_decision_tree(df, start_params, col, prefix_pipeline): # # For AdaBoost range of base_estimator max_depth
        tree = df[col][0][0]  # not needed
        for i in start_params[col]:
            x = re.split("\=", str(i))
            y = re.split("\)", str(x[1]))[0]
            max_depth = int(str(y))
        output = sklearn.tree.DecisionTreeRegressor(max_depth=max_depth)

        if col == prefix_pipeline+'base_estimator':
            range_output = 3

        lower_output = max_depth - range_output
        upper_output = max_depth + range_output

        if (range_output != 0) & (lower_output > 0):
            data_to_append = {str(col): [
                sklearn.tree.DecisionTreeRegressor(max_depth=lower_output),
                output,
                sklearn.tree.DecisionTreeRegressor(max_depth=upper_output)
            ]}

        elif (range_output != 0) & (lower_output <= 0):
            data_to_append = {str(col): [
                output,
                sklearn.tree.DecisionTreeRegressor(max_depth=upper_output)
            ]}

        elif (range_output == 0):
            data_to_append = {str(col): [
                output
            ]}

        return data_to_append


    def gb_ranges(self, col, prefix_pipeline, output):

        if col == prefix_pipeline+'learning_rate':
            range_output = 0 # FIX: is learning rate max == 1?
        elif col == prefix_pipeline+'n_estimators':
            range_output = 5
        elif col == prefix_pipeline+'subsample':
            range_output = 0
        elif col == prefix_pipeline+'min_samples_split':
            range_output = 2
        elif col == prefix_pipeline+'min_samples_leaf':
            range_output = 1
        elif col == prefix_pipeline+'min_weight_fraction_leaf':
            range_output = 0.05
        elif col == prefix_pipeline+'max_depth':
            range_output = 3
        elif col == prefix_pipeline+'min_impurity_decrease':
            range_output = 0.2
        elif col == prefix_pipeline+'max_features':
            range_output = 0
        elif col == prefix_pipeline+'alpha':
            range_output = 0
        elif col == prefix_pipeline+'max_leaf_nodes':
            range_output = 0
        elif col == prefix_pipeline+'tol':
            range_output = 0
        elif col == prefix_pipeline+'ccp_alpha':
            range_output = 0

        lower_output = output - range_output
        upper_output = output + range_output

        return range_output, lower_output, upper_output


    def xgb_ranges(self, col, prefix_pipeline, output):

        if col == prefix_pipeline+'n_estimators':
            range_output = 5
        elif col == prefix_pipeline+'max_depth':
            range_output = 3
        elif col == prefix_pipeline+'learning_rate':
            range_output = 0 # FIX: is learning rate max == 1?
        elif col == prefix_pipeline+'gamma':
            range_output = 0
        elif col == prefix_pipeline+'min_child_weight':
            range_output = 0
        elif col == prefix_pipeline+'max_delta_stop':
            range_output = 0
        elif col == prefix_pipeline+'subsample':
            range_output = 0
        elif col == prefix_pipeline+'colsample_bytree':
            range_output = 0
        elif col == prefix_pipeline+'colsample_bylevel':
            range_output = 0
        elif col == prefix_pipeline+'colsample_bynode':
            range_output = 0
        elif col == prefix_pipeline+'reg_alpha':
            range_output = 0
        elif col == prefix_pipeline+'reg_lambda':
            range_output = 0
        elif col == prefix_pipeline+'scale_pos_weight':
            range_output = 0
        elif col == prefix_pipeline+'base_score':
            range_output = 0
        elif col == prefix_pipeline+'monotone_constraints':
            range_output = 0
        elif col == prefix_pipeline+'interaction_constraints':
            range_output = 0

        lower_output = output - range_output
        upper_output = output + range_output

        return range_output, lower_output, upper_output

##
    def create_outputs(self, output, range_output, string_key, lower_output, upper_output):
        if range_output == 0:
            data_to_append = {string_key: [
                output
            ]}

        elif (range_output != 0) & (lower_output > 0):
            data_to_append = {string_key: [
                lower_output,
                output,
                upper_output
            ]}

        # FIX could be controversial in certain instances in case you want lower bound to be 0
        elif (range_output != 0) & (lower_output == 0):
            data_to_append = {string_key: [
                output,
                upper_output
            ]}

        elif (lower_output < 0) & (output != 0):
            data_to_append = {string_key: [
                0,
                output,
                upper_output]}

        elif (lower_output < 0) & (output == 0):
            data_to_append = {string_key: [
                output,
                upper_output]}

        return data_to_append




    def best_model_pipeline(X, Y, pipeline, params_range, cv, scoring='neg_mean_squared_error'):

        optimal_model = GridSearchCV(pipeline,
                                    params_range,
                                    scoring=scoring,
                                    cv=cv,
                                    refit=True)  # when there is a list in scoring, it needs an explicit one. NP because here comes "s", not "scoring"
        print('Below are the params_range')
        print(params_range)

        result = optimal_model.fit(X, Y)
        best_params = result.best_estimator_  # result.best_params_ needed when refit=False

        dict_parameters_pipeline = {}
        for param in params_range:  # list of parameters
            dict_parameters_pipeline[str(param)] = [best_params.get_params()[str(param)]]

        print('Below are the best params')
        print(dict_parameters_pipeline)

        return result, dict_parameters_pipeline



##

    def NestedCV(X, Y, params, pipeline, prefix_pipeline=None, estimator=None, estimator_name=None,
                 NUM_TRIALS=1,  # for repeated. Note that the sample is anew every time CHECK
                 inner_n_splits=5,
                 outer_n_splits=5,
                 adaptive_grid='yes',
                 scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                 ):


        best_params = pd.DataFrame()
        df_feature_importance = pd.DataFrame()
        mse = list()
        mae = list()
        rmse = list()
        score_list = list()
        score_metric = list()

    # PROPOSAL: evaluate with metrics listed. BLOCK for metrics]
    ## WRONG FIX. it is just differnet metric at the end. Shouldn0t be in loop like this
        for s in scoring:

    # PROPOSAL: nested CV
            for i in tqdm(range(NUM_TRIALS)):
                # configure the cross-validation procedure
                cv_outer = KFold(n_splits=outer_n_splits, shuffle=True, random_state=i)

                # [BLOCK for nested CV?
                for train_ix, test_ix in cv_outer.split(X):
                    # split data
                    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
                    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]

                    # configure the cross-validation procedure
                    cv_inner = KFold(n_splits=inner_n_splits,
                                     shuffle=True,
                                     random_state=i)

                    if (pipeline==None) & (adaptive_grid=='yes'):
                        params = ModelsParameters(params).create_spaces(prefix_pipeline='', estimator=estimator) # create range RF params in format for pipeline
                        result, best_params_grid = ModelsParameters.best_model_pipeline(X_train, Y_train,
                                                                              pipeline=estimator,
                                                                              params_range=params,
                                                                              scoring=s, cv=cv_inner)
                        #print(best_params_grid)
                        print('done without pipeline')
                        # FIX test this version
                        feature_importances = pd.DataFrame(result.best_estimator_.feature_importances_.reshape(1, -1),
                                                           columns=X_train.columns)
                        df_feature_importance = df_feature_importance.append(
                            feature_importances, ignore_index=True)

                    # FIX features importance of the best model. Excluded with pipeline because there are no
                    elif (pipeline!=None) & (adaptive_grid=='yes'):
                        params = ModelsParameters(params).create_spaces(prefix_pipeline=prefix_pipeline, estimator_name=estimator_name) # create range RF params in format for pipeline
                        result, best_params_grid = ModelsParameters.best_model_pipeline(X_train, Y_train,
                                                                              pipeline=pipeline,
                                                                              params_range=params,
                                                                              scoring=s, cv=cv_inner)

                        for value in best_params_grid:
                            data_to_update = {str(value): best_params_grid[value]}
                            params.update(data_to_update)
                        #print(best_params_grid)
                        print('done one with pipeline')

                    elif (pipeline!=None) & (adaptive_grid=='no'):
                        result, best_params_grid = ModelsParameters.best_model_pipeline(X_train, Y_train,
                                                                              pipeline=pipeline,
                                                                              params_range=params,
                                                                              scoring=s, cv=cv_inner)
                        #print(best_params_grid)
                        print('done one with pipeline with no adaptive grid')

                    elif (pipeline==None) & (adaptive_grid=='no'):
                        result, best_params_grid = ModelsParameters.best_model_pipeline(X_train, Y_train,
                                                                              pipeline=estimator,
                                                                              params_range=params,
                                                                              scoring=s, cv=cv_inner)
                        #print(best_params_grid)
                        print('done without pipeline and without adaptive grid')
                        feature_importances = pd.DataFrame(result.best_estimator_.feature_importances_.reshape(1, -1),
                                                           columns=X_train.columns)
                        df_feature_importance = df_feature_importance.append(
                            feature_importances, ignore_index=True)


    # PROPOSAL: END function for metrics
                    # evaluate model on the hold out dataset (i.e. the test set)

                    # FIX the score_ variables names
                    if s == 'neg_mean_squared_error':
                        #best_model.fit(X_train, Y_train)
                        #score = best_model.score(X_test, Y_test) # for using pipeline?
                        score = mean_squared_error(Y_test, result.predict(X_test))  # result.predict(X_test): needed when refit=False
                        score_list.append(score)                                       # which quickest? FIX
                        score_metric.append('MSE')
                        print('MSE=%.3f, est=%.3f' % (score, -result.best_score_))

                    elif s == 'neg_mean_absolute_error':
                        score = mean_absolute_error(Y_test, result.predict(X_test))
                        score_list.append(score)
                        score_metric.append('MAE')
                        print('MAE=%.3f, est=%.3f' % (score, -result.best_score_))

                    elif s == 'neg_root_mean_squared_error':
                        score = mean_squared_error(Y_test, result.predict(X_test), squared=False)
                        score_list.append(score)
                        score_metric.append('RMSE')
                        print('RMSE=%.3f, est=%.3f' % (score, -result.best_score_))

                    # best parameters
                    best_params = best_params.append(best_params_grid,
                                                     ignore_index=True)

        # [BLOCK for proper output? or just not?
        best_params['Score Value'] = score_list
        best_params['Metric'] = score_metric

        return best_params, df_feature_importance



    """ FIND overall metrics/hyperparameters as sum/count/.. """

    def find_best_params(best_params):
        df = pd.DataFrame()

        # Pretransform the format of the params dataframe
        for col in best_params.loc[:, (best_params.columns != 'Score Value') & (best_params.columns != 'Metric')]: # & (best_params.columns != 'RMSE')]:
            new_list = []
            for i in range(len(best_params)):
                new_item = best_params[str(col)][i][0]
                new_list.append(new_item)
            best_params[str(col)] = new_list



        for metric in best_params['Metric'].unique():
            data = {}
            selected_metric_df = best_params.loc[best_params['Metric']==metric]
            for col in selected_metric_df.columns:
                number = 0
                string = 0
                nones = 0
                trees = 0
                string_col = str(col)

                for i in selected_metric_df[col]:
                    type_i = type(i)
                    if (type_i == int) | (type_i == float):
                        number += 1
                    elif type_i == str:
                        string += 1
                    elif i == None:
                        nones += 1
                    elif (type_i == DecisionTreeRegressor):  # for Adaboost
                        trees += 1

                # If the params are numbers, it reports the mean
                if (number == len(selected_metric_df[col])) & (nones == 0):
                    integer = 0
                    for i in selected_metric_df[col]:
                        type_i = type(i)
                        if (type_i == int):
                            output = int(selected_metric_df[col].mean())
                            data_to_append = {string_col: [output]}
                            data.update(data_to_append)

                        elif (type_i == float):
                            output = float(selected_metric_df[col].mean())
                            data_to_append = {string_col: [output]}
                            data.update(data_to_append)

                # In case of string params, the most frequent one
                elif (string == len(selected_metric_df[col])) & (nones == 0):
                    output = selected_metric_df[col].value_counts().reset_index()['index'][0]
                    data_to_append = {string_col: [output]}
                    data.update(data_to_append)

                # In case of all Nones values, the value will be None
                elif nones == len(selected_metric_df[col]):
                    data_to_append = {string_col: [None]}
                    data.update(data_to_append)

                # In case some are Nones and other not, see below FIX
                elif (nones > 0) & (nones != len(selected_metric_df[col])):
                    if (type(i) == int for i in selected_metric_df[col]):
                        if ((selected_metric_df[col].value_counts().sum() >= len(selected_metric_df)/2) == True):
                            output = int(selected_metric_df[col].value_counts().reset_index()['index'][0])
                            data_to_append = {string_col: [output]}
                            data.update(data_to_append)

                        else:
                            data_to_append = {string_col: [None]}
                            data.update(data_to_append)

                    elif (type(i) == float for i in selected_metric_df[col]):
                        if ((selected_metric_df[col].value_counts().sum() >= len(selected_metric_df)/2) == True):
                            output = float(selected_metric_df[col].value_counts().reset_index()['index'][0])
                            data_to_append = {string_col: [output]}
                            data.update(data_to_append)

                        else:
                            data_to_append = {string_col: [None]}
                            data.update(data_to_append)

            df_partial = pd.DataFrame(data)
            df_partial['Metric'] = metric
            df = df.append(df_partial, ignore_index=True)

        return df

