from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import tensorflow as tf 



class myOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features = None): # no *args or **kargs  , features=None
        self.features = features
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        X = X.copy()
        df = pd.get_dummies(X,drop_first=True) 
        return df.values        
    

def get_pipeline(num_attribs, cat_attribs):
    '''
    function to create pipeline using standarded sklearn API
    '''
    num_pipeline = Pipeline([
    #     ('Data2matrix', Data2matrix()),
    #     ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy='mean')), 
        ('std_scalar', StandardScaler()) ])


    cat_pipeline = Pipeline([
    #         ('selector', DataFrameSelector(cat_attribs)),
            ('imputer', SimpleImputer(strategy='constant')),  # most_frequent
            ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
        ])


    num_cat_pipeline = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_attribs),
        ('cat_pipeline', cat_pipeline, cat_attribs),
    ], remainder = 'drop' )


    data_prep_pipeline = Pipeline([
        ('num_cat_pipe', num_cat_pipeline)        
    ]) 
    
    return data_prep_pipeline



class feature_init_fn(object):
    def __init__(self, full_feature):
        self.full_feature = full_feature
        self.num_attribs = [ 'BBD0102', 'BBD1302', 'VR_0101', 'VR_1301',
                    'NUMORG0101', 'AGE0101',   'PRIORUTI0101',  'NUM_UTI01',
                    'Highest_reflux0101',  'Highest_reflux1301',  'a1a3.cn', 
                    'dmbt1_CNV1', 'dmbt1_CNV2', 'DCHS1_CNV', 'VUR_delta' ,
                    'NUM_UTI01_rate'

                    # VR_3L_0101', 'VR_3R_0101', 'VR_3L_1301', 'VR_3R_1301'
                    ]
        self.cat_attribs = ['CHR_CONST0102', 'CHR_CONST1302', 'VR_0101_cat', 'VR_1301_cat',    #  'SCAR_NEW04',
                  'SEX0101','ETHNIC0101', 'RACE0102', 
                   'TXGROUP' ,   'UTI_TYPE0103',
                   'VUR_LATERAL0101', 
                   'VUR_LATERAL1301' ,
                    'A3.deficiency',
                    'A1.deficiency', 
                    'rs1263872', 'rs1243469', 'rs3138060', 'rs2304204', 
                    'rs2304206', 'rs10759931', 'rs4293393'         
                  ]
        
        if self.full_feature:
            self.num_attribs = ['DVQSCORE0104', 'DVQSCORE1304', 'BBD0102', 'BBD1302', 'VR_0101', 'VR_1301', 
                       'NUMORG0101', 'AGE0101',  'PRIORUTI0101', 'NUM_UTI01',
                       'Highest_reflux0101', 'VR_3L_0101', 'VR_3R_0101', 'Highest_reflux1301',
                       'VR_3L_1301', 'VR_3R_1301', 'a1a3.cn', 
                    'dmbt1_CNV1', 'dmbt1_CNV2', 'DCHS1_CNV', 'VUR_delta', 'NUM_UTI01_rate'
                      ]
            self.cat_attribs = ['CHR_CONST0102', 'CHR_CONST1302', 'PORG_SPECIES0101',  'VR_0101_cat', 'VR_1301_cat',
                      'SEX0101','ETHNIC0101', 'RACE0102', 'SCAR_NEW04',
                       'TXGROUP' ,   'UTI_TYPE0103',
                      'Worst_scarring0101',  'Worst_scarring_side0101', 
                       'Worst_scarring1301',  'Worst_scarring_side1301',
                       'DM6A_01', 'DM6B_01', 'DM6A_13', 'DM6B_13', 
                        'VUR_LATERAL0101', 'HIGHEST_REFLUX_SIDE0101',
                       'VUR_LATERAL1301' , 'HIGHEST_REFLUX_SIDE1301',
                    'A3.deficiency',
                    'A1.deficiency', 
                    'rs1263872', 'rs1243469', 'rs3138060', 'rs2304204', 
                    'rs2304206', 'rs10759931', 'rs4293393'         
                      ]

    def __call__(self, target):
        if target == 'NUM_UTI01':
            self.num_attribs.remove('NUM_UTI01_rate')
        try:
            self.cat_attribs.remove(target)
        except:
            self.cat_attribs
        try:
            self.num_attribs.remove(target)
        except:
            self.num_attribs    
            
        return self.cat_attribs, self.num_attribs







def get_X_y(riv, target, feature_init_fn = feature_init_fn, standardization = True, full_feature = True):
    
    '''
    Get X - features, y - target for downstream machine learning process
    using customized data processing pipeline
    '''
    
    feature = feature_init_fn(full_feature = full_feature)
    cat_attribs, num_attribs = feature(target)
    
    
    # feature engineering
    # riv['VR_0101'] = riv[['VR_3L_0101','VR_3R_0101']].apply(lambda x: max(x[0], x[1]), axis = 1)
    # riv['VR_1301'] = riv[['VR_3L_1301','VR_3R_1301']].apply(lambda x: max(x[0], x[1]), axis = 1)
    # riv['VR_0101_cat'] = np.where(riv['VR_0101'] < 3, 'low', 'high')
    # riv['VR_1301_cat'] = np.where(riv['VR_1301'] < 3, 'low', 'high')
    
    
    # for categorical features
    imputer = SimpleImputer(strategy='constant')
    cat = imputer.fit_transform(riv[cat_attribs])
    cat = pd.DataFrame(cat, columns = cat_attribs)
    cat = pd.get_dummies(cat, drop_first = False)
    
    
    # for numerical features
    imputer = SimpleImputer(strategy = 'mean')
    std_scalar = StandardScaler()
    min_max = MinMaxScaler()
    num = imputer.fit_transform(riv[num_attribs])
    if standardization:
        num = std_scalar.fit_transform(num)
    num = pd.DataFrame(num, columns = num_attribs)
      
    # combine 
    df = pd.concat([num, cat], axis = 1)
    
    
    # Target
    y = riv[target].replace({0: 'No', 1: 'No', 2: 'Yes', 3: 'Yes', 4: 'Yes'})
    return df, y 




def get_X_y_tensorflow(data, target):
	label = data.pop(target)
	feature = data

	dataset = tf.data.Dataset.from_tensor_slices((data, label))

	return dataset


