import os
import pandas as pd
import numpy as np


class clean_data(object):
    def __init__(self):
        pass
    
    def get_knum(self, k):
        k = str(k)
        return k.split('A')[0].split('B')[0].split('C')[0].split('-')[0].strip()
    
    

    
    
# get pipeline
# pipeline = get_pipeline(num_attribs, cat_attribs)
# xx = pipeline.fit_transform(riv)


# # get full feature names
# ## categorical feature names
# cat_pipeline = (pipeline.named_steps['num_cat_pipe'].transformers[1])[1].named_steps['ohe']
# cat_pipelines.fit(riv[cat_attribs])
# f_names = cat_pipelines.named_steps['ohe'].get_feature_names()

def get_featureName(cat_attribs, f_names):
    '''
    get feature names from OneHotEncoder transformation
    '''
    def get_index(f):
        return int(f.split('_')[0].replace('x', '')), f.split('_')[1]

    feature_names = list()
    for fn in f_names: 
        ind = get_index(fn)[0]
        feature_names.append(cat_attribs[ind] + '_' + get_index(fn)[1])    
    return feature_names 

# cat_features = get_featureName(cat_attribs, f_names)    



class feature_engineering(object):
    def __init__(self):
        self.VUR_max_completed = False

    def delta_VUR(self, data):
        if not self.VUR_max_completed:
            data = self.VUR_max(data)
        data['VUR_delta'] = data['VR_1301'] - data['VR_0101']
        return data


    def VUR_max(self, data):
        data['VR_0101'] = data[['VR_3L_0101','VR_3R_0101']].apply(lambda x: max(x[0], x[1]), axis = 1)
        data['VR_1301'] = data[['VR_3L_1301','VR_3R_1301']].apply(lambda x: max(x[0], x[1]), axis = 1)
        data['VR_0101_cat'] = np.where(data['VR_0101'] < 3, 'low', 'high')
        data['VR_1301_cat'] = np.where(data['VR_1301'] < 3, 'low', 'high')
        self.VUR_max_completed = True
        return data

    def UTI_rate(self, data):
        data['NUM_UTI01_rate'] = data['NUM_UTI01']/data['AGE0101']
        return data


    def __call__(self, data):
        data = self.VUR_max(data)
        data = self.delta_VUR(data)
        data = self.UTI_rate(data)
        return data
                


# data loading
# DATA_DIR = '../raw_data'

def load_data(fname, DATA_DIR):
    in_path = os.path.join(DATA_DIR, "{}".format(fname))
    df = pd.read_csv(in_path)
    print ("{} loaded".format(fname))
    return df
    
    
def get_datasets(toolkit, in_path):
    # toolkits
    clean_data = toolkit[0]()
    get_knum = clean_data.get_knum
    feature_engineering = toolkit[1]()

    # DCHS1 dataset
    dchs1 = load_data('DCHS1.csv', in_path)
    knum = dchs1['KNUMBER'].apply(lambda x: get_knum(x))
    knum.orig = dchs1['Knum Original'].apply(lambda x: get_knum(x))
    dchs1['sampleID'] = np.where(knum==knum.orig, knum, np.where(knum.orig != 'nan', knum.orig, knum))

    # riv dataset
    riv = load_data('RV111201 A1A3.csv', in_path)
    knum = riv['KNUMBER'].apply(lambda x: get_knum(x))
    knum.orig = riv['Knum Original'].apply(lambda x: get_knum(x))
    riv['sampleID'] = np.where(knum == knum.orig, knum, np.where(knum.orig != 'nan', knum.orig, knum))
    riv['RACE0102'] = riv.RACE0102.astype('str')
    
    # snp dataset
    snp = load_data('RIVUR SNPs.csv', in_path)
    name = list(snp.columns)
    col = name[name.index('RUID / Sample ID')::]
    snp = snp.loc[:, col]
        
    # RIVUR patient info dataset from a1a3 file 
    name = list(riv.columns)
    col = name[1:name.index('Knum Original')]
    col.insert(0, 'KNUMBER')
    patient_info = riv[col]
    
    # a1a3 dataset
    col = name[name.index('KNUMBER')::]
    a1a3 = riv[col]
    
    # a1a3 control dataset
    a1a3_control = load_data('Copy of DefA1A3 CNV HW resuts with patient information.csv', in_path)
    a1a3_control.rename(columns = {a1a3_control.columns.tolist()[0]: 'sampleID'}, inplace = True)
    a1a3_ctl_info = a1a3_control.drop(['DefA1A3 CN'], axis=1)
    a1a3_ctl_genotype = a1a3_control.iloc[:, [0,1]]
    
    # dmbt1 dataset
    dmbt1_ = load_data('OHIO_after CNVtools.csv', in_path)
    name = list(dmbt1_.columns)
    col = name[name.index('CNV1'):name.index('pp_CNV2')]
    col.insert(0, 'Sample #')
    dmbt1 = dmbt1_[col]
    
    ##### DMBT1_all ######
    dmbt1_all_ = load_data("OHIO_after CNVtools_dmbt1_all.csv", in_path)
    name = list(dmbt1_all_.columns)
    col = name[name.index('CNV1'):name.index('pp_CNV2')]
    col.insert(0, 'Sample #')
    col.insert(1, 'Ethnicity')
    dmbt1_all = dmbt1_all_[col]
    dmbt1_all.drop_duplicates('Sample #', keep = 'first', inplace = True) # remove duplicates by keeping the first occurrence.
    
    # additional sample info dataset (rivur + control)
    col = name[name.index('Sample #'):name.index('CNV1')]
    sample_info = dmbt1_.loc[:, col]

    # data cleansing
    a1a3.rename(columns = {a1a3.columns.tolist()[0]: 'sampleID'}, inplace = True)
    snp.rename(columns = {snp.columns.tolist()[0]: 'sampleID'}, inplace = True)
    dmbt1.rename(columns = {dmbt1.columns.tolist()[0]: 'sampleID'}, inplace = True)
    dmbt1_all.rename(columns = {dmbt1_all.columns.tolist()[0]: 'sampleID'}, inplace = True)
      
    # merge data for rivur cohort
    x = pd.merge(riv, snp, how = 'left', on = 'sampleID')
    x = pd.merge(x, dchs1, how = 'left', on = 'sampleID')
    riv_genotype = pd.merge(x, dmbt1, how = 'left', on = 'sampleID')
    
    
    # feature engineering
    riv_genotype = feature_engineering(riv_genotype)


    # rename
    col = ['sampleID', 'DEFA1/A3 CN', 'A3 homozygous deficiency yes,1; no, 0',
       'A1 homozygous deficiency yes,1; no, 0', 'actual_CNV1_new_nomenclature', 'actual_CNV2',  # 'actual_CNV1' 
      'RNASE7 rs1263872', 'RNASE7 rs1243469', 'CXCR1 rs3138060', 'rs2304204', 
      'rs2304206', 'rs10759931', 'UMOD_rs4293393', 'DCHS1 copy number'
      ]
    to = ['sampleID', 'a1a3.cn', 'A3.deficiency',
                    'A1.deficiency', 'dmbt1_CNV1', 'dmbt1_CNV2',
                    'rs1263872', 'rs1243469', 'rs3138060', 'rs2304204', 
                    'rs2304206', 'rs10759931', 'rs4293393', 'DCHS1_CNV'
                   ]
    
    riv_genotype.rename(index = str, columns = {a : b for a, b in zip(col, to)}, inplace = True)
    
    return riv_genotype, (dchs1, riv, dmbt1_all, dmbt1, a1a3, snp, patient_info, sample_info, a1a3_ctl_info, a1a3_ctl_genotype)