""" Python module for the mlcscore analysis
"""

import os, sys, pickle, joblib, matplotlib, random
from scipy import stats
from scipy.special import expit # Sigmoid function
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # Remove pandas warnings about assignments
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('grayscale')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
})
import seaborn as sns
import statsmodels.formula.api as smf
import lightgbm as lgb
import xgboost as xgb
from itertools import product
from stargazer.stargazer import Stargazer
# Load Tensorflow for Neural Networks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflowjs as tfjs



########################
# Set global variables #
########################

def set_folder(folder_name):
    global folder, folder_plots, folder_models, folder_tables, folder_html
    folder = folder_name
    folder_plots = folder+'/plots/'
    folder_models = folder+'/models/'
    folder_tables = folder+'/tables/'
    folder_html = folder+'/html/'
    for f in [folder, folder_plots, folder_models, folder_tables, folder_html]:
        if not os.path.isdir(f):
            print('Create folder '+str(f))
            os.mkdir(f)

def set_ncpu(n, tf='gpu'):
    global ncpu
    ncpu = n
    if tf=='cpu':
        print('Use CPU for TF.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

###############################
# Functions to get parameters #
###############################

def get_specparam(specification, samples='gvkeys', suffix=''):
    specparam = {}
    #####################
    # Global parameters #
    #####################

    specparam['suffix'] = suffix
    
    # Name of the return and earnings variables
    specparam['retvar'] = 'yretm'
    specparam['earningsvar'] = 'ib_l1eq'

    # Additional variables for which missing values are set to 0
    specparam['fillzeros'] = []

    # Samples
    specparam['samples'] = samples
    if samples == 'gvkeys':
        dfs= pd.read_csv(folder_models+'samples.csv')
        specparam['samples_gvkeys'] = {'train': list(dfs[dfs.train==1]['gvkey']),
                                       'validation': list(dfs[dfs.valid==1]['gvkey']),
                                       'test': list(dfs[dfs.test==1]['gvkey'])}
    elif samples == 'years':
        specparam['samples_years'] = {'train': (1963, 1994),
                                      'validation': (1995, 2008),
                                      'test': (2009, 2019)}

    ############
    # Features #
    ############

    # Replication
    features_replication = ['logeq', 'm_b', 'flev']

    # Basic
    features_basic = ['logeqdef', 'm_b', 'flev']

    # KW
    features_kw = features_basic + ['yvolatd', 'noacc', 'cfoa', 'invcycle', 'age']

    # Full features (to use in analyses)
    specparam['full_features'] = ['logeq', 'logeqdef', 'm_b', 'flev', 'yvolatd', 'noacc', 'cfoa', 'invcycle', 'age', 'problit', 'pin', 'dbas']
    #specparam['full_features'] = ['logeq', 'm_b', 'flev', 'yvolatd', 'noacc', 'cfoa', 'invcycle', 'age', 'problit', 'pin', 'dbas']

    # Names
    specparam['features_names'] = {'yretm': 'Return', 'ib_l1eq': 'Earnings',
                                   'logeq': 'Size', 'logeqdef': 'Size',
                                   'm_b':'M/B ratio', 'flev': 'Leverage',
                                   'yvolatd': 'Volatility', 'noacc': 'NOAcc', 'cfoa': 'CFOA',
                                   'invcycle': 'Inv. Cycle', 'problit': 'Problit', 'age': 'Age',
                                   'pin': 'PIN', 'dbas': 'Bid-Ask Spread'}


    ####################
    # Cross-Validation #
    ####################
    # Neural Network
    specparam['cv_nn_paramsnames'] = {'n_iterations': 'N Iterations', 'layer_size': 'Layer Size', 'learning_rate': 'Learning Rate'}

    cv_nn = {
        'n_iterations': [20000],
        'layer_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'learning_rate': [1e-2]
    }

    ##################
    # DGP parameters #
    ##################
    coefs_dgp_basic = {'logeqdef': 4, 'm_b': -2, 'flev': 1}
    coefs_dgp_kw = {**coefs_dgp_basic, 'yvolatd': 4.3, 'noacc': -8, 'cfoa': 2.3, 'invcycle': -1.2, 'problit': 3.7, 'age': -6.1}


    # Assign the correct parameter values according to the specification
    if specification == 'replication':
        specparam['name'] = 'replication'
        specparam['features'] = features_replication
    elif specification == 'basic':
        specparam['name'] = 'basic'
        specparam['features'] = features_basic
        specparam['cv_nn'] = cv_nn
        specparam['coefs_dgp'] = coefs_dgp_basic
    elif specification == 'kw':
        specparam['name'] = 'kw'
        specparam['features'] = features_kw
        specparam['cv_nn'] = cv_nn
        specparam['coefs_dgp'] = coefs_dgp_kw
    else:
        raise Exception("Incorrect specification. It should be 'replication', 'basic' or 'kw'")

    # Create list of all features and all features with interactions
    p = specparam
    p['features_all'] = p['features']
    # Create interaction variables list
    featuresXret = []
    featuresXretXD = []
    featuresXD = []
    for f in p['features_all']:
        featuresXret += [p['retvar']+'X'+f]
        featuresXretXD += [p['retvar']+'XDX'+f]
        featuresXD += ['DX'+f]

    features0 = [p['retvar'], 'D', p['retvar']+'XD']
    p['features_all_int'] = features0+p['features_all']+featuresXret+featuresXretXD+featuresXD

    return p

##################
# Data functions #
##################

def define_samples(data, suffix=''):
    # Create random training and testing sample by splitting the sample by firms
    random.seed(1)
    # Define percentages (rest is test)
    train_perc = .5
    valid_perc = .8
    size = len(data.index)
    size_train = train_perc * size
    size_valid = valid_perc * size
    # Shuffle the gvkeys
    dfk = data[['gvkey', 'fyear']].groupby(['gvkey']).size().reset_index(name='count')
    keys = dfk['gvkey'].unique()
    random.shuffle(keys)
    dfkr = pd.DataFrame(keys, columns=['gvkey'])
    dfkr = pd.merge(dfkr, dfk)
    dfkr['sum'] = dfkr['count'].cumsum()
    # Define the samples
    dfkr[['train', 'valid', 'test']] = 0
    dfkr.loc[dfkr['sum'] < size_train, 'train'] = 1
    dfkr.loc[(dfkr['sum'] < size_valid) & (dfkr.train==0), 'valid'] = 1
    dfkr.loc[(dfkr.train==0) & (dfkr.valid==0), 'test'] = 1
    dfkr = dfkr[['gvkey', 'train', 'valid', 'test']]
    # Save the keys to file
    dfkr.to_csv(folder_models+'samples'+suffix+'.csv',index=False)

def winsorize(data, columns, threshold=.01):
    df = data[['fyear']+columns]

    # Winsorize
    def win(df):
        return df.clip(df.quantile(threshold), df.quantile(1-threshold), axis=1)

    grouped = df.groupby('fyear')
    data_win = grouped.apply(win)
    data[columns] = data_win[columns]
    return(data)

def trim(data, columns, threshold=.01):
    def trim(s):
        l = s[columns].quantile(threshold)
        h = s[columns].quantile(1-threshold)
        mask = np.all(s[columns]>l, axis=1) & np.all(s[columns]<h, axis=1)
        return s[mask]

    grouped = data.groupby('fyear')
    data_trim = grouped.apply(trim)
    return(data_trim.reset_index(drop=True))

def prepare_data(file, replicatekw=False, min_year=None):
    """ Prepare the data and complete the specparam dictionary """
    # Winsorization thresold (symmetric), performed by fiscal year
    winthreshold = .01
    # Variables to not winsorize
    notwinsorize = ['gvkey', 'fyear',
                    #'datadate', 'permno',
                    # Do not winsorize on 'clean' variables
                    'age', 'problit', 'pin', 'naics', 'sic',
                    # Do not winsorize binary variables
                    'crash',
                    # Do not winsorize FRED data
                    'tbill3m', 'tbill3y', 'tbill10y', 'gdpchg', 'infl', 'unemp', 'aaa', 'recession']

    # Open data
    data = pd.read_parquet(file)

    # Remove observations with price per share of less than $1
    data = data[data.price>=1]
    # Remove data after 2019 (fiscal year end, not fiscal year)
    #data['datadate'] = pd.to_datetime(data.datadate)
    data = data[data.datadate.dt.year < 2020]
    #data = data[data.datadate.dt.year < 2019]
    # Keep fiscal year after 1962
    data = data[data.fyear >= 1963]
    # Enforce minimum year if specified
    if min_year is not None:
        data = data[data.fyear >= min_year]
    
    # Correct fyear type (somehow winsorize does not work with Int32 type)
    data['fyear'] = data.fyear.astype(int)

    # Drop some columns
    data = data.drop(columns=['datadate', 'permno'])
    cols = list(data.columns)
    cols_notna = ['ib_l1eq', 'yretm', 'logeq', 'm_b', 'flev']

    if replicatekw:
        # To replicate KW paper - Trim
        cols_trim = ['ib_l1eq', 'yretm', 'logeq', 'm_b', 'flev', 'dp']
        cols_notna += ['noacc', 'cfoa', 'invcycle', 'age', 'dp']
        data = trim(data, cols_trim)
        # Keep KW sample
        data = data[(data.fyear <= 2005)]
    else:
        # Winsorize
        notwincols = list(set(cols).intersection(notwinsorize))
        cols_win = cols.copy()
        [cols_win.remove(i) for i in notwincols]
        data = winsorize(data, cols_win, threshold=winthreshold)

    # Remove NA for specifc columns
    data = data.dropna(subset=cols_notna)

    return(data)

def create_interactions(df, specparam):
    # This function creates interaction variables
    p = specparam
    data = df.copy()

    # Create interactions
    ## If the dummy already exists, do not touch it. Otherwise, create the dummy
    if 'D' not in data.columns:
        data['D'] = (data[p['retvar']] < 0).astype(int)
    for f in p['features_all']:
        data.loc[:,p['retvar']+'X'+f] = data[f]*data[p['retvar']]
        data.loc[:,p['retvar']+'XDX'+f] = data[f]*data[p['retvar']]*data['D']
        data.loc[:,'DX'+f] = data[f]*data['D']
    data.loc[:,p['retvar']+'XD'] = data[p['retvar']]*data['D']

    return(data)

def get_sets(df, specparam, features, yfe=False):
    p = specparam
    data = df.copy()
    data = create_interactions(data, specparam)
    if yfe:
        # Add the years to the features
        years = data['fyear'].unique().astype(int)
        years.sort()
        years_str = years.astype(str)
        data['fyear'] = data['fyear'].astype(int)
        for y in years:
            ys = y.astype(str)
            data[ys]=0
            data.loc[data.fyear==y, ys] = 1
    else:
        years_str = []
    X = data[p['earningsvar']]
    Y = data[features+list(years_str)]
    return X, Y

def full_sample(df, specparam):
    p = specparam
    sample = df.copy()
    # Drop missing values
    sample = sample.dropna(subset=[p['earningsvar'], *p['features']])
    return(sample)

def train_sample(df, specparam):
    p = specparam
    sample = full_sample(df, specparam)
    if p['samples'] == 'gvkeys':
        gvkeys = p['samples_gvkeys']['train'] + p['samples_gvkeys']['validation']
        sample = sample[sample.gvkey.isin(gvkeys)]
    elif p['samples'] == 'years':
        ymin, _ = p['samples_years']['train']
        _, ymax = p['samples_years']['validation']
        sample = sample[(sample.fyear>=ymin) & (sample.fyear<=ymax)]
    return(sample)

def test_sample(df, specparam):
    p = specparam
    sample = full_sample(df, specparam)
    if p['samples'] == 'gvkeys':
        gvkeys = p['samples_gvkeys']['test']
        sample = sample[sample.gvkey.isin(gvkeys)]
    elif p['samples'] == 'years':
        ymin, ymax = p['samples_years']['test']
        sample = sample[(sample.fyear>=ymin) & (sample.fyear<=ymax)]
    return(sample)

def cv_train_sample(df, specparam):
    p = specparam
    sample = full_sample(df, specparam)
    if p['samples'] == 'gvkeys':
        gvkeys = p['samples_gvkeys']['train']
        sample = sample[sample.gvkey.isin(gvkeys)]
    elif p['samples'] == 'years':
        ymin, ymax = p['samples_years']['train']
        sample = sample[(sample.fyear>=ymin) & (sample.fyear<=ymax)]        
    return(sample)

def cv_valid_sample(df, specparam):
    p = specparam
    sample = full_sample(df, specparam)
    if p['samples'] == 'gvkeys':
        gvkeys = p['samples_gvkeys']['validation']
        sample = sample[sample.gvkey.isin(gvkeys)]
    elif p['samples'] == 'years':
        ymin, ymax = p['samples_years']['validation']
        sample = sample[(sample.fyear>=ymin) & (sample.fyear<=ymax)]        
    return(sample)

##################
# Replication KW #
##################

def coefs_kw(specparam, fullsample=False, suffix=''):
    folder_fm = _get_folder_fm(specparam, fullsample, suffix)
    coefs = pd.read_csv(folder_fm+'coefficients.csv')
    dtm = coefs.mean()
    dtsd = coefs.std()                   # Standard Deviation
    dtse = dtsd / np.sqrt(len(coefs))    # Standard Error
    dtts = dtm / dtse                    # t-stat with beta0 = 0
    dt = pd.DataFrame([dtm, dtts], index=['Coefficient', 't-stat'])
    names = {'intercept': 'Intercept',
         'yretm': 'Ret',
         'D': 'D',
         'yretmXD': 'D x Ret',
         'logeq': 'Size',
         'm_b': 'M/B',
         'flev': 'Lev',
         'yretmXlogeq': 'Ret x Size',
         'yretmXm_b': 'Ret x M/B',
         'yretmXflev': 'Ret x Lev',
         'yretmXDXlogeq': 'D x Ret x Size',
         'yretmXDXm_b': 'D x Ret x M/B',
         'yretmXDXflev': 'D x Ret x Lev',
         'DXlogeq': 'D x Size',
         'DXm_b': 'D x M/B',
         'DXflev': 'D x Lev',
         'r2': 'R2'}
    dt = dt[['intercept', 'D', 'yretm', 'yretmXlogeq', 'yretmXm_b', 'yretmXflev', 'yretmXD', 'yretmXDXlogeq', 'yretmXDXm_b', 'yretmXDXflev',
        'logeq', 'm_b', 'flev', 'DXlogeq', 'DXm_b', 'DXflev', 'r2']]
    dt.columns = [names[c] for c in dt.columns]
    return dt.transpose()

def table_coefs_kw(specparam):
    filename = 'coefs_kw'
    # Full Sample
    col_full = coefs_kw(specparam, fullsample=True).transpose()
    col_full.loc['t-stat', 'R2'] = np.nan
    # KW replication
    col_kw = coefs_kw(specparam).transpose()
    col_kw.loc['t-stat', 'R2'] = np.nan
    # KW coefs from paper
    col_kw_paper = col_kw.copy()
    col_kw_paper.loc['Coefficient'] = [.083, -.024,
                                       .031, .005, -.006, .005,
                                       .237, -.033, -.007, .033,
                                       .005, -.017, -.008, .003, -.001, -.002,
                                       .24]
    col_kw_paper.loc['t-stat'] = [7.53, -3.56,
                                  1.84, 2.25, -2., .77,
                                  10.78, -7.42, -0.93, 1.86,
                                  4.83, -7.93, -3.61, 3.45, -0.42, -0.88,
                                  np.nan]
    # Creae final table
    df = pd.concat([col_full, col_kw, col_kw_paper], axis=0)
    i = pd.MultiIndex.from_product([['Full Sample', 'Replication', 'Khan and Watts (2009)'], col_kw.index])
    df = df.set_index(i)
    df = df.transpose()
    # Write to file
    df.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, escape=True, na_rep='')
    return(df)

##################
# Neural Network #
##################

class mLayer(tf.keras.layers.Layer):
    # Defines the last layer of the full model
    def __init__(self, **kwargs):
        super(mLayer, self).__init__(**kwargs)

    def call(self, inputs):
        R = inputs[0]
        D = inputs[1]
        f = inputs[2]
        g = inputs[3]
        c = inputs[4]
        d = inputs[5]
        return f + R*g + R*D*c + D*d

class cLayer(tf.keras.layers.Layer):
    # Defines the last layer of the full model
    def __init__(self, **kwargs):
        super(cLayer, self).__init__(**kwargs)

    def call(self, inputs):
        g = inputs[0]
        gc = inputs[1]
        return gc - g

class NeuralNetwork():

    def __init__(self, n_iterations=100, learning_rate=.01, layer_size=0, verbose=0, random_state=1,
                 loss='huber', linear=False, pos_constraint=True,
                 CViterfreq=None, CVfilename=None, specparam=None, params_pr=None,
                 scaler=None):
        #tf.random.set_seed(random_state)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.display_step = int(round(self.n_iterations / 10))
        self.display_perc = int(round(self.n_iterations / 100))
        self.verbose = verbose
        self.linear = linear
        self.pos_constraint = pos_constraint
        # Needed arguments when cross-validating
        self.CViterfreq = CViterfreq
        self.CVfilename = CVfilename
        self.specparam = specparam
        self.params_pr = params_pr
        # Unused
        self.scaler = scaler
        # No intermediate layer and no positivity constraint if 'linear' model
        if self.linear:
            self.layer_size = 0
            self.pos_constraint = False
        else:
            self.layer_size = layer_size
        # Set the positivity constraint activation function
        if self.pos_constraint:
            self.activation_gc = 'exponential'
        else:
            self.activation_gc = 'linear'
        # Set the loss function
        if loss == 'huber':
            # Create a Huber Loss function with delta=1
            self.L = tf.keras.losses.Huber()
        elif loss == 'mse':
            self.L = tf.keras.losses.MeanSquaredError()

    def _get_model(self, inputs, activation='linear'):
        # Returns the model for one of the functions (f, g, c, d)
        #inputs = tf.keras.layers.Input(shape=(num_features,))
        if self.layer_size > 0:
            out = tf.keras.layers.Dense(self.layer_size, activation="tanh")(inputs)
        else:
            out = inputs
        outputs = tf.keras.layers.Dense(1, activation=activation)(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def _get_full_model(self, num_features):
        # Features (e.g. [size, mb, leverage])
        inputs_features = tf.keras.layers.Input(shape=(num_features,))
        # Return and Dummy (R, D)
        inputs_R = tf.keras.layers.Input(shape=(1,))
        inputs_D = tf.keras.layers.Input(shape=(1,))

        # Create models for each function (f, g, c, d)
        self.f = self._get_model(inputs_features)
        self.d = self._get_model(inputs_features)
        self.g = self._get_model(inputs_features, activation=self.activation_gc)
        self.gc = self._get_model(inputs_features, activation=self.activation_gc)

        out_c = cLayer()([self.g.output, self.gc.output])
        out = mLayer()([inputs_R, inputs_D, self.f.output, self.g.output, out_c, self.d.output])
        #out = FunctionLayer()([inputs_R, inputs_D, self.f.output, self.g.output, self.gc.output, self.d.output])
        model = tf.keras.Model(inputs=[inputs_features, inputs_R, inputs_D], outputs=[out])
        # Compile model?
        return model

    def loss(self, x, y, r, d):
        y_pred = self.model([x,r,d])
        loss = self.L(y, y_pred)
        return loss

    def predict(self, X, R, D):
        x = self._get_tf_x(X)
        r = self._get_tf_var(R)
        d = self._get_tf_var(D)
        y_pred = self.model([x,r,d])
        return y_pred.numpy()

    def compute_r2(self, X, Y, R, D):
        x, y, r, d = self._get_tf_vars(X, Y, R, D)
        y_pred = self.model([x,r,d])
        y_mean = np.mean(y_pred)
        SStot = np.sum((y-y_mean)**2)
        SSres = np.sum((y-y_pred)**2)
        R2 = 1 - SSres / SStot
        return R2

    @tf.function
    def run_optimization(self, x, y, r, d):
        # Get the gradients
        with tf.GradientTape() as g:
            loss = self.loss(x, y, r, d)
        grad = g.gradient(loss, self.θ)
        # Upate the weights
        self.opt.apply_gradients(zip(grad, self.θ))

    def _get_tf_x(self, X):
        # Normalize
        if self.scaler is not None:
            X = self.scaler.transform(X)
        x = tf.constant(X, dtype=tf.float32)
        return x

    def _get_tf_var(self, Y):
        y = tf.expand_dims(tf.constant(Y, dtype=tf.float32), axis=1)
        return y

    def _get_tf_vars(self, X, Y, R, D):
        # Get the inputs and labels
        x = self._get_tf_x(X)
        y = self._get_tf_var(Y)
        # Get the return and dummy
        r = self._get_tf_var(R)
        d = self._get_tf_var(D)
        return x, y, r, d

    def fit(self, X, Y, R, D, Xv=None, Yv=None, Rv=None, Dv=None):
        num_features = X.shape[1]
        # Create the NN model
        self.model = self._get_full_model(num_features)
        # Weights
        self.θ = self.model.trainable_weights
        # Optimizer
        self.opt = tf.optimizers.Adam(self.learning_rate)
        # Train
        self.training_loop(X, Y, R, D, Xv, Yv, Rv, Dv)

    def training_loop(self, X, Y, R, D, Xv, Yv, Rv, Dv):
        # Get TF variables
        x, y, r, d = self._get_tf_vars(X, Y, R, D)
        for step in range(self.n_iterations):
            self.run_optimization(x, y, r, d)
            if self.CViterfreq is not None and step % self.CViterfreq == 0:
                # Compute R2 on train and test samples and add to CV file
                res = {}
                res['r2train'] = self.compute_r2(X, Y, R, D)
                res['r2valid'] = self.compute_r2(Xv, Yv, Rv, Dv)
                self.params_pr['n_iterations'] = step
                add_to_cv(self.specparam, self.CVfilename, self.params_pr, res)
            if self.verbose==2 and step % self.display_step == 0:
                loss = self.loss(x, y, r, d)
                r2 = self.compute_r2(X, Y, R, D)
                print("step: %i, loss: %f, R2: %f" % (step, loss, r2))
            if self.verbose==1 and step % self.display_perc == 0:
                perc = int((step / self.n_iterations) * 100)
                print("\r {:2d}%".format(perc), end='')
        print('\r', end='')

class CNetwork():

    def __init__(self, n_iterations=100, learning_rate=.01, layer_size=0, verbose=0, loss='huber'):
        #tf.random.set_seed(random_state)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.display_step = int(round(self.n_iterations / 10))
        self.display_perc = int(round(self.n_iterations / 100))
        self.verbose = verbose
        self.layer_size = layer_size
        # Set the loss function
        if loss == 'huber':
            # Create a Huber Loss function with delta=1
            self.L = tf.keras.losses.Huber()
        elif loss == 'mse':
            self.L = tf.keras.losses.MeanSquaredError()

    def _get_model(self, num_features):
        # Returns the model for one of the functions (f, g, c, d)
        inputs = tf.keras.layers.Input(shape=(num_features,))
        out = tf.keras.layers.Dense(self.layer_size, activation="tanh")(inputs)
        outputs = tf.keras.layers.Dense(1, activation='linear')(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def loss(self, x, y):
        y_pred = self.model([x])
        loss = self.L(y, y_pred)
        return loss

    def predict(self, X):
        x = tf.constant(X, dtype=tf.float32)
        y_pred = self.model([x])
        return y_pred.numpy()

    def compute_r2(self, X, Y):
        x = tf.constant(X, dtype=tf.float32)
        y = tf.expand_dims(tf.constant(Y, dtype=tf.float32), axis=1)
        y_pred = self.model([x])
        y_mean = np.mean(y_pred)
        SStot = np.sum((y-y_mean)**2)
        SSres = np.sum((y-y_pred)**2)
        R2 = 1 - SSres / SStot
        return R2

    @tf.function
    def run_optimization(self, x, y):
        # Get the gradients
        with tf.GradientTape() as g:
            loss = self.loss(x, y)
        grad = g.gradient(loss, self.θ)
        # Upate the weights
        self.opt.apply_gradients(zip(grad, self.θ))

    def fit(self, X, Y):
        num_features = X.shape[1]
        # Create the NN model
        self.model = self._get_model(num_features)
        # Weights
        self.θ = self.model.trainable_weights
        # Optimizer
        self.opt = tf.optimizers.Adam(self.learning_rate)
        # Train
        self.training_loop(X, Y)

    def training_loop(self, X, Y):
        # Get TF variables
        x = tf.constant(X, dtype=tf.float32)
        y = tf.expand_dims(tf.constant(Y, dtype=tf.float32), axis=1)
        for step in range(self.n_iterations):
            self.run_optimization(x, y)
            if self.verbose==2 and step % self.display_step == 0:
                loss = self.loss(x, y)
                r2 = self.compute_r2(X, Y)
                print("step: %i, loss: %f, R2: %f" % (step, loss, r2))
            if self.verbose==1 and step % self.display_perc == 0:
                perc = int((step / self.n_iterations) * 100)
                print("\r {:2d}%".format(perc), end='')
        print('\r', end='')      

def save_model_nn(nn, file_f, file_g, file_gc, file_d, file_model, file_nn):
    # Remember the models
    f = nn.f
    g = nn.g
    gc = nn.gc
    d = nn.d
    model = nn.model
    # Save the models
    f.save(file_f+'.h5')
    g.save(file_g+'.h5')
    gc.save(file_gc+'.h5')
    d.save(file_d+'.h5')
    model.save(file_model+'.h5')
    # Remove the models from the NeuralNetwork object
    nn.f = None
    nn.g = None
    nn.gc = None
    nn.d = None
    nn.model = None
    # Save the object
    joblib.dump(nn, file_nn+'.sav')
    # Reconstruct the NeuralNetwork object
    nn.f = f
    nn.g = g
    nn.gc = gc
    nn.d = d
    nn.model = model

def load_model_nn(file_f, file_g, file_gc, file_d, file_model, file_nn):
    # Load the NeuralNetwork object
    nn = joblib.load(file_nn+'.sav')
    # Load the models
    nn.f = tf.keras.models.load_model(file_f+'.h5', compile=False)
    nn.g = tf.keras.models.load_model(file_g+'.h5', compile=False)
    nn.gc = tf.keras.models.load_model(file_gc+'.h5', compile=False)
    nn.d = tf.keras.models.load_model(file_d+'.h5', compile=False)
    nn.model = tf.keras.models.load_model(file_model+'.h5', custom_objects={'mLayer': mLayer, 'cLayer': cLayer}, compile=False)
    return nn

##############################
# Cross-Validation functions #
##############################

def cv_filename(specparam, yfesuffix, suffix):
    p = specparam
    return folder_models+'nn_grid_cv_'+p['name']+yfesuffix+suffix+'.csv'

def add_to_cv(specparam, filename, params_pr, r):
    p = specparam
    # Open file if aready exists
    if os.path.exists(filename):
        resdf = pd.read_csv(filename)
    else:
        resdf = pd.DataFrame([], columns=list(p['cv_nn'])+['r2train', 'r2valid'])
    res = [list(params_pr.values()) + [r['r2train'], r['r2valid']]]
    res = pd.DataFrame(res, columns=list(p['cv_nn'])+['r2train', 'r2valid'])
    resdf = pd.concat([resdf, res])
    resdf.to_csv(filename, index=False)

def run_search_nn(df, specparam, recompute=False, yfe=True, pos_constraint=True, suffix=''):
    p = specparam
    if yfe:
        yfesuffix = '_yfe'
    else:
        yfesuffix = ''
    filename = cv_filename(p, yfesuffix, suffix)
    if recompute:
        # Open the current CV file to avoid recomputing if possible
        if os.path.exists(filename):
            resdf = pd.read_csv(filename)
        else:
            resdf = pd.DataFrame([], columns=list(p['cv_nn'])+['r2train', 'r2valid'])
        keys = list(p['cv_nn'].keys())
        iterator = list(product(*list(p['cv_nn'].values())))
        for i in iterator:
            params = {keys[j]: i[j] for j in range(len(i))}
            params_pr = params.copy()
            params['verbose'] = 1
            print('##### NN for '+str(params_pr)+' #####')
            # Do not recompute if already computed
            if not (resdf[params_pr]==params_pr.values()).all(1).any():
                params['CViterfreq'] = 100
                params['CVfilename'] = filename
                params['specparam'] = p
                params['params_pr'] = params_pr
                # Run the model 10 times to have enough points
                for i in range(10):
                    train_nn(df, p, params, cv=True, recompute=True, yfe=yfe, pos_constraint=pos_constraint, suffix=suffix)
    resdf = pd.read_csv(filename)
    return resdf

######################
# Training functions #
######################

def _get_filename_nn(fct, specparam, fullsample, yfe, suffix):
    p = specparam
    if yfe:
        yfesuffix = '_yfe'
    else:
        yfesuffix = ''
    if fullsample:
        fs = '_fullsample'
    else:
        fs = ''
    filename = folder_models+'nn'+fct+'_'+p['name']+yfesuffix+fs+'_model'+suffix
    return filename

def _get_folder_fm(specparam, fullsample, suffix):
    p = specparam
    if fullsample:
        folder_fm = folder_models+'FM_'+p['name']+'_fullsample'+suffix+'/'
    else:
        folder_fm = folder_models+'FM_'+p['name']+suffix+'/'
    # Create the folder if it does not exist
    if not os.path.isdir(folder_fm):
        os.mkdir(folder_fm)
    return(folder_fm)

def _get_r2(df, specparam):
    p = specparam
    # Get X and X_pred
    X, Y = get_sets(df, p, p['features_all_int'])
    X_pred = df['X_pred']
    # Compute the R2
    Xmean = np.mean(X_pred)
    SStot = np.sum((X-Xmean)**2)
    SSres = np.sum((X-X_pred)**2)
    R2 = 1 - SSres / SStot
    # Return the R2
    return(R2)

def compute_fm_linear(df, specparam, fullsample=False, suffix=''):
    """ Fama-Macbeth regression (as in KW) """
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    # Define training sample and set folder to save the yearly models
    if p['name'] == 'replication' or fullsample:
        df_train = full_sample(df, p)
    else:
        df_train = train_sample(df, p)

    min_fyear = int(min(df_train.fyear))
    max_fyear = int(max(df_train.fyear))
    # Create the dataframe to store coefs
    cols = ['fyear','intercept']+p['features_all_int']+['r2']
    cdf = pd.DataFrame(None,cols).transpose()
    # Run linear regressions by year
    for y in range(min_fyear, max_fyear+1):
        X, Y = get_sets(df_train[df_train.fyear==y], p, p['features_all_int'])
        # Fit the model
        m = LinearRegression(n_jobs=24)
        m.fit(Y, X)
        # Add the coefs to the dataframe
        v = pd.DataFrame(np.concatenate(([y, m.intercept_], m.coef_, [m.score(Y,X)])), cols).transpose()
        cdf = pd.concat([cdf, v])
        # Save this year's model
        joblib.dump(m, folder_fm+'model_'+str(y)+'.sav')
    # Save the coefs to file
    cdf.to_csv(folder_fm+'coefficients.csv', index=False)
    # Create an 'average' model
    cdf_final = cdf.drop(columns=['fyear'])
    coeffs = np.mean(cdf_final)
    m.coef_ = np.array(coeffs[1:-1])
    m.intercept_ = coeffs[0]
    # Save the 'average' model to file
    joblib.dump(m, folder_fm+'model_average.sav')

def train_linear(df, specparam, fullsample=False, recompute=False, yfe=False, suffix='', save=True):
    p = specparam
    filename = folder_models+'linear_'+p['name']+'_model'+suffix+'.sav'
    data = full_sample(df, p)
    if fullsample:
        df_train = full_sample(df, p)
    else:
        df_train = train_sample(df, p)

    # Fit the model
    m = LinearRegression(n_jobs=ncpu)
    if recompute:
        X_train, Y_train = get_sets(df_train, p, p['features_all_int'], yfe=yfe)
        m.fit(Y_train, X_train)
        if save:
            joblib.dump(m, filename)
    else:
        m = joblib.load(filename)

    # Compute prediction (on full sample)
    X, Y = get_sets(data, p, p['features_all_int'], yfe=yfe)
    data['X_pred'] = m.predict(Y)
    # get R2
    if fullsample:
        R2 = _get_r2(data, p)
        R2dict = {'full': R2}
    else:
        # Separate train and test samples
        df_train = train_sample(data, p)
        df_test = test_sample(data, p)
        # Compute R2
        R2_train = _get_r2(df_train, p)
        R2_test = _get_r2(df_test, p)
        R2dict = {'train': R2_train, 'test': R2_test}
    return m, R2dict

def train_nn(df, specparam, params, cv=False, fullsample=False, recompute=False, standardize=False, yfe=False, pos_constraint=True, suffix='', nmodels=100, save=True):
    p = specparam
    # Get the models filenames
    filename_f = _get_filename_nn('f', p, fullsample, yfe, suffix)
    filename_g = _get_filename_nn('g', p, fullsample, yfe, suffix)
    filename_gc = _get_filename_nn('gc', p, fullsample, yfe, suffix)
    filename_d = _get_filename_nn('d', p, fullsample, yfe, suffix)
    filename_model = _get_filename_nn('model', p, fullsample, yfe, suffix)
    filename_nn = _get_filename_nn('nn', p, fullsample, yfe, suffix)

    # Create the datasets
    if cv: # Cross-Validation
        train = cv_train_sample(df, p)
        valid = cv_valid_sample(df, p)
        X_train, Y_train = get_sets(train, p, p['features_all'], yfe=yfe)
        X_valid, Y_valid = get_sets(valid, p, p['features_all'], yfe=yfe)
        Rtr = train[p['retvar']]
        Dtr = (Rtr<0.).astype('float')
        Rva = valid[p['retvar']]
        Dva = (Rva<0.).astype('float')
    elif fullsample:
        train = full_sample(df, p)
        X_train, Y_train = get_sets(train, p, p['features_all'], yfe=yfe)
        Rtr = train[p['retvar']]
        Dtr = (Rtr<0.).astype('float')
    else:
        train = train_sample(df, p)
        test = test_sample(df, p)
        X_train, Y_train = get_sets(train, p, p['features_all'], yfe=yfe)
        X_test, Y_test = get_sets(test, p, p['features_all'], yfe=yfe)
        Rtr = train[p['retvar']]
        Dtr = (Rtr<0.).astype('float')
        Rte = test[p['retvar']]
        Dte = (Rte<0.).astype('float')

    # Create the Scaler (for normalization)
    scaler = None
    if standardize:
        X_scaler, Y_scaler = get_sets(train_sample(df, p), p, p['features_all'], yfe=yfe)
        scaler = StandardScaler()
        scaler.fit(Y_scaler)
    if cv:
        # Create and fit the NN
        nn = NeuralNetwork(**params, pos_constraint=pos_constraint, scaler=scaler)
        # Use training and validation
        nn.fit(Y_train, X_train, Rtr, Dtr, Y_valid, X_valid, Rva, Dva)
        # Compute R2 on the training and validation sample
        #r2train = nn.compute_r2(Y_train, X_train, Rtr, Dtr)
        #r2valid = nn.compute_r2(Y_valid, X_valid, Rva, Dva)
        #return {'r2train': r2train, 'r2valid': r2valid}
    else:
        if recompute:
            # Train the neural network nmodels times
            print('Train '+str(nmodels)+' models.')
            for i in range(nmodels):
                print('Model '+str(i))
                # Create and fit the NN
                nn = NeuralNetwork(**params, pos_constraint=pos_constraint, scaler=scaler)
                # Use training and test
                nn.fit(Y_train, X_train, Rtr, Dtr)
                if False:
                    # Print R2 (for now)
                    r2_train = nn.compute_r2(Y_train, X_train, Rtr, Dtr)
                    if fullsample:
                        nn_r2 = {'full': r2_train}
                    else:
                        r2_test = nn.compute_r2(Y_test, X_test, Rte, Dte)
                        nn_r2 = {'train': r2_train, 'test': r2_test}
                    print(nn_r2)
                # Save the models
                if save:
                    save_model_nn(nn, filename_f+str(i), filename_g+str(i), filename_gc+str(i), filename_d+str(i), filename_model+str(i), filename_nn+str(i))                   

def r2_fm(df, specparam, fullsample=False, suffix='')  :
    # Compute the R2 for average yearly linear
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    # Get the sets
    X, Y = get_sets(df, p, p['features_all_int'])
    # Load the average model
    m = joblib.load(folder_fm+'model_average.sav')
    # Compute the prediction on the full sample (both train and test at the same time)
    df['X_pred'] = m.predict(Y)

    if replicatekw or fullsample:
        R2 = _get_r2(df, p)
        return {'full': R2}
    else:
        # Separate train and test samples
        df_train = train_sample(df, p)
        df_test = test_sample(df, p)
        # Compute R2
        R2_train = _get_r2(df_train, p)
        R2_test = _get_r2(df_test, p)
        return {'train': R2_train, 'test': R2_test}

def r2_fmy(df, specparam, fullsample=False, suffix=''):
    # Compute the R2 for yearly linear
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    data = full_sample(df, p)
    # Predict using yearly regressions
    min_fyear = int(min(df.fyear))
    max_fyear = int(max(df.fyear))
    # Create the dataframe to store the prediction (for both train and test)
    X_pred = pd.DataFrame(None,['X_pred']).transpose()
    for y in range(min_fyear, max_fyear+1):
        # Get the data for that year and the sets
        dfy = data[data.fyear==y]
        X, Y = get_sets(dfy, p, p['features_all_int'])
        # Load the model for that year
        m = joblib.load(folder_fm+'model_'+str(y)+'.sav')
        # Compute the prediction on the full sample (both train and test at the same time)
        dfy['X_pred'] = m.predict(Y)
        # Add to the final dataframe
        X_pred = pd.concat([X_pred, dfy[['X_pred']]])
    # Merge the data to the dataframe
    data['X_pred'] = X_pred['X_pred']

    if fullsample:
        R2 = _get_r2(data, p)
        return {'full': R2}
    else:
        # Separate train and test samples
        df_train = train_sample(data, p)
        df_test = test_sample(data, p)
        # Compute R2
        R2_train = _get_r2(df_train, p)
        R2_test = _get_r2(df_test, p)
        return {'train': R2_train, 'test': R2_test}

def r2_nn(df, specparam, fullsample=False, yfe=False, suffix='', nmodels=10, model0index=0):
    # Compute the R2 for NN
    p = specparam    
    # Get the average prediction
    data = full_sample(df, p)
    data['X_pred'] = predict_average_nn(data, specparam, fullsample, yfe, suffix, nmodels, model0index)
    
    # Compute the R2
    if fullsample:
        R2 = _get_r2(data, p)
        return {'full': R2}
    else:
        # Separate train and test samples
        df_train = train_sample(data, p)
        df_test = test_sample(data, p)
        # Compute R2
        R2_train = _get_r2(df_train, p)
        R2_test = _get_r2(df_test, p)
        return {'train': R2_train, 'test': R2_test}

def predict_average_nn(df, specparam, fullsample=False, yfe=False, suffix='', nmodels=100, model0index=0):
    p = specparam
    # Get the filenames
    filename_f = _get_filename_nn('f', p, fullsample, yfe, suffix)
    filename_g = _get_filename_nn('g', p, fullsample, yfe, suffix)
    filename_gc = _get_filename_nn('gc', p, fullsample, yfe, suffix)
    filename_d = _get_filename_nn('d', p, fullsample, yfe, suffix)
    filename_model = _get_filename_nn('model', p, fullsample, yfe, suffix)
    filename_nn = _get_filename_nn('nn', p, fullsample, yfe, suffix)
    # Get the sets
    X, Y = get_sets(df, p, p['features_all'], yfe=yfe)
    R = df[p['retvar']]
    D = (R<0.).astype('float')

    # Compute average prediction over the models
    dfpred = pd.DataFrame(index=df.index)
    for i in range(model0index, model0index+nmodels):
        nn = load_model_nn(filename_f+str(i), filename_g+str(i),
                           filename_gc+str(i), filename_d+str(i),
                           filename_model+str(i), filename_nn+str(i))
        dfpred['X_pred'+str(i)] = nn.predict(Y, R, D)
    
    # Average the prediction
    df['X_pred'] = np.mean(dfpred, axis=1)
    # Return the averaged prediction
    return df.X_pred
 
def load_full_model(specparam, fullsample=False, yfe=False, suffix='', i=0):
    p = specparam
    # Get the filenames
    filename_f = _get_filename_nn('f', p, fullsample, yfe, suffix)
    filename_g = _get_filename_nn('g', p, fullsample, yfe, suffix)
    filename_gc = _get_filename_nn('gc', p, fullsample, yfe, suffix)
    filename_d = _get_filename_nn('d', p, fullsample, yfe, suffix)
    filename_model = _get_filename_nn('model', p, fullsample, yfe, suffix)
    filename_nn = _get_filename_nn('nn', p, fullsample, yfe, suffix)
    nn = load_model_nn(filename_f+str(i), filename_g+str(i),
                       filename_gc+str(i), filename_d+str(i),
                       filename_model+str(i), filename_nn+str(i))
    return nn

    
###################
# Compute C score #
###################

def m0plus(m, dfcopy, specparam, Rplus):
    p = specparam
    dfcopy.loc[:,p['retvar']] = Rplus
    dfcopy.loc[:,'D'] = 0
    X, Y = get_sets(dfcopy, specparam, p['features_all_int'])
    return m.predict(Y)

def m1plus(m, dfcopy, specparam, Rplus):
    p = specparam
    dfcopy.loc[:,p['retvar']] = Rplus
    dfcopy.loc[:,'D'] = 1
    X, Y = get_sets(dfcopy, specparam, p['features_all_int'])
    return m.predict(Y)

def m0minus(m, dfcopy, specparam, Rminus):
    p = specparam
    dfcopy.loc[:,p['retvar']] = Rminus
    dfcopy.loc[:,'D'] = 0
    X, Y = get_sets(dfcopy, specparam, p['features_all_int'])
    return m.predict(Y)

def m1minus(m, dfcopy, specparam, Rminus):
    p = specparam
    dfcopy.loc[:,p['retvar']] = Rminus
    dfcopy.loc[:,'D'] = 1
    X, Y = get_sets(dfcopy, specparam, p['features_all_int'])
    return m.predict(Y)

def c(m, df, specparam, Rplus=1, Rminus=-1):
    # Compute C score for linear models
    p = specparam
    dfcopy = full_sample(df, p)
    m0p = m0plus(m, dfcopy, specparam, Rplus)
    m1p = m1plus(m, dfcopy, specparam, Rplus)
    m0m = m0minus(m, dfcopy, specparam, Rminus)
    m1m = m1minus(m, dfcopy, specparam, Rminus)
    dfcopy['cscore'] =  ((m1p-m0p) - (m1m-m0m)) / (Rplus-Rminus)
    return dfcopy['cscore']

def g(m, df, specparam, Rplus=1, Rminus=-1):
    # Compute G score for linear models
    p = specparam
    dfcopy = full_sample(df, p)
    m0p = m0plus(m, dfcopy, specparam, Rplus)
    m0m = m0minus(m, dfcopy, specparam, Rminus)
    dfcopy['gscore'] = (m0p-m0m) / (Rplus-Rminus)
    return dfcopy['gscore']

def c_score_fm(df, specparam, fullsample=False, suffix=''):
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    # Get the sets
    X, Y = get_sets(df, p, p['features_all_int'])
    # Load the average model
    m = joblib.load(folder_fm+'model_average.sav')
    # Compute the C score
    df['cscore'] = c(m, df, p)
    # Return the C score
    return df['cscore']

def g_score_fm(df, specparam, fullsample=False, suffix=''):
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    # Get the sets
    X, Y = get_sets(df, p, p['features_all_int'])
    # Load the average model
    m = joblib.load(folder_fm+'model_average.sav')
    # Compute the G score
    df['gscore'] = g(m, df, p)
    # Return the G score
    return df['gscore']

def c_score_fmy(df, specparam, fullsample=False, suffix=''):
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    data = df.copy()
    # Compute the C score using yearly regressions
    min_fyear = int(min(data.fyear))
    max_fyear = int(max(data.fyear))
    # Create the dataframe to store the C score
    cdf = pd.DataFrame(None,['cscore']).transpose()
    for y in range(min_fyear, max_fyear+1):
        dfy = data[df.fyear==y]
        # Load the model for that year
        m = joblib.load(folder_fm+'model_'+str(y)+'.sav')
        # Compute the C score
        dfy['cscore'] = c(m, dfy, p)
        # Add to the final dataframe
        cdf = pd.concat([cdf, dfy[['cscore']]])
    # Merge the data to the dataframe
    data['cscore'] = cdf['cscore']
    # Return the C score
    return data['cscore']

def g_score_fmy(df, specparam, fullsample=False, suffix=''):
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    # Compute the C score using yearly regressions
    min_fyear = int(min(df.fyear))
    max_fyear = int(max(df.fyear))
    # Create the dataframe to store the C score
    gdf = pd.DataFrame(None,['gscore']).transpose()
    for y in range(min_fyear, max_fyear+1):
        dfy = df[df.fyear==y]
        # Load the model for that year
        m = joblib.load(folder_fm+'model_'+str(y)+'.sav')
        # Compute the G score
        dfy['gscore'] = g(m, dfy, p)
        # Add to the final dataframe
        gdf = pd.concat([gdf, dfy[['gscore']]])
    # Merge the data to the dataframe
    df['gscore'] = gdf['gscore']
    # Return the G score
    return df['gscore']

def c_score_nn(df, specparam, fullsample=False, yfe=False, suffix='', nmodels=100, model0index=0):
    p = specparam
    # Get the sets
    X, Y = get_sets(df, p, p['features_all'], yfe=yfe)
    x_nn = tf.constant(Y, dtype=tf.float32)
    # Get the models filenames
    file_g = _get_filename_nn('g', p, fullsample, yfe, suffix)
    file_gc = _get_filename_nn('gc', p, fullsample, yfe, suffix)
    # Compute the prediction for each models
    predC = pd.DataFrame(index=df.index)
    for i in range(model0index, model0index+nmodels):
        nn_g = tf.keras.models.load_model(file_g+str(i)+'.h5', compile=False)
        nn_gc = tf.keras.models.load_model(file_gc+str(i)+'.h5', compile=False)
        g = tf.squeeze(nn_g(x_nn)).numpy()
        gc = tf.squeeze(nn_gc(x_nn)).numpy()
        c = gc - g
        predC['predC'+str(i)] = c
    # Compute average
    C = np.mean(predC, axis=1)
    return C

def g_score_nn(df, specparam, fullsample=False, yfe=False, suffix='', nmodels=100, model0index=0):
    p = specparam
    # Get the sets
    X, Y = get_sets(df, p, p['features_all'], yfe=yfe)
    x_nn = tf.constant(Y, dtype=tf.float32)
    # Get the models filenames
    file_g = _get_filename_nn('g', p, fullsample, yfe, suffix)
    # Compute the prediction for each models
    predG = pd.DataFrame(index=df.index)
    for i in range(model0index, model0index+nmodels):
        nn_g = tf.keras.models.load_model(file_g+str(i)+'.h5', compile=False)
        g = tf.squeeze(nn_g(x_nn)).numpy()
        predG['predG'+str(i)] = g
    # Compute average
    G = np.mean(predG, axis=1)
    return G


######################
# Plotting Functions #
######################

def save_plot(filename):
    #tikzplotlib.save(folder_plots+filename+'.tikz')
    plt.savefig(folder_plots+filename+'.pdf', bbox_inches='tight')

def plot_density(var, df, xlabel, xlim=None, filename=None):
    plt.figure()
    ax = sns.kdeplot(x=var, data=df, color='black', gridsize=1000)
    ax.set_xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if filename:
        save_plot(filename)
    plt.show()
    plt.close()

def plot_score_density(var, df, specparam, sample='test', xlim=None, score='C score', suffix=''):
    p = specparam
    if sample == 'train':
        data = train_sample(df, specparam)
    elif sample == 'test':
        data = test_sample(df, specparam)
    elif sample == 'full':
        data = df
    else:
        raise Excpetion("sample should be 'train', 'test' or 'full'." )
    filename = 'density_'+var+'_'+sample+suffix
    plot_density(var, data, score, xlim=xlim, filename=filename)

def plot_densities(variables, legend, df, xlabel, xlim=None, filename=None):
    l = legend.copy()
    l.reverse()
    plt.figure()
    ax = sns.kdeplot(data=df[variables], palette='Greys')
    ax.legend(l)
    ax.set_xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if filename:
        save_plot(filename)
    plt.show()
    plt.close()

def plot_c_densities(Rlist, R, df, specparam, sample, xlim, name):
    p = specparam
    if sample == 'train':
        data = train_sample(df, specparam)
    elif sample == 'test':
        data = valid_sample(df, specparam)
    elif sample == 'full':
        data = df
    else:
        raise Excpetion("sample should be 'train', 'test' or 'full'." )
    filename = 'densities_'+''.join(Rlist)+'_'+name+'_'+sample
    legend = [R['legend'][i] for i in Rlist]
    variables = ['c_'+name+'_'+r for r in Rlist]
    plot_densities(variables, legend, data, 'C score', xlim=xlim, filename=filename)

def plot_cv(specparam, name, params={}, yfe=True, suffix='', min_niter=0, max_niter=100000):
    p = specparam
    if yfe:
        yfesuffix = '_yfe'
    else:
        yfesuffix = ''
    filename_grid = cv_filename(p, yfesuffix, suffix)
    filename = 'cv_nn_'+p['name']+yfesuffix
    res = pd.read_csv(filename_grid)
    # Remove the training r2 info
    res = res.drop(columns=['r2train'])
    # Force the selection of given parameters
    for par in params:
        res = res[res[par]==params[par]]
    # Keep high enough n_iterations
    res = res[(res.n_iterations > min_niter) & (res.n_iterations < max_niter)]
    # Compute mean and std by group
    params = p['cv_'+name+'_paramsnames']
    k = list(params.keys())
    res = res.groupby(k).agg({'r2valid': ['mean', 'std']}).reset_index()
    res.columns = k + ['r2valid', 'std']
    for p in params:
        # Index of max r2valid along that parameter
        res = res.sort_values(p)
        idx = res.groupby(p)['r2valid'].transform(max) == res['r2valid']
        df = res[idx]
        print(df)
        plt.figure()
        ax = sns.lineplot(x=p, y='r2valid', data=df)
        plt.errorbar(x=df[p], y=df.r2valid, yerr=df['std'], fmt='none')
        ax.set_xlabel(params[p])
        ax.set_ylabel('$R^2$')
        ax.plot()
        save_plot(filename+'_'+p)
        plt.show()
        plt.close()

def plot_var(x, y, df, nq, xlabel, ylabel, quantiles=True, xlim=None, ylim=None, filename=None):
    # Look at subsets if specified
    if xlim is not None:
        df = df[df[x]>xlim[0]]
        df = df[df[x]<xlim[1]]
    if ylim is not None:
        df = df[df[y]>ylim[0]]
        df = df[df[y]<ylim[1]]
    # Remove NA
    dfuse = df[[x, y]].dropna()
    plt.figure()
    if quantiles:
        filename += '_quant'
        dfuse['mean'] = dfuse.groupby(pd.qcut(dfuse[x], nq, duplicates='drop'))[x].transform('mean')
        # Remove the first and last quantile
        dfuse = dfuse[(dfuse['mean'] > dfuse['mean'].min()) & (dfuse['mean'] < dfuse['mean'].max())]
    else:
        dfuse['mean'] = dfuse.groupby(pd.cut(dfuse[x], nq, duplicates='drop'))[x].transform('mean')
    print('##### Plot for x:'+x+', y:'+y+' #####')
    ax = sns.lineplot(x='mean',y=y,data=dfuse, color='black', ci=95)
    #ax = sns.lineplot(y=y,data=df, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if filename:
        save_plot(filename)
    plt.show()
    plt.close()

def plot_all_vars(var, df, specparam, nq=20, quantiles=True, xlim=None, ylim=None, sample='train'):
    p = specparam
    if sample == 'train':
        data = train_sample(df, specparam)
    elif sample == 'test':
        data = test_sample(df, specparam)
    elif sample == 'full':
        data = df
    else:
        raise Exception("sample should be 'train', 'test' or 'full'." )
    #for i in range(0, len(p['features'])):
    for f in p['full_features']:
        n = p['features_names'][f]
        #filename_varf = f+'_'+var+'_'+sample
        filename_fvar = var+'_'+f+'_'+sample
        #plot_var(var, f, data, nq, 'C score', n, quantiles=quantiles, xlim=xlim, ylim=ylim, filename=filename_varf)
        plot_var(f, var, data, nq, n, 'C score', quantiles=quantiles, xlim=xlim, ylim=ylim, filename=filename_fvar)

def plot_time(df, specparam, var, feature=False, estimator='mean', ci=None, suffix='',
              add_events=False):
    p = specparam
    spec = p['name']
    filename = 'time_'+var+suffix
    # Yearly FM
    if feature:
        varname = p['features_names'][var]
    else:
        varname = 'C Score'
    data = df[['fyear', var]]
    #data = data[data.fyear>min(data.fyear)]
    data.columns = ['Year', varname]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.lineplot(data=data, x='Year', y=varname, estimator=estimator, ci=ci)
    if add_events:
        # SOX (2002 but in effect mid-2004)
        plt.axvline(2004, -1, 1, linestyle=":", color='grey')
        plt.text(2004.1, .9, 'SOX', transform=ax.get_xaxis_transform())
        # Financial Crisis (2007-2008)
        plt.axvline(2007, -1, 1, linestyle=":", color='grey')
        plt.text(2007.1, .9, 'Financial Crisis', transform=ax.get_xaxis_transform())

    save_plot(filename)
    plt.show()
    plt.close()

def plot_time_coefs(specparam, fullsample=False, suffix=''):
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)
    if fullsample:
        filename = 'time_coefs_'+p['name']+'_fullsample'
    else:
        filename = 'time_coefs_'+p['name']
    if p['name'] == 'replication':
        names = {'yretmXDXlogeq': 'Size', 'yretmXDXm_b': 'M/B', 'yretmXDXflev': 'Leverage'}
    else:
        names = {'yretmXDXlogeqdef': 'Size (defl.)', 'yretmXDXm_b': 'M/B', 'yretmXDXflev': 'Leverage'}

    coefs = pd.read_csv(folder_fm+'coefficients.csv')

    dc = pd.DataFrame(columns=['Year', 'Value', 'Coefficient'])
    for n in names:
        dcc = coefs[['fyear', n]]
        dcc.columns = ['Year', 'Value']
        dcc['Coefficient'] = names[n]
        dc = pd.concat([dc, dcc])
    sns.lineplot(data=dc, x='Year', y='Value', hue='Coefficient')
    save_plot(filename)
    plt.show()
    plt.close()

#############################
# Tables creation functions #
#############################

def save_star(star, filename):
    # Hack the stargazer output a little bit
    starlatex = star.render_latex(only_tabular=True)
    ## Remove the cline in the header and add proper midrules
    slines = starlatex.split('\n')
    slines[4] = '\cr'
    ### Create proper midrules
    if star.column_separators is not None:
        line_midrule = ''
        s_start = 2
        for s in star.column_separators:
            s_end = s_start + s - 1
            if s > 1:
                line_midrule += ' \cmidrule{'+str(s_start)+'-'+str(s_end)+'} '
            s_start = s_end + 1
        slines_new = slines[:6]+[line_midrule]+slines[6:]
    else:
        slines_new = slines
    starlatex = '\n'.join(slines_new)
    # Write the stargazer table to file
    f = open(folder_tables+filename+'.tex', 'w')
    f.write(starlatex)
    f.close()
    # Write to HTML as well for reference
    starhtml = star.render_html()
    f = open(folder_html+filename+'.html', 'w')
    f.write(starhtml)
    f.close()

def obs_stats(data, specparam_basic, specparam_kw):
    # Data Description
    pkw = specparam_kw
    pba = specparam_basic

    filename = 'obs_stats'

    ## Fiscal Years
    min_fyear = data.fyear.min()
    max_fyear = data.fyear.max()
    ## Firm-year and unique firms full sample
    min_fyear = data.fyear.min()
    max_fyear = data.fyear.max()
    nobs = len(data[['gvkey', 'fyear']].drop_duplicates())
    nfirms = len(data.gvkey.drop_duplicates())

    ## Same for smaller sample smaller sample
    data_small = data[['gvkey', 'fyear']+pkw['features']].dropna()
    min_fyear_small = data.fyear.min()
    max_fyear_small = data.fyear.max()
    nobs_small = len(data_small[['gvkey', 'fyear']].drop_duplicates())
    nfirms_small = len(data_small.gvkey.drop_duplicates())

    ## Variables used
    fba = pba['features']
    fkw_add = [f for f in pkw['features'] if f not in fba]
    vars_ba = ', '.join([pba['features_names'][f] for f in fba])
    vars_kw_add = ', '.join([pkw['features_names'][f] for f in fkw_add])

    ## Models names in paper
    models_full_l1 = 'L1, L1y'
    models_full_l2 = 'MLC1, MLC1y'
    models_small_l1 = 'L2, L2y'
    models_small_l2 = 'MLC2, MLC2y'

    # Create the table
    dt_list = [[str(min_fyear)+' - '+str(max_fyear), models_full_l1, nobs, nfirms, vars_ba],
               ['', models_full_l2, '', '', ''],
               [str(min_fyear_small)+' - '+str(max_fyear_small), models_small_l1, nobs_small, nfirms_small, vars_ba],
               ['', models_small_l2, '', '', vars_kw_add]]
    dt = pd.DataFrame(dt_list, index=['Full Sample', '', 'Restricted Sample', ''], columns=['Timespan', 'Models', 'Firm-year obs.', 'Unique firms', 'Features'])
    dt.to_latex(folder_tables+filename+'.tex', escape=True, na_rep='')
    return dt

def desc_stats(df, specparam, suffix=''):
    p = specparam
    filename='desc_stat'+suffix
    cols = [p['earningsvar'], p['retvar']] + p['full_features']
    # Remove size deflated
    cols.remove('logeqdef')
    names = [p['features_names'][i] for i in cols]
    #if variables is None:
    #else:
    #    cols = variables.keys()
    #    names = variables.values()
    #    filename = 'desc_stat_'+'scores'+suffix
    ds = df[cols].describe()
    ds.columns = names
    ds = ds.transpose()
    ds = ds.astype({'count': int})
    ds.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, escape=True)
    return ds

def correlation_matrix(df, star=.05):
    c = pd.DataFrame(columns=df.columns, index=df.columns)

    # Compute Pearson correlation in upper triangular matrix
    cols = list(df.columns)
    rows = list(df.columns)
    for row in rows:
        cols.remove(row)
        for col in cols:
            dft = df[[row, col]].dropna()
            r, p = stats.pearsonr(dft[row], dft[col])
            c.loc[row, col] = format(r, '.3f')
            if p < star:
                c.loc[row, col] += '*'
            else:
                c.loc[row, col] += ' '

    # Compute Spearman correlation in lower triangular matrix
    cols = list(df.columns)
    rows = list(df.columns)
    for col in cols:
        rows.remove(col)
        for row in rows:
            r, p = stats.spearmanr(df[row], df[col], nan_policy='omit')
            c.loc[row, col] = format(r, '.3f')
            if p < star:
                c.loc[row, col] += '*'
            else:
                c.loc[row, col] += ' '

    return c

def desc_corr(df, specparam, suffix=""):
    p = specparam
    filename='desc_corr'+suffix
    cols_desc = [p['earningsvar'], p['retvar']] + p['full_features']
    # Remove size deflated
    cols_desc.remove('logeqdef')
    cols_names = [p['features_names'][i] for i in cols_desc]
    corr = correlation_matrix(df[cols_desc])
    corr.columns = corr.index = cols_names
    corr.to_latex(folder_tables+filename+'.tex', escape=True, na_rep='', multicolumn_format='r')
    return corr

def regression_cscore_lags(df, cscore):
    filename = 'cscore_lags_'+cscore
    # Create years fixed effects
    df['fyear'] = df.fyear.astype(int)
    yfe = pd.get_dummies(df['fyear'], prefix='y')
    years = list(yfe.columns)
    df = pd.concat([df, pd.get_dummies(df['fyear'], prefix='y')], axis=1)
    # Compute lags and create formulas
    df = df.sort_values(['gvkey', 'fyear'])
    df['cscore'] = df[cscore]
    formula = 'cscore ~ 1 '
    formulas = []
    for lag in [1,2,3]:
        n = 'cscore'+str(lag)
        df[n] = df.groupby('gvkey')['cscore'].shift(lag)
        formula += ' + '+n
        formulas += [formula]

    yfe = '+'.join(str(y) for y in years)
    formulas += [f+' + '+yfe for f in formulas]
    # Run the regressions
    res = [smf.ols(f, data=df).fit() for f in formulas]
    indvars = {'cscore1': 'C score (t-1)', 'cscore2': 'C score (t-2)', 'cscore3': 'C score (t-3)', 'Intercept': 'Intercept'}

    # Output table
    star = Stargazer(res)
    star.dependent_variable_name('C Score')
    #star.dep_var_name = None
    #star.custom_columns(colnames, [2 for c in colnames])
    star.covariate_order(indvars.keys())
    star.rename_covariates(indvars)
    star.add_line('Years FE', ['No', 'No', 'No', 'Yes', 'Yes', 'Yes'], location='ft')
    star.show_f_statistic = False
    star.show_adj_r2 = False
    star.show_residual_std_err = False
    # Write to file
    save_star(star, filename)
    return(star)

def fit_table(models, labels, ncoefs, suffix=''):
    filename = 'fit'+suffix
    dft = []
    dff = []
    for m in models['train_models']:
        dft += [pd.DataFrame(m, ['R2'])]
    for m in models['full_models']:
        dff += [pd.DataFrame(m, ['R2'])]
        
    df = pd.DataFrame()
    df['{\\bf R squared}'] = np.nan
    df[['Training Sample', 'Test Sample']] = pd.concat(dft, axis=0)
    df[['Full Sample']] = pd.concat(dff, axis=0)
    df['{\\bf Number of coefficients}'] = np.nan
    
    df = df.set_index(pd.Index(labels))
    df = df.transpose()
    
    df = pd.concat([df, ncoefs])
    textable = df.to_latex(float_format="{:0.3f}".format, na_rep='', multicolumn_format='c', escape=False)
    textable = textable.replace('.000', '')
    file = open(folder_tables+filename+'.tex', 'wt')
    file.write(textable)
    file.close()
    
    return df

def corr_table(variables, specparam, legend, df, suffix=''):
    p = specparam
    filename = 'corr_cscores'+suffix
    #cvars = variables + p['full_features']
    features = ['logeq', 'm_b', 'flev', 'yvolatd', 'problit', 'pin', 'dbas']
    cvars = variables + features
    clegend = legend + [p['features_names'][f] for f in features]

    corr = correlation_matrix(df[cvars])
    corr.columns = corr.index = clegend
    corr.to_latex(folder_tables+filename+'.tex', escape=True, na_rep='', multicolumn_format='r')

    #pears = df[cvars].corr(method='pearson')
    #spear = df[cvars].corr(method='spearman')
    #shape = pears.shape[0]
    #corr_matrix = (np.tril(spear) + np.triu(pears)) - np.eye(shape)
    #np.fill_diagonal(corr_matrix, np.nan)
    #c = pd.DataFrame(corr_matrix, columns=pears.columns, index=spear.columns)
    #colsname = {cvars[i]: clegend[i] for i in range(0,len(cvars))}
    #c = c.rename(columns=colsname, index=colsname)
    #c.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, escape=True, na_rep='')
    return corr

def regression(variables, legend, df, specparam, drop_features, suffix=''):
    p = specparam
    depname = 'C Score'
    features = p['full_features'].copy()
    # Remove size non-deflated and add size deflated ?
    features.remove('logeq')
    varsize = 'logeqdef'

    for f in drop_features:
        features.remove(f)
    covnames = {f: p['features_names'][f] for f in features}
    colnames = legend
    filename = 'reg_'+p['name']+suffix

    # Copy the data first
    dfc = df[variables+features].dropna()

    # Compute the polynomials (only 2nd degree)
    nf = len(features)
    features2 = []
    deg = 2
    for f in features:
        fd = f+str(deg)
        dfc[fd] = dfc[f]**deg
        features2 += [fd]
        covnames[fd] = covnames[f]+'$^'+str(deg)+'$'

    # Compute interaction variables
    features_int = []
    ## Size X MB
    dfc[varsize+'Xm_b'] = dfc[varsize] * dfc.m_b
    features_int += [varsize+'Xm_b']
    covnames[varsize+'Xm_b'] = 'Size X M/B'
    ## Size X Leverage
    dfc[varsize+'Xflev'] = dfc[varsize] * dfc.flev
    features_int += [varsize+'Xflev']
    covnames[varsize+'Xflev'] = 'Size X Leverage'
    ## MB X Leverage
    dfc['m_bXflev'] = dfc.m_b * dfc.flev
    features_int += ['m_bXflev']
    covnames['m_bXflev'] = 'M/B X Leverage'
    ## Size X Volat
    dfc[varsize+'Xyvolatd'] = dfc[varsize] * dfc.yvolatd
    features_int += [varsize+'Xyvolatd']
    covnames[varsize+'Xyvolatd'] = 'Size X Volatility'
    ## MB X Volat
    dfc['m_bXyvolatd'] = dfc.m_b * dfc.yvolatd
    features_int += ['m_bXyvolatd']
    covnames['m_bXyvolatd'] = 'M/B X Volatility'
    ## Leverage X Volat
    dfc['flevXyvolatd'] = dfc.flev * dfc.yvolatd
    features_int += ['flevXyvolatd']
    covnames['flevXyvolatd'] = 'Leverage X Volatility'

    # Standardize (zscore)
    dfc[list(covnames.keys())] = dfc[list(covnames.keys())].apply(stats.zscore)


    rhs_features = '+'.join(features)
    rhs_features2 = '+'.join(features2)
    rhs_features_int = '+'.join(features_int)

    rhs = [rhs_features,
           rhs_features+'+'+rhs_features2,
           rhs_features+'+'+rhs_features_int]#,
           #rhs_features+'+'+rhs_features2+'+'+rhs_features_int]


    res = [smf.ols(variables[0]+'~'+rhs_features, data=dfc).fit()]
    res += [smf.ols(variables[1]+'~'+r, data=dfc).fit() for r in rhs]
    res += [smf.ols(variables[2]+'~'+r, data=dfc).fit() for r in rhs]
    res += [smf.ols(variables[3]+'~'+r, data=dfc).fit() for r in rhs]


    star = Stargazer(res)
    star.dependent_variable_name(depname)
    star.custom_columns(colnames, [1, 3, 3, 3])
    star.covariate_order(list(covnames.keys())+['Intercept'])
    star.rename_covariates(covnames)
    star.show_f_statistic = False
    star.show_adj_r2 = False
    star.show_residual_std_err = False
    # Write to file
    save_star(star, filename)
    return star

def generate_data_linear(m, df, specparam):
    p = specparam
    data = df.copy()
    # Keep useful columns
    #missing_features = [f for f in p['full_features'] if f not in p['features_all']]
    data = data[['gvkey', 'fyear', p['earningsvar'], p['retvar'], *p['full_features']]]
    data = data.dropna(subset=[p['earningsvar'], *p['features']])
    # Create dataset
    X, Y = get_sets(data, p, p['features_all_int'], yfe=False)
    #  R = data[p['retvar']]
    #  D = (R<0.).astype('float')
    # Compute the predicted earnings
    data['X'] = m.predict(Y)
    # Compute the stddev of residuals
    noise = np.std(data.X-data[p['earningsvar']])
    # Create the new earnings
    data[p['earningsvar']] = data.X + np.random.normal(0, noise, len(data))
    # Remove unecessary columns
    data = data.drop(columns=['X'])
    return (data)

def generate_data_fmy(df, specparam, fullsample=True, suffix=''):
    df = df.copy()
    p = specparam
    folder_fm = _get_folder_fm(p, fullsample, suffix)

    # Prepare the data
    data_lab = full_sample(df, p)
    data_lab = create_interactions(data_lab, p)
    # Create data using yearly linear regressions estimations
    min_fyear = int(min(df.fyear))
    max_fyear = int(max(df.fyear))
    # Create the dataframe to store the generated data
    missing_features = [f for f in p['full_features'] if f not in p['features_all']]
    data = pd.DataFrame(None,['gvkey', 'fyear', p['earningsvar'], *p['features_all_int'], *missing_features]).transpose()
    for y in range(min_fyear, max_fyear+1):
        dfy = data_lab[data_lab.fyear==y][['gvkey', 'fyear', p['earningsvar'], *p['features_all_int'], *missing_features]]
        # Load the model for that year
        m = joblib.load(folder_fm+'model_'+str(y)+'.sav')
        #  # Compute the C score (for positivity constraints)
        #  dfy['c'] = c(m, dfy[[p['earningsvar'], p['retvar'], *p['features_all']]], p)
        #  # Compute the G score (for positivity constraints)
        #  dfy['g'] = g(m, dfy[[p['earningsvar'], p['retvar'], *p['features_all']]], p)
        # Compute the predicted earnings
        X, Y = get_sets(dfy, p, p['features_all_int'])
        dfy['X'] = m.predict(Y)
        # Compute the stddev of residuals
        noise = np.std(dfy.X-dfy[p['earningsvar']])
        # Create the new earnings
        dfy[p['earningsvar']] = dfy.X + np.random.normal(0, noise, len(dfy))
        #  # Remove observations with negative g or c+g score
        #  dfy = dfy[dfy.g>=0]
        #  dfy = dfy[(dfy.c+dfy.g>=0)]
        #  # Remove unecessary columns
        #  dfy = dfy.drop(columns=['g', 'c', 'X'])
        # Add to the final dataframe
        data = pd.concat([data, dfy])
    return(data)

def generate_data_nn(df, specparam, yfe=True):
    p = specparam
    data = df.copy()
    # Keep useful columns
    data = data[['gvkey', 'fyear', p['earningsvar'], p['retvar'], *p['full_features']]]
    # Compute predicted earnings
    data['X'] = predict_average_nn(data, p, fullsample=True, yfe=yfe, nmodels=100)
    # Compute the stddev of residuals
    noise = np.std(data.X-data[p['earningsvar']])
    # Create the new earnings
    data[p['earningsvar']] = data.X + np.random.normal(0, noise, len(data))
    # Remove unecessary columns
    data = data.drop(columns=['X'])
    return (data)

def compute_errors(cscores, labels, dfs):
    #filename = 'laboratory_errors'
    dferr = pd.DataFrame()
    for i in range(len(dfs)):
        df = dfs[i]
        errors = np.mean(np.abs(df[cscores].sub(df.c.squeeze(), axis=0)), axis=0)
        # Divide by std of the c score to normalize
        errors = errors / np.std(df.c)
        dfe = pd.DataFrame(errors, columns=[cscores[i]])
        dferr = pd.concat([dferr, dfe], axis=1)
    dgp_index = pd.MultiIndex.from_product([['DGP'], labels])
    est_index = pd.MultiIndex.from_product([['Estimation'], labels])
    dferr.index = est_index
    dferr.columns = dgp_index
    #dferr.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, multicolumn_format='c', escape=True)
    return(dferr)

def compute_coefs(dfs, specparam, vars_cscore, legend, noise_var=0, coef_cscore=3):
    p = specparam
    #filename = 'laboratory_noise'+str(noise_var)
    filename_reg = 'laboratory_coefs'+str(noise_var)+'_regression'
    coefs = {'logeqdef': 4, 'm_b': -2, 'flev': 1}#, 'yvolatd': 4.3, 'noacc': -8, 'cfoa': 2.3, 'invcycle': -1.2, 'problit': 3.7, 'age': -6.1}
    # Define intercept
    coef_intercept = 2

    def gen_data(df):
        # Generate data
        data = df.copy()
        ## Variable and covariate names
        ### All coefs except C score
        data['var'] = coef_intercept
        for coef in coefs:
            data['var'] += coefs[coef] * data[coef]
        names['Intercept'] = 'Intercept'
        ### C score
        data['var'] += coef_cscore * data['c']
        ### Add some noise
        data['var'] += np.random.normal(0, noise_var, len(data))
        # Return data and names
        return data

    names = {f: p['features_names'][f] for f in coefs}
    for coef in coefs:
        names[coef] = names[coef] + ' [coef = ' + str(coefs[coef]) + ']'
    for i in range(len(vars_cscore)):
        names[vars_cscore[i]] = 'C score '+legend[i]+' [coef = ' + str(coef_cscore) + ']'
    names['Intercept'] = 'Intercept'+' [coef = ' + str(coef_intercept) + ']'

    res = []
    values = []
    pvalues = []
    stderrs = []
    for df in dfs:
        data_lab = gen_data(df)
        for var_cscore_fit in vars_cscore:
            # Regress the variable using a given cscore measure
            # Covariates
            covariates = [var_cscore_fit]+list(coefs.keys())
            reg = smf.ols('var ~ '+'+'.join(covariates), data=data_lab).fit()
            res += [reg]
            values += [reg.params[1]]
            pvalues += [reg.pvalues[1]]
            stderrs += [reg.bse[1]]

    all_covariates = vars_cscore + list(coefs.keys()) + ['Intercept']

    # Create regression tables for reference
    star = Stargazer(res)
    star.dependent_variable_name('Variable')
    star.covariate_order(all_covariates)
    star.rename_covariates(names)
    star.custom_columns(legend, [len(vars_cscore) for i in vars_cscore])
    star.show_f_statistic = False
    star.show_adj_r2 = False
    star.show_residual_std_err = False
    # Write to file
    save_star(star, filename_reg)

    # Create matrices
    dgp_index = pd.MultiIndex.from_product([['DGP'], legend])
    est_index = pd.MultiIndex.from_product([['Estimation'], legend])
    ## Create the matrix of C score coefs
    coefs_array = np.array(values)
    coefs_matrix = coefs_array.reshape((len(vars_cscore), len(vars_cscore)))
    ct = pd.DataFrame(coefs_matrix, columns=legend, index=legend).transpose()
    ct.index = est_index
    ct.columns = dgp_index
    ## Create the matrix of P-values
    coefs_array = np.array(pvalues)
    coefs_matrix = coefs_array.reshape((len(vars_cscore), len(vars_cscore)))
    pt = pd.DataFrame(coefs_matrix, columns=legend, index=legend).transpose()
    pt.index = est_index
    pt.columns = dgp_index
    ## Create the matrix of standard errors
    coefs_array = np.array(stderrs)
    coefs_matrix = coefs_array.reshape((len(vars_cscore), len(vars_cscore)))
    st = pd.DataFrame(coefs_matrix, columns=legend, index=legend).transpose()
    st.index = est_index
    st.columns = dgp_index

    # Write to file
    #ct.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, multicolumn_format='c', escape=True, na_rep='')
    return ct, pt, st

def table_with_se(means, pvalues, stderrs):
    ''' Return a dataframe with significance stars and standard errors '''
    # Create an index that includes rows for standard errors
    index = []
    for ci in means.index:
        index += [ci]
        index += [(ci[0], '')]
    index = pd.MultiIndex.from_tuples(index)

    table = pd.DataFrame(index=index, columns=means.columns)

    for r in range(len(pvalues.index)):
        for c in range(len(pvalues.columns)):
            table.iloc[r*2,c] = means.iloc[r,c].round(3).astype(str)
            table.iloc[r*2+1,c] = '('+stderrs.iloc[r,c].round(3).astype(str)+')'
            if pvalues.iloc[r,c] < 0.1:
                table.iloc[r*2,c] += '*'
            if pvalues.iloc[r,c] < 0.05:
                table.iloc[r*2,c] += '*'
            if pvalues.iloc[r,c] < 0.01:
                table.iloc[r*2,c] += '*'

    return table

def table_with_tstat(means, pvalues, tstats):
    ''' Return a dataframe with significance stars and standard errors '''
    # Create an index that includes rows for standard errors
    index = []
    for ci in means.index:
        index += [ci]
        index += [(ci[0], '')]
    index = pd.MultiIndex.from_tuples(index)

    table = pd.DataFrame(index=index, columns=means.columns)

    for r in range(len(pvalues.index)):
        for c in range(len(pvalues.columns)):
            table.iloc[r*2,c] = means.iloc[r,c].round(3).astype(str)
            table.iloc[r*2+1,c] = '['+tstats.iloc[r,c].round(3).astype(str)+']'
            if pvalues.iloc[r,c] < 0.1:
                table.iloc[r*2,c] += '*'
            if pvalues.iloc[r,c] < 0.05:
                table.iloc[r*2,c] += '*'
            if pvalues.iloc[r,c] < 0.01:
                table.iloc[r*2,c] += '*'

    return table

def table_lab(values, pvalues=None, stderrs=None, tstats=None, suffix=''):
    filename = 'laboratory'+suffix

    if pvalues is None:
        # No stars or std errors
        values.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}".format, multicolumn_format='c', escape=True)
    else:
        #values = table_with_se(values, pvalues, stderrs)
        values = table_with_tstat(values, pvalues, tstats)
        values.to_latex(folder_tables+filename+'.tex', float_format="{:0.3f}", multicolumn_format='c', escape=True)

    return values

def table_lab_robust(n_iter, table='coefs', noise_var=0, coef_cscore=0):
    file_lab = folder_models+'laboratory_data.h5'

    # Open the file
    store = pd.HDFStore(folder_models+'laboratory_data.h5', mode='r')


    ##### Create table with multiple draws #####
    
    # Create index for all tables
    values0 = pd.read_hdf(store, key=table+'_noise'+str(noise_var)+'coef'+str(coef_cscore)+'iter0')
    # Remove the L1 estimation (first row)
    values0 = values0[1:]
    index_orig = values0.index
    columns_orig = values0.columns
    estimations = values0.index.get_level_values(1)
    columns = values0.columns.get_level_values(1)
    tuples = []
    for e in estimations:
        for c in columns:
            tuples += [(e, c)]
    index = pd.MultiIndex.from_tuples(tuples, names=['Estimation', 'DGP'])

    # Create DataFrames
    values = pd.DataFrame(index=index)
    iteration = 0
    for iteration in range(n_iter):
        values_i = pd.read_hdf(store, key=table+'_noise'+str(noise_var)+'coef'+str(coef_cscore)+'iter'+str(iteration))
        # Remove L1 estimation (frst row)
        values_i = values_i[1:]
        values[iteration] = values_i.values.flatten()

    # Close the file
    store.close()

    ##### Compute means, standard errors and p-values #####

    # Compute means
    values_m = pd.DataFrame(values.mean(axis=1))

    # Compute standard errors and pvalues
    Nobs = len(values.columns)
    values_sd = np.sqrt(((values.sub(values_m.squeeze(), axis=0))**2).sum(axis=1) / (Nobs-1))
    values_se = values_sd / np.sqrt(Nobs)
    values_tstat = values_m.squeeze() / values_se
    values_pvalues = stats.t.sf(np.abs(values_tstat), Nobs-1)

    # Reshape to matrices
    n = len(index_orig)
    values_m = pd.DataFrame(values_m.values.reshape(values0.shape), index=index_orig, columns=columns_orig)
    values_se = pd.DataFrame(values_se.values.reshape(values0.shape), index=index_orig, columns=columns_orig)
    values_pvalues = pd.DataFrame(values_pvalues.reshape(values0.shape), index=index_orig, columns=columns_orig)
    values_tstat = pd.DataFrame(values_tstat.values.reshape(values0.shape), index=index_orig, columns=columns_orig)

    return table_lab(values_m, values_pvalues, values_se, values_tstat, suffix='_robust_'+table+'_noise'+str(noise_var)+'coef'+str(coef_cscore))

def table_lab_sig(n_iter, noise_var=0, coef_cscore=0):
    ''' Compute fraction of significant coefs from the draws. '''
    file_lab = folder_models+'laboratory_data.h5'

    # Open the file
    store = pd.HDFStore(folder_models+'laboratory_data.h5', mode='r')


    ##### Create table with multiple draws #####

    # Create index for all tables
    pvalues0 = pd.read_hdf(store, key='pvalues_noise'+str(noise_var)+'coef'+str(coef_cscore)+'iter0')
    # Remove L1 estimation
    pvalues0 = pvalues0[1:]
    index_orig = pvalues0.index
    columns_orig = pvalues0.columns
    estimations = pvalues0.index.get_level_values(1)
    columns = pvalues0.columns.get_level_values(1)
    tuples = []
    for e in estimations:
        for c in columns:
            tuples += [(e, c)]
    index = pd.MultiIndex.from_tuples(tuples, names=['Estimation', 'DGP'])

    # Create DataFrames
    pvalues = pd.DataFrame(index=index)
    iteration = 0
    for iteration in range(n_iter):
        pvalues_i = pd.read_hdf(store, key='pvalues_noise'+str(noise_var)+'coef'+str(coef_cscore)+'iter'+str(iteration))
        # Remove L1 estimation
        pvalues_i = pvalues_i[1:]
        pvalues[iteration] = pvalues_i.values.flatten()

    # Close the file
    store.close()

    ##### Compute Fraction of significant coefs #####
    sig = (pvalues<.01).astype(int)
    sig_m = sig.mean(axis=1)

    # Reshape to matrices
    n = len(index_orig)
    sig_m = pd.DataFrame(sig_m.values.reshape(pvalues0.shape), index=index_orig, columns=columns_orig)

    # Write to file
    return table_lab(sig_m, suffix='_robust_sig'+'_noise'+str(noise_var)+'coef'+str(coef_cscore))

def laboratory(data, L_basic, L_kw, specparam_basic, specparam_kw, iteration, noise_vars=[0, 5, 8, 10], coefs_cscore=[0, 3, 5]):

    # Decide wether to save the models or not
    suffixdgp = ''
    if iteration is not None:
        suffixdgp = '_iter'+str(iteration)

    # Define file to save the robust results
    file_lab = folder_models+'laboratory_data.h5'

    # Function to give the key of the result in HDF store
    def key(name, noise_var, coef_cscore):
        return name+'_noise'+str(noise_var)+'coef'+str(coef_cscore)+'iter'+str(iteration)

    ###################
    # Hyperparameters #
    ###################
    opt_basic, opt_basic_yfe, opt_kw, opt_kw_yfe = hyperparameters()

    #---------------------------------------#
    #---- Generate Data and define sets ----#
    #---------------------------------------#

    print('### Generate Data ###')

    # Generate Data
    ## Basic Specification
    print('Basic Linear', end='\r')
    data_linear = generate_data_linear(L_basic, data, specparam_basic)
    print('Basic FM', end='\r')
    data_fmy = generate_data_fmy(data, specparam_basic)
    print('Basic NN', end='\r')
    data_nn = generate_data_nn(data, specparam_basic, yfe=False)
    print('Basic NN FE', end='\r')
    data_nny = generate_data_nn(data, specparam_basic, yfe=True)
    ## KW Specification
    print('KW Linear', end='\r')
    datakw_linear = generate_data_linear(L_kw, data, specparam_kw)
    print('KW FM', end='\r')
    datakw_fmy = generate_data_fmy(data, specparam_kw)
    print('KW NN', end='\r')
    datakw_nn = generate_data_nn(data, specparam_kw, yfe=False)
    print('KW NN FE', end='\r')
    datakw_nny = generate_data_nn(data, specparam_kw, yfe=True)
    # Actual C scores
    ## Basic specification
    print('Basic C score Linear', end='\r')
    data_linear['c'] = c(L_basic, data_linear, specparam_basic)
    print('Basic C score FM', end='\r')
    data_fmy['c'] = c_score_fmy(data_fmy, specparam_basic)
    print('Basic C score NN', end='\r')
    data_nn['c'] = c_score_nn(data_nn, specparam_basic, fullsample=True, yfe=False)
    print('Basic C score NN FE', end='\r')
    data_nny['c'] = c_score_nn(data_nny, specparam_basic, fullsample=True, yfe=True)
    ## KW specification
    print('KW C score Linear', end='\r')
    datakw_linear['c'] = c(L_kw, datakw_linear, specparam_kw)
    print('KW C score FM', end='\r')
    datakw_fmy['c'] = c_score_fmy(datakw_fmy, specparam_kw)
    print('KW C score NN', end='\r')
    datakw_nn['c'] = c_score_nn(datakw_nn, specparam_kw, fullsample=True, yfe=False)
    print('KW C score NN FE', end='\r')
    datakw_nny['c'] = c_score_nn(datakw_nny, specparam_kw, fullsample=True, yfe=True)


    # Define all the datasets and labels
    dfs = [data_linear, data_fmy, data_nn, data_nny, datakw_linear, datakw_fmy, datakw_nn, datakw_nny]
    dgps = ['_dgp_linear_basic', '_dgp_fmy_basic', '_dgp_nn_basic', '_dgp_nny_basic', '_dgp_linear_kw', '_dgp_fmy_kw', '_dgp_nn_kw', '_dgp_nny_kw']
    cscores = ['c_L_basic', 'c_Lfmy_basic', 'c_NN_basic', 'c_NNy_basic', 'c_L_kw', 'c_Lfmy_kw', 'c_NN_kw', 'c_NNy_kw']
    labels = ['L1', 'L1y', 'MLC1', 'MLC1y', 'L2', 'L2y', 'MLC2', 'MLC2y']

    #------------------------------------#
    #---- Compute Estimation Methods ----#
    #------------------------------------#

    print('### Compute Estimation Methods ###')

    # FM Yearly
    for i in range(len(dfs)):
        df = dfs[i]
        dgp = dgps[i]
        print('Estimate linear models for DGP: '+str(dgp))
        #print('Linear basic for DGP: '+str(dgp))
        L_basic, _ = train_linear(df, specparam_basic, fullsample=True, recompute=True, suffix=dgp+suffixdgp)
        df['c_L_basic'] = c(L_basic, df, specparam_basic)
        #print('Linear KW for DGP: '+str(dgp))
        L_kw, _ = train_linear(df, specparam_kw, fullsample=True, recompute=True, suffix=dgp+suffixdgp)
        df['c_L_kw'] = c(L_kw, df, specparam_kw)
        #print('FM basic for DGP: '+str(dgp))
        compute_fm_linear(df, specparam_basic, fullsample=True, suffix=dgp+suffixdgp)
        df['c_Lfmy_basic'] = c_score_fmy(data_linear, specparam_basic, fullsample=True, suffix=dgp+suffixdgp)
        #print('FM KW for DGP: '+str(dgp))
        compute_fm_linear(df, specparam_kw, fullsample=True, suffix=dgp+suffixdgp)
        df['c_Lfmy_kw'] = c_score_fmy(data_linear, specparam_kw, fullsample=True, suffix=dgp+suffixdgp)

    # Neural Networks
    for i in range(len(dfs)):
        df = dfs[i]
        dgp = dgps[i]
        print('Estimate neural networks for DGP: '+str(dgp))
        #print('NN basic for DGP: '+str(dgp))      
        train_nn(df, specparam_basic, params=opt_basic, fullsample=True, recompute=True, yfe=False, suffix=dgp+suffixdgp, nmodels=1)
        df['c_NN_basic'] = c_score_nn(df, specparam_basic, fullsample=True, yfe=False, suffix=dgp+suffixdgp, nmodels=1)        
        #print('NNy basic for DGP: '+str(dgp))
        train_nn(df, specparam_basic, params=opt_basic, fullsample=True, recompute=True, yfe=True, suffix=dgp+suffixdgp, nmodels=1)
        df['c_NNy_basic'] = c_score_nn(df, specparam_basic, fullsample=True, yfe=True, suffix=dgp+suffixdgp, nmodels=1)        
        #print('NN kw for DGP: '+str(dgp))
        train_nn(df, specparam_kw, params=opt_kw, fullsample=True, recompute=True, yfe=False, suffix=dgp+suffixdgp, nmodels=1)
        df['c_NN_kw'] = c_score_nn(df, specparam_kw, fullsample=True, yfe=False, suffix=dgp+suffixdgp, nmodels=1)
        #print('NNy kw for DGP: '+str(dgp))
        train_nn(df, specparam_kw, params=opt_kw, fullsample=True, recompute=True, yfe=True, suffix=dgp+suffixdgp, nmodels=1)
        df['c_NNy_kw'] = c_score_nn(df, specparam_kw, fullsample=True, yfe=True, suffix=dgp+suffixdgp, nmodels=1)


    #--------------------#
    #---- Comparison ----#
    #--------------------#

    print('### Comparisons ###')


    def comparison(noise_var, coef_cscore):
        print('Coef C score: '+str(coef_cscore)+' Noise var:'+str(noise_var))
        errors = compute_errors(cscores, labels, dfs)
        coefs, pvalues, stderrs = compute_coefs(dfs, specparam_basic, cscores, labels, noise_var=noise_var, coef_cscore=coef_cscore)
        # Write to file
        errors.to_hdf(file_lab, key=key('errors', noise_var, coef_cscore))
        coefs.to_hdf(file_lab, key=key('coefs', noise_var, coef_cscore))
        pvalues.to_hdf(file_lab, key=key('pvalues', noise_var, coef_cscore))
        stderrs.to_hdf(file_lab, key=key('stderrs', noise_var, coef_cscore))


    if iteration is not None:
        for noise_var in noise_vars:
            for coef_cscore in coefs_cscore:
                comparison(noise_var, coef_cscore)
    else:
        # Should have only one noise_var and coefs_cscore
        noise_var = noise_vars[0]
        coef_cscore = coefs_cscore[0]
        print('Coef C score: '+str(coef_cscore)+' Noise var:'+str(noise_var))
        errors = compute_errors(cscores, labels, dfs)
        coefs, pvalues, stderrs = compute_coefs(dfs, specparam_basic, cscores, labels, noise_var=noise_var, coef_cscore=coef_cscore)
        return errors, coefs, pvalues, stderrs

########################
# All-in-one functions #
########################
# These functions can be run directly after importing the module
# in order to simplify running long computations.

def hyperparameters(suffix=''):
    ###################################
    # Cross-Validation Neural Network #
    ###################################
    # To perform cross-validation, run 'run_crossvalidation_nn()'
    # and analyze the plots with plot_cv() to find the hyperparameters


    if suffix=='':
        ## Hyperparameters for 'basic' specification, without and with years FE
        #opt_basic = {'n_iterations': 4000, 'layer_size': 50, 'learning_rate': .005, 'verbose': 1}
        #opt_basic = {'n_iterations': 9000, 'layer_size': 70, 'learning_rate': .005, 'verbose': 1} # 04-16-2021
        #opt_basic = {'n_iterations': 19000, 'layer_size': 30, 'learning_rate': 1e-3, 'verbose': 1} # 05-13-2021 (learning_rate=1e-3)
        opt_basic = {'n_iterations': 12000, 'layer_size': 25, 'learning_rate': 1e-3, 'verbose': 1} # 05-26-2021 (learning_rate=1e-3)
        
        #opt_basic_yfe = {'n_iterations': 4000, 'layer_size': 70, 'learning_rate': .001, 'verbose': 1}
        #opt_basic_yfe = {'n_iterations': 5000, 'layer_size': 70, 'learning_rate': .001, 'verbose': 1} # 04-16-2021
        #opt_basic_yfe = {'n_iterations': 5000, 'layer_size': 20, 'learning_rate': 1e-3, 'verbose': 1} # 05-13-2021 (learning_rate=1e-3)
        opt_basic_yfe = {'n_iterations': 4000, 'layer_size': 40, 'learning_rate': 1e-3, 'verbose': 1} # 05-26-2021 (learning_rate=1e-3)
        
        ## Hyperparameters for 'kw' specification, without and with years FE
        #opt_kw = {'n_iterations': 7000, 'layer_size': 30, 'learning_rate': .005, 'verbose': 1}
        #opt_kw = {'n_iterations': 8000, 'layer_size': 30, 'learning_rate': .005, 'verbose': 1} # 04-16-2021
        #opt_kw = {'n_iterations': 15000, 'layer_size': 30, 'learning_rate': 1e-3, 'verbose': 1} # 05-13-2021 (learning_rate=1e-3)
        opt_kw = {'n_iterations': 10000, 'layer_size': 40, 'learning_rate': 1e-3, 'verbose': 1} # 05-26-2021 (learning_rate=1e-3)
        
        #opt_kw_yfe = {'n_iterations': 4000, 'layer_size': 25, 'learning_rate': .001, 'verbose': 1}
        #opt_kw_yfe = {'n_iterations': 5000, 'layer_size': 25, 'learning_rate': .001, 'verbose': 1} # 04-16-2021
        #opt_kw_yfe = {'n_iterations': 6000, 'layer_size': 25, 'learning_rate': 1e-3, 'verbose': 1} # 05-13-2021 (learning_rate=1e-3)
        opt_kw_yfe = {'n_iterations': 4500, 'layer_size': 50, 'learning_rate': 1e-3, 'verbose': 1} # 05-26-2021 (learning_rate=1e-3)

    elif suffix=='_years':
        opt_basic = {'n_iterations': 10000, 'layer_size': 10, 'learning_rate': 1e-3, 'verbose': 1} # 04-20-2021 (learning_rate=1e-3)
        opt_basic_yfe = None
        opt_kw = {'n_iterations': 12000, 'layer_size': 10, 'learning_rate': 1e-3, 'verbose': 1} # 04-20-2021 (learning_rate=1e-3)
        opt_kw_yfe = None
    
    elif suffix=='_post2004':
        None
        
    elif suffix=='_robust_nopos':
        opt_basic = {'n_iterations': 6000, 'layer_size': 60, 'learning_rate': .02, 'verbose': 1}
        opt_basic_yfe = {'n_iterations': 5000, 'layer_size': 10, 'learning_rate': .001, 'verbose': 1}
        ## Hyperparameters for 'kw' specification, without and with years FE
        opt_kw = {'n_iterations': 4000, 'layer_size': 10, 'learning_rate': .02, 'verbose': 1}
        opt_kw_yfe = {'n_iterations': 6000, 'layer_size': 14, 'learning_rate': .001, 'verbose': 1}


    return opt_basic, opt_basic_yfe, opt_kw, opt_kw_yfe

def run_crossvalidation_nn(specification, folder, pos_constraint=True, samples='gvkeys', min_year=None, suffix=''):
    set_folder(folder)
    set_ncpu(30)
    data = prepare_data("data/data.parquet", min_year=min_year)
    # Specify the suffix
    if min_year is not None:
        suffix += '_post'+str(min_year)
    if samples=='years':
        suffix += '_years'
    # Get the specification parameters
    specparam = get_specparam(specification, samples=samples, suffix=suffix)
    # Perform the grid search
    ## Without Years FE
    print("################# Without YFE ###################")
    rescv = run_search_nn(data, specparam, recompute=True, yfe=False, pos_constraint=pos_constraint, suffix=suffix)
    ## With Years FE
    if samples != 'years':
        print("################# With YFE ###################")
        rescv = run_search_nn(data, specparam, recompute=True, yfe=True, pos_constraint=pos_constraint, suffix=suffix)

def run_crossvalidation_nn_final(specification, folder, pos_constraint=True, samples='gvkeys', min_year=None, suffix=''):
    set_folder(folder)
    set_ncpu(30)
    data = prepare_data("data/data.parquet", min_year=min_year)
    # Specify the suffix
    if min_year is not None:
        suffix += '_post'+str(min_year)
    if samples=='years':
        suffix += '_years'
    suffix_cv = suffix + '_final'
    # Get the specification parameters
    specparam = get_specparam(specification, samples=samples, suffix=suffix)
    ev = specparam['earningsvar']
    
    # Perform the grid search
    ## Without Years FE
    print("################# Without YFE ###################")
    # Compute the models average (the new variable to predicct)
    data['NN'] = predict_average_nn(data, specparam, fullsample=False, yfe=False, suffix=suffix, nmodels=100)
    specparam['earningsvar'] = 'NN'
    run_search_nn(data, specparam, recompute=True, yfe=False, pos_constraint=pos_constraint, suffix=suffix_cv)
    specparam['earningsvar'] = ev

    ## With Years FE
    if samples != 'years':
        print("################# With YFE ###################")
        data['NN'] = predict_average_nn(data, specparam, fullsample=False, yfe=True, suffix=suffix, nmodels=100)
        specparam['earningsvar'] = 'NN'
        run_search_nn(data, specparam, recompute=True, yfe=True, pos_constraint=pos_constraint, suffix=suffix_cv)
        specparam['earningsvar'] = ev

def run_crossvalidation_nn_final_C(specification, folder, pos_constraint=True, samples='gvkeys', min_year=None, suffix=''):
    set_folder(folder)
    set_ncpu(30)
    data = prepare_data("data/data.parquet", min_year=min_year)
    # Specify the suffix
    if min_year is not None:
        suffix += '_post'+str(min_year)
    if samples=='years':
        suffix += '_years'
    suffix_cv = suffix + '_final_C'
    # Get the specification parameters
    specparam = get_specparam(specification, samples=samples, suffix=suffix)
    ev = specparam['earningsvar']
    
    # Perform the grid search
    ## Without Years FE
    print("################# Without YFE ###################")
    # Compute the models average (the new variable to predicct)
    data['Cscore'] = c_score_nn(data, specparam, fullsample=True, yfe=False, nmodels=100, suffix=suffix)
    specparam['earningsvar'] = 'Cscore'
    run_search_nn(data, specparam, recompute=True, yfe=False, pos_constraint=pos_constraint, suffix=suffix_cv)
    specparam['earningsvar'] = ev

    ## With Years FE
    if samples != 'years':
        print("################# With YFE ###################")
        data['Cscore'] = c_score_nn(data, specparam, fullsample=True, yfe=True, nmodels=100, suffix=suffix)
        specparam['earningsvar'] = 'Cscore'
        run_search_nn(data, specparam, recompute=True, yfe=True, pos_constraint=pos_constraint, suffix=suffix_cv)
        specparam['earningsvar'] = ev
        
def run_train_nn(specification, folder, pos_constraint=True, samples='gvkeys', min_year=None, suffix='', nmodels=100):
    set_folder(folder)
    set_ncpu(30)
    data = prepare_data("data/data.parquet", min_year=min_year)
    # Specify the suffix
    #if min_year is not None:
    #    suffix += '_post'+str(min_year)
    #if samples=='years':
    #    suffix += '_years'
    # Get the specification parameters
    specparam = get_specparam(specification, samples=samples, suffix=suffix)
    # Get the CV hyperparameters
    opt_basic, opt_basic_yfe, opt_kw, opt_kw_yfe = hyperparameters()
    if specification=='basic':
        opt = opt_basic
        opt_yfe = opt_basic_yfe
    else:
        opt = opt_kw
        opt_yfe = opt_kw_yfe
    ## Without Years FE
    recompute=True
    print("################# Without YFE ###################")
    print('Training Sample')
    train_nn(data, specparam, cv=False, fullsample=False, params=opt, recompute=recompute, yfe=False, nmodels=nmodels, suffix=suffix)
    print('Full Sample')
    train_nn(data, specparam, cv=False, fullsample=True, params=opt, recompute=recompute, yfe=False, nmodels=nmodels, suffix=suffix)
    
    ## With Years FE
    if samples != 'years':
        print("################# With YFE ###################")
        print('Training Sample')
        train_nn(data, specparam, cv=False, fullsample=False, params=opt_yfe, recompute=recompute, yfe=True, nmodels=nmodels, suffix=suffix)
        print('Full Sample')
        train_nn(data, specparam, cv=False, fullsample=True, params=opt_yfe, recompute=recompute, yfe=True, nmodels=nmodels, suffix=suffix)
        
def run_laboratory(folder, N=100, start=0):
    """ Run the laboratory experiments.
    Assumes that the models (Fama-McBeth and Neural networks)
    have been trained already.
    """
    
    ###################################
    # Cross-Validation Neural Network #
    ###################################
    opt_basic, opt_basic_yfe, opt_kw, opt_kw_yfe = hyperparameters()

    #######################
    # Data and Parameters #
    #######################
    print("##### Data and Parameters #####")
    set_folder(folder)
    set_ncpu(30, tf='gpu')
    data = prepare_data("data/data.parquet")
    define_samples(data)
    specparam_basic = get_specparam('basic')
    specparam_kw = get_specparam('kw')

    #######################
    # Train Linear Models #
    #######################
    # Linear model
    print('##### Training - Linear #####')
    L_basic, _ = train_linear(data, specparam_basic, fullsample=True, recompute=True)
    L_kw, _ = train_linear(data, specparam_kw, fullsample=True, recompute=True)

    ##############################
    # Run the Laboratory N times #
    ##############################

    print('############################')
    print('######## Laboratory ########')
    print('############################')

    for iteration in range(start, start+N):
        print('########## Iteration '+str(iteration)+' ##########')
        laboratory(data, L_basic, L_kw, specparam_basic, specparam_kw, iteration)

def run_results(folder, n_iter_lab=100):
    ''' Run all the results.

    Note: Before this function is run, a few steps are required:
    
    1)  The cross-validation of the Neural Networks needs to be performed
        with run_crossvalidation_nn() and plot_cv() to find the hyperparameters.
        Then hard code the function hyperparameters(). Hard coding is done because
        the choice of the hyperparamters is hard to automate.
        
    2)  Train the models with run_train_nn()
    
    3)  Run the laboratory with run_laboratory() (very long ~ 10 days with GPU )

    '''

    ###################################
    # Cross-Validation Neural Network #
    ###################################
    opt_basic, opt_basic_yfe, opt_kw, opt_kw_yfe = hyperparameters()

    #######################
    # Data and Parameters #
    #######################
    print("##### Data and Parameters #####")
    set_folder(folder)
    set_ncpu(30, tf='gpu')
    data = prepare_data("data/data.parquet")
    define_samples(data)
    specparam_basic = get_specparam('basic')
    specparam_kw = get_specparam('kw')
    specparam_rep = get_specparam('replication')

    ##########################
    # Descriptive Statistics #
    ##########################
    print("##### Descriptive Statistics #####")
    obs_stats(data, specparam_basic, specparam_kw)
    desc_stats(data, specparam_basic)
    desc_corr(data, specparam_basic)
    desc_stats(cv_train_sample(data, specparam_basic), specparam_basic, suffix='_cvtrain')
    desc_stats(cv_valid_sample(data, specparam_basic), specparam_basic, suffix='_cvvalid')
    desc_stats(train_sample(data, specparam_basic), specparam_basic, suffix='_train')
    desc_stats(test_sample(data, specparam_basic), specparam_basic, suffix='_test')

    ###############################
    # Compute Fama-MacBeth Models #
    ###############################
    print("##### Compute FM models (Yearly) #####")
    # On Training Samples
    compute_fm_linear(data, specparam_basic)
    compute_fm_linear(data, specparam_kw)
    # On Full Samples
    compute_fm_linear(data, specparam_basic, fullsample=True)
    compute_fm_linear(data, specparam_kw, fullsample=True)

    #####################################
    # Replication Khan and Watts (2009) #
    #####################################
    print("##### Replication Khan and Watts (2009) #####")
    data_rep = prepare_data("data/data.parquet", replicatekw=True)
    # Descriptive Statistics
    desc_stats(data_rep, specparam_rep, suffix='_replication')
    desc_corr(data_rep, specparam_rep, suffix='_replication')
    # Compute the models
    compute_fm_linear(data_rep, specparam_rep)
    compute_fm_linear(data, specparam_rep, fullsample=True)
    # Table coefs KW
    table_coefs_kw(specparam_rep)

    ##############################
    # Analysis C score FM Yearly #
    ##############################
    print("##### Analysis C score FM Yearly #####")
    # Compute the C score
    data['c_Lfmy'] = c_score_fmy(data, specparam_rep, fullsample=True)
    data['g_Lfmy'] = g_score_fmy(data, specparam_rep, fullsample=True)
    data['gc_Lfmy'] = data.c_Lfmy + data.g_Lfmy
    # Regression of C score on lags
    regression_cscore_lags(data, 'c_Lfmy')
    # Plot coefs over time
    plot_time_coefs(specparam_rep)
    plot_time_coefs(specparam_rep, fullsample=True)
    plot_time_coefs(specparam_basic, fullsample=True)
    plot_time_coefs(specparam_kw, fullsample=True)
    # Densities g and g+c scores
    xlim=(-.4, .3)
    plot_score_density('g_Lfmy', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    xlim=(-.6, 1.)
    plot_score_density('gc_Lfmy', data, specparam_basic, sample='full', xlim=xlim, score='G+C Score')

    #########################################
    # Analysis of negative G and G+C scores #
    #########################################
    desc_stats(data[data.g_Lfmy < 0], specparam_basic, suffix='_neg_g_kw')
    desc_stats(data[data.gc_Lfmy < 0], specparam_basic, suffix='_neg_gc_kw')
    
    ###########################
    # Compute R2 and C scores #
    ###########################
    ## Linear model
    print('##### Training and R2 - Linear #####')
    L_basic_train, L_r2_basic_train = train_linear(data, specparam_basic, fullsample=False, recompute=True)
    L_basic, L_r2_basic= train_linear(data, specparam_basic, fullsample=True, recompute=True)
    L_kw_train, L_r2_kw_train = train_linear(data, specparam_kw, fullsample=False, recompute=True)
    L_kw, L_r2_kw= train_linear(data, specparam_kw, fullsample=True, recompute=True)
    # Compute
    print('# Compute C scores #')
    data['c_L_basic'] = c(L_basic, data, specparam_basic)
    data['c_L_kw'] = c(L_kw, data, specparam_kw)
    
    ## FM Yearly
    print('##### R2 - FM Yearly #####')
    Lfmy_r2_basic_train = r2_fmy(data, specparam_basic, fullsample=False)
    Lfmy_r2_basic = r2_fmy(data, specparam_basic, fullsample=True)
    Lfmy_r2_kw_train = r2_fmy(data, specparam_kw, fullsample=False)
    Lfmy_r2_kw = r2_fmy(data, specparam_kw, fullsample=True)
    # Compute
    print('# Compute C scores #')
    data['c_Lfmy_basic'] = c_score_fmy(data, specparam_basic, fullsample=True)
    data['c_Lfmy_kw'] = c_score_fmy(data, specparam_kw, fullsample=True)

    ## Neural Networks
    nmodels=100
    model0index=0
    print("##### NN - without YFE")
    NN_r2_basic_train = r2_nn(data, specparam_basic, fullsample=False, yfe=False, nmodels=nmodels, model0index=model0index)
    NN_r2_basic = r2_nn(data, specparam_basic, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    NN_r2_kw_train = r2_nn(data, specparam_kw, fullsample=False, yfe=False, nmodels=nmodels, model0index=model0index)
    NN_r2_kw = r2_nn(data, specparam_kw, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    print("##### NN - with YFE")
    NNy_r2_basic_train = r2_nn(data, specparam_basic, fullsample=False, yfe=True, nmodels=nmodels)
    NNy_r2_basic = r2_nn(data, specparam_basic, fullsample=True, yfe=True, nmodels=nmodels)
    NNy_r2_kw_train = r2_nn(data, specparam_kw, fullsample=False, yfe=True, nmodels=nmodels)
    NNy_r2_kw = r2_nn(data, specparam_kw, fullsample=True, yfe=True, nmodels=nmodels)
    # Compute
    print('# Compute C scores #')
    data['c_NN_basic'] = c_score_nn(data, specparam_basic, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    data['c_NN_kw'] = c_score_nn(data, specparam_kw, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    data['c_NNy_basic'] = c_score_nn(data, specparam_basic, fullsample=True, yfe=True, nmodels=nmodels, model0index=model0index)
    data['c_NNy_kw'] = c_score_nn(data, specparam_kw, fullsample=True, yfe=True, nmodels=nmodels, model0index=model0index)

    ## Compute G and G+C scores
    print('## Compute G and G+C scores ##')
    # G scores
    data['g_L_basic'] = g(L_basic, data, specparam_basic)
    data['g_L_kw'] = g(L_kw, data, specparam_kw)
    data['g_Lfmy_basic'] = g_score_fmy(data, specparam_basic, fullsample=True)
    data['g_Lfmy_kw'] = g_score_fmy(data, specparam_kw, fullsample=True)
    data['g_NN_basic'] = g_score_nn(data, specparam_basic, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    data['g_NN_kw'] = g_score_nn(data, specparam_kw, fullsample=True, yfe=False, nmodels=nmodels, model0index=model0index)
    data['g_NNy_basic'] = g_score_nn(data, specparam_basic, fullsample=True, yfe=True, nmodels=nmodels, model0index=model0index)
    data['g_NNy_kw'] = g_score_nn(data, specparam_kw, fullsample=True, yfe=True, nmodels=nmodels, model0index=model0index)
    # G+C Scores
    data['gc_L_basic'] = data.g_L_basic + data.c_L_basic
    data['gc_L_kw'] = data.g_L_kw + data.c_L_kw
    data['gc_Lfmy_basic'] = data.g_Lfmy_basic + data.c_Lfmy_basic
    data['gc_Lfmy_kw'] = data.g_Lfmy_kw + data.c_Lfmy_kw
    data['gc_NN_basic'] = data.g_NN_basic + data.c_NN_basic
    data['gc_NN_kw'] = data.g_NN_kw + data.c_NN_kw
    data['gc_NNy_basic'] = data.g_NNy_basic + data.c_NNy_basic
    data['gc_NNy_kw'] = data.g_NNy_kw + data.c_NNy_kw
    
    ###########################################
    # Train the final networks on the average #
    ###########################################
    print("##### Train final Networks #####")
    # Without years FE
    ## Basic
    p = specparam_basic
    df = full_sample(data, p)
    X, Y = get_sets(df, p, p['features_all'], yfe=False)
    opt = opt_basic.copy()
    opt['verbose'] = 2
    print('Basic: ', opt)
    X = df['c_NN_basic']
    nn_basic = CNetwork(**opt)
    nn_basic.fit(Y,X)
    ## KW
    p = specparam_kw
    df = full_sample(data, p)
    X, Y = get_sets(df, p, p['features_all'], yfe=False)
    opt = opt_kw.copy()
    opt['verbose'] = 2
    #opt['layer_size'] = 30
    opt['n_iterations'] = 30000
    print('KW: ', opt)
    X = df['c_NN_kw']
    nn_kw = CNetwork(**opt)
    nn_kw.fit(Y,X)

    # With years FE
    ## Basic
    p = specparam_basic
    df = full_sample(data, p)
    X, Y = get_sets(df, p, p['features_all'], yfe=True)
    opt = opt_basic_yfe.copy()
    opt['verbose'] = 2
    print('Basic YFE: ', opt)
    X = df['c_NNy_basic']
    nny_basic = CNetwork(**opt)
    nny_basic.fit(Y,X)
    ## KW
    p = specparam_kw
    df = full_sample(data, p)
    X, Y = get_sets(df, p, p['features_all'], yfe=True)
    opt = opt_kw_yfe.copy()
    opt['verbose'] = 2
    print('KW YFE: ', opt)
    X = df['c_NNy_kw']
    nny_kw = CNetwork(**opt)
    nny_kw.fit(Y,X)
    
    # Compute
    print("# Compute C scores with the final networks #")
    ## Basic No FE
    p = specparam_basic
    X, Y = get_sets(data, p, p['features_all'], yfe=False)
    data['c_NN_basic_final_C'] = nn_basic.predict(Y)
    ## KW No FE
    p = specparam_kw
    X, Y = get_sets(data, p, p['features_all'], yfe=False)
    data['c_NN_kw_final_C'] = nn_kw.predict(Y)
    ## Basic YFE
    p = specparam_basic
    X, Y = get_sets(data, p, p['features_all'], yfe=True)
    data['c_NNy_basic_final_C'] = nny_basic.predict(Y)
    ## KW YFE
    p = specparam_kw
    X, Y = get_sets(data, p, p['features_all'], yfe=True)
    data['c_NNy_kw_final_C'] = nny_kw.predict(Y)

    
    ########################
    # Save data and models #
    ########################
    print("##### Save Results Data #####")
    cols_cscore = ['c_L_basic', 'c_L_kw',
                   'c_Lfmy_basic', 'c_Lfmy_kw',
                   'c_NN_basic', 'c_NN_kw',
                   'c_NNy_basic', 'c_NNy_kw',
                   'c_NN_basic_final_C', 'c_NN_kw_final_C',
                   'c_NNy_basic_final_C', 'c_NNy_kw_final_C']
    data_c = data[['gvkey', 'fyear']+cols_cscore]
    data_c.columns = ['gvkey', 'fyear', 'L1', 'L2', 'L1y', 'L2y', 'MLC1_mean', 'MLC2_mean', 'MLC1y_mean', 'MLC2y_mean', 'MLC1', 'MLC2', 'MLC1y', 'MLC2y']
    data_c.to_parquet('data/cscores_'+folder+'.parquet', compression=None, index=False)
    data_c.to_csv('data/cscores_'+folder+'.csv', index=False)
    data_raw = pd.read_parquet("data/data.parquet")
    data_raw = data_raw[data_raw.fyear.notnull()]
    data_c = data_raw.merge(data_c, how='left', on=['gvkey', 'fyear'])
    data_c.to_parquet('data/data_'+folder+'.parquet', compression=None, index=False)
    data_c.to_csv('data/data_'+folder+'.csv', index=False)

    print("##### Save models #####")
    # For python
    nn_basic.model.save(folder+'/nn_basic.h5')
    nn_kw.model.save(folder+'/nn_kw.h5')
    nny_basic.model.save(folder+'/nny_basic.h5')
    nny_kw.model.save(folder+'/nny_kw.h5')
    # For Javascript
    tfjs.converters.save_keras_model(nn_basic.model, folder+'/nn_basic_js')
    tfjs.converters.save_keras_model(nn_kw.model, folder+'/nn_kw_js')
    tfjs.converters.save_keras_model(nny_basic.model, folder+'/nny_basic_js')
    tfjs.converters.save_keras_model(nny_kw.model, folder+'/nny_kw_js')
    
    
    ########################
    # Trainable parameters #
    ########################
    # To fit earnings
    nyears = 57
    ncoefs_full = {}
    ncoefs_full['L1'] = (3+1)*4
    ncoefs_full['L1y'] = (3+1)*4*nyears
    nn = load_full_model(specparam_basic, fullsample=False, yfe=False)
    ncoefs_full['MLC1'] = np.sum([K.count_params(w) for w in nn.model.trainable_weights])
    nn = load_full_model(specparam_basic, fullsample=False, yfe=True)
    ncoefs_full['MLC1y'] = np.sum([K.count_params(w) for w in nn.model.trainable_weights])
    ncoefs_full['L2'] = (8+1)*4
    ncoefs_full['L2y'] = (8+1)*4*nyears
    nn = load_full_model(specparam_kw, fullsample=False, yfe=False)
    ncoefs_full['MLC2'] = np.sum([K.count_params(w) for w in nn.model.trainable_weights])
    nn = load_full_model(specparam_kw, fullsample=False, yfe=True)
    ncoefs_full['MLC2y'] = np.sum([K.count_params(w) for w in nn.model.trainable_weights])
    # For C score
    ncoefs_c = {}
    ncoefs_c['L1'] = (3+1)
    ncoefs_c['L1y'] = (3+1)*nyears
    ncoefs_c['MLC1'] = np.sum([K.count_params(w) for w in nn_basic.model.trainable_weights])
    ncoefs_c['MLC1y'] = np.sum([K.count_params(w) for w in nny_basic.model.trainable_weights])
    ncoefs_c['L2'] = (8+1)
    ncoefs_c['L2y'] = (8+1)*nyears
    ncoefs_c['MLC2'] = np.sum([K.count_params(w) for w in nn_kw.model.trainable_weights])
    ncoefs_c['MLC2y'] = np.sum([K.count_params(w) for w in nny_kw.model.trainable_weights])
    # Write table
    ncoefs = pd.DataFrame([ncoefs_full, ncoefs_c])
    ncoefs.index = ['Full Model', 'C score']
    #filename='ncoefs'
    #ncoefs.to_latex(folder_tables+filename+'.tex', escape=True)
    
    ####################################
    # Fits, Correlations and Densities #
    ####################################
    print('##### Fits #####')
    models = {'train_models': [L_r2_basic_train, Lfmy_r2_basic_train, NN_r2_basic_train, NNy_r2_basic_train, L_r2_kw_train, Lfmy_r2_kw_train, NN_r2_kw_train, NNy_r2_kw_train],
              'full_models': [L_r2_basic, Lfmy_r2_basic, NN_r2_basic, NNy_r2_basic, L_r2_kw, Lfmy_r2_kw, NN_r2_kw, NNy_r2_kw]}
    labels = ['L1', 'L1y', 'MLC1', 'MLC1y', 'L2', 'L2y', 'MLC2', 'MLC2y']
    fit_table(models, labels, ncoefs, suffix='')

    print('##### Correlations #####')
    # Using averages
    variables = ['c_L_basic', 'c_Lfmy_basic', 'c_NN_basic', 'c_NNy_basic', 'c_L_kw', 'c_Lfmy_kw', 'c_NN_kw', 'c_NNy_kw']
    legends = ['L1', 'L1y', 'MLC1', 'MLC1y','L2', 'L2y', 'MLC2', 'MLC2y']
    corr_table(variables, specparam_kw, legends, data, suffix='_average')
    # Using final models
    variables = ['c_L_basic', 'c_Lfmy_basic', 'c_NN_basic_final_C', 'c_NNy_basic_final_C', 'c_L_kw', 'c_Lfmy_kw', 'c_NN_kw_final_C', 'c_NNy_kw_final_C']
    corr_table(variables, specparam_kw, legends, data, suffix='_final')

    print('##### C Scores Densities #####')
    # Linear
    xlim=(-0.15, 0.6)
    plot_score_density('c_L_basic', data, specparam_basic, sample='full', xlim=xlim)
    xlim=(-0.3, 0.4)
    plot_score_density('c_L_kw', data, specparam_basic, sample='full', xlim=xlim)
    # FM Yearly
    xlim=(-.5, 1)
    plot_score_density('c_Lfmy_basic', data, specparam_basic, sample='full', xlim=xlim)
    xlim=(-1,1)
    plot_score_density('c_Lfmy_kw', data, specparam_basic, sample='full', xlim=xlim)
    # Neural Network - Without YFE
    xlim=(-.1, .5)
    plot_score_density('c_NN_basic', data, specparam_basic, sample='full', xlim=xlim)
    plot_score_density('c_NN_basic_final_C', data, specparam_basic, sample='full', xlim=xlim)
    xlim=(-.05, .1)
    plot_score_density('c_NN_kw', data, specparam_kw, sample='full', xlim=xlim)
    plot_score_density('c_NN_kw_final_C', data, specparam_kw, sample='full', xlim=xlim)
    # Neural Network - Without YFE
    xlim=(-.1, .6)
    plot_score_density('c_NNy_basic', data, specparam_basic, sample='full', xlim=xlim)
    plot_score_density('c_NNy_basic_final_C', data, specparam_basic, sample='full', xlim=xlim)
    xlim=(-.1, .25)
    plot_score_density('c_NNy_kw', data, specparam_kw, sample='full', xlim=xlim)
    plot_score_density('c_NNy_kw_final_C', data, specparam_kw, sample='full', xlim=xlim)

    print('##### G and G+C Scores Densities #####')
    # FM Yearly
    xlim=(-.3, .3)
    plot_score_density('g_Lfmy_basic', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    print('G+C Scores')
    xlim=(-.6, 1.)
    plot_score_density('gc_Lfmy_basic', data, specparam_basic, sample='full', xlim=xlim, score='G+C Score')
    # Neural Network - Without YFE
    xlim=(0, .05)
    plot_score_density('g_NN_basic', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    xlim=(0, .5)
    plot_score_density('gc_NN_basic', data, specparam_basic, sample='full', xlim=xlim, score='G+C Score')
    xlim=(0, .1)
    plot_score_density('g_NN_kw', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    xlim=(0, .25)
    plot_score_density('gc_NN_kw', data, specparam_basic, sample='full', xlim=xlim, score='G+C Score')
    # Neural Network - With YFE
    xlim=(0, .04)
    plot_score_density('g_NNy_basic', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    xlim=(0, .7)
    plot_score_density('gc_NNy_basic', data, specparam_basic, sample='full', xlim=xlim, score='G+C Scores')
    xlim=(0, .1)
    plot_score_density('g_NNy_kw', data, specparam_basic, sample='full', xlim=xlim, score='G Score')
    xlim=(0, .35)
    plot_score_density('gc_NNy_kw', data, specparam_basic, sample='full', xlim=xlim, score='G+C Scores')
    
    
    ###############
    # Time Trends #
    ###############
    print('##### Time Trends #####')
    estimator = 'mean'
    ci = 'sd'
    # Basic specification
    plot_time(data, specparam_basic, 'c_L_basic', estimator=estimator, ci=ci)
    plot_time(data, specparam_basic, 'c_Lfmy_basic', estimator=estimator, ci=ci)
    plot_time(data, specparam_basic, 'c_NN_basic', estimator=estimator, ci=ci)
    plot_time(data, specparam_basic, 'c_NN_basic_final_C', estimator=estimator, ci=ci)
    plot_time(data, specparam_basic, 'c_NNy_basic', estimator=estimator, ci=ci)
    plot_time(data, specparam_basic, 'c_NNy_basic_final_C', estimator=estimator, ci=ci)
    # KW specification
    plot_time(data, specparam_kw, 'c_L_kw', estimator=estimator, ci=ci)
    plot_time(data, specparam_kw, 'c_Lfmy_kw', estimator=estimator, ci=ci)
    plot_time(data, specparam_kw, 'c_NN_kw', estimator=estimator, ci=ci)
    plot_time(data, specparam_kw, 'c_NN_kw_final_C', estimator=estimator, ci=ci)
    plot_time(data, specparam_kw, 'c_NNy_kw', estimator=estimator, ci=ci)
    plot_time(data, specparam_kw, 'c_NNy_kw_final_C', estimator=estimator, ci=ci)
    
    # Plots without the standard errors
    plot_time(data, specparam_basic, 'c_L_basic', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_basic, 'c_Lfmy_basic', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_basic, 'c_NN_basic', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_basic, 'c_NN_basic_final_C', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_basic, 'c_NNy_basic', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_basic, 'c_NNy_basic_final_C', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_L_kw', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_Lfmy_kw', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_NN_kw', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_NN_kw_final_C', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_NNy_kw', estimator=estimator, ci=None, suffix='_nosd')
    plot_time(data, specparam_kw, 'c_NNy_kw_final_C', estimator=estimator, ci=None, suffix='_nosd')
    
    # Focus on after 2000
    figure(figsize=(10,5))
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_L_basic', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_L_kw', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_Lfmy_basic', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_Lfmy_kw', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_NN_basic', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_NN_kw', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_NNy_basic', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_NNy_kw', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_NN_basic_final_C', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_NN_kw_final_C', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_basic, 'c_NNy_basic_final_C', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)
    plot_time(data[data.fyear>=2000], specparam_kw, 'c_NNy_kw_final_C', estimator=estimator, ci=None, suffix='_nosd_recent', add_events=True)

    
    ###############
    # Regressions #
    ###############

    ##### Regressions on Lags #####
    print('##### Regressions on Lags #####')
    # FM Yearly
    regression_cscore_lags(data, 'c_Lfmy_basic')
    regression_cscore_lags(data, 'c_Lfmy_kw')
    # Neural Network - Without YFE
    regression_cscore_lags(data, 'c_NN_basic')
    regression_cscore_lags(data, 'c_NN_kw')
    regression_cscore_lags(data, 'c_NN_basic_final_C')
    regression_cscore_lags(data, 'c_NN_kw_final_C')
    # Neural Network - With YFE
    regression_cscore_lags(data, 'c_NNy_basic')
    regression_cscore_lags(data, 'c_NNy_kw')
    regression_cscore_lags(data, 'c_NNy_basic_final_C')
    regression_cscore_lags(data, 'c_NNy_kw_final_C')


    ##### Regressions #####
    print('##### Regressions #####')
    drop_features = ['pin']
    variables1 = ['c_L_basic', 'c_Lfmy_basic', 'c_NN_basic', 'c_NNy_basic']
    variables1_final = ['c_L_basic', 'c_Lfmy_basic', 'c_NN_basic_final_C', 'c_NNy_basic_final_C']
    legend1 = ['L1', 'L1y', 'MLC1', 'MLC1y']
    variables2 = ['c_L_kw', 'c_Lfmy_kw', 'c_NN_kw', 'c_NNy_kw']
    variables2_final = ['c_L_kw', 'c_Lfmy_kw', 'c_NN_kw_final_C', 'c_NNy_kw_final_C']
    legend2 = ['L2', 'L2y', 'MLC2', 'MLC2y']
    # For 'basic' specification
    regression(variables1, legend1, data, specparam_basic, drop_features, suffix='')
    regression(variables1_final, legend1, data, specparam_basic, drop_features, suffix='_final')
    # For 'kw' specification
    regression(variables2, legend2, data, specparam_kw, drop_features, suffix='')
    regression(variables2_final, legend2, data, specparam_kw, drop_features, suffix='_final')

    #########
    # Plots #
    #########
    print('##### Plots #####')
    ## Number of points to consider for the plots
    nq = 20
    xlim=None
    ## Plot both 'basic' and 'kw' specifications
    # Linear
    plot_all_vars('c_L_basic', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_L_kw', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')
    # FM Yearly
    plot_all_vars('c_Lfmy_basic', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_Lfmy_kw', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')
    # Neural Network - Without YFE
    plot_all_vars('c_NN_basic', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NN_basic_final_C', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NN_kw', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NN_kw_final_C', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')
    # Neural Network - With YFE
    plot_all_vars('c_NNy_basic', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NNy_basic_final_C', data, specparam_basic, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NNy_kw', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')
    plot_all_vars('c_NNy_kw_final_C', data, specparam_kw, nq=nq, quantiles=True, xlim=xlim, sample='full')

    
    ##########################
    # Laboratory Experiments #
    ##########################

    print('##### Laboratory Experiments #####')
    ##### Robust laboratory with multiple draws #####
    print('### Robust laboratory ###')
    print('# Errors with no noise #')
    table_lab_robust(n_iter_lab, table='errors', noise_var=0, coef_cscore=0)
    print('# Coefs with no noise and coef C score=3 #')
    table_lab_robust(n_iter_lab, table='coefs', noise_var=0, coef_cscore=3)
    print('# Coefs with no noise and coef C score=0 #')
    table_lab_robust(n_iter_lab, table='coefs', noise_var=0, coef_cscore=0)
    print('# Fraction of significant coefs with noise (5) and coef C score=3 #')
    table_lab_sig(n_iter_lab, noise_var=5, coef_cscore=3)
    print('# Fraction of significant coefs with noise (5) and coef C score=0 #')
    table_lab_sig(n_iter_lab, noise_var=5, coef_cscore=0)

def run_robustness(folder, recompute=True):
    ''' Run robustness tests.

    One needs to first run the function run_results() with the same
    folder argument.

    '''

    ####################
    # Robustness Tests #
    ####################

    # NN without positivity restrictions


    # Train / Validation / Test sample along the years for L1, L2, MLC1, MLC2

    # Train on data after 2004

    None
