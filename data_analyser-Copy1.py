import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

class DataAnalyser():
    
    def __init__(self, 
                 original_data, 
                 synthetic_data):
        self.original_data = original_data
        self.synthetic_data = synthetic_data

    def get_top_transactors(self, typ = 'original', 
                        key_column = '50f_payor_add_ln_2',
                        value_column = 'usd_amt',
                        n = 100):
        dataframe = self.original if typ == 'original' else self.synthetic_data
        return OrderedDict(sorted(
                    (dataframe.groupby([key_column])[value_column].sum() / dataframe.groupby([key_column])
                    [value_column].count()).to_dict().items(), 
                    key=lambda x: x[1], reverse=True)[:n])


    def plot_transaction_distribution(self, 
                                      bins = 10, top_n = 500,
                                      value_col = 'usd_amt', 
                                      category_col = '33b_cur'):
        
        orig_data = self.original_data.sort_values(by=[value_col, category_col], ascending = False)
        synthetic_data = self.synthetic_data.sort_values(by=[value_col, category_col], ascending = False)
        
        bucket = 1/bins
        for quant in range(bins):
            high = np.round(1 - quant * bucket, decimals=2)
            low = (high - bucket)            
            orig = orig_data[(orig_data[value_col] > np.quantile(orig_data[value_col], low)) & 
               (orig_data[value_col] <= np.quantile(orig_data[value_col], high ))] \
                .sort_values(by=['usd_amt', '33b_cur'])[:top_n]
            
            synthetic = synthetic_data[(synthetic_data[value_col] > np.quantile(synthetic_data[value_col], low)) & 
              (synthetic_data[value_col] <= np.quantile(synthetic_data[value_col], high))][:top_n] \
                .sort_values(by=['usd_amt', '33b_cur'])[:top_n]
        
            sns.set(style='whitegrid')
            fig, axes = plt.subplots(1, 2, figsize=(17, 3))
            # fig.suptitle('Box Plots for transactions')
            
            ax = sns.boxplot(data=orig, x='33b_cur', y='usd_amt', ax = axes[0])
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
            ax.set_xlabel('Original Data')
            
            ax = sns.boxplot(data=synthetic, x='33b_cur', y='usd_amt', ax = axes[1])
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
            ax.set_xlabel("Synthetic Data")
            plt.tight_layout()
            plt.title(f'Currency distribution in top {top_n} transactions')
            plt.show()

            
    def plot_analysis(self, 
                      lower = 0.7, 
                      value_col = 'usd_amt',
                      category_col = '33b_cur',
                      transactor = 'src',
                      top_n = 1000):

        orig_data = self.original_data.sort_values(by=[value_col, category_col], ascending = False)
        synthetic_data = self.synthetic_data.sort_values(by=[value_col, category_col], ascending = False)
            
        orig = orig_data[(orig_data[value_col] > np.quantile(orig_data[value_col], lower)) & 
                         (orig_data[value_col] <= np.quantile(orig_data[value_col], 1))].sort_values(by=value_col)[:top_n]
        # lat = orig[orig['50k_payor_add_ln_2'] != '']['50k_payor_add_lat'].values.tolist()
        # lon = orig[orig['50k_payor_add_ln_2'] != '']['50k_payor_add_lon'].values.tolist()
        # amount = orig[orig['50k_payor_add_ln_2'] != 'K']['usd_amt'].values.tolist()
            
        lat = orig[transactor + '_lat'].values.tolist()
        lon = orig[transactor + '_lon'].values.tolist()
        amount = orig[value_col].values.tolist()
            
        self.plot_world_map(lat, lon, amount)
                
        synthetic = synthetic_data[(synthetic_data[value_col] > np.quantile(synthetic_data[value_col], lower)) & 
                    (synthetic_data[value_col] <= np.quantile(synthetic_data[value_col], 1))].sort_values(by=value_col)[:top_n]

        # lat = synthetic[synthetic['src_xfrr_type'] == 'K']['50k_payor_add_lat'].values.tolist()
        # lon = synthetic[synthetic['src_xfrr_type'] == 'K']['50k_payor_add_lon'].values.tolist()
        # amount = synthetic[synthetic['src_xfrr_type'] == 'K']['usd_amt'].values.tolist()
        lat = synthetic[transactor + '_lat'].values.tolist()
        lon = synthetic[transactor + '_lon'].values.tolist()
        amount = synthetic[value_col].values.tolist()
        self.plot_world_map(lat, lon, amount)
        
                
    def plot_world_map(self, lat, lon, amount):
        # 1. Draw the map background
        fig = plt.figure(figsize=(20, 10))
        m = Basemap(projection='cyl', resolution='c',
                llcrnrlat = 15, urcrnrlat = 60,
                llcrnrlon = -130, urcrnrlon = 30)

        m.shadedrelief()
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')

        # 2. scatter city data, with color reflecting population
        # and size reflecting area
        m.scatter(lon, lat, latlon=True, 
                    c=np.log10(amount), 
        s = (np.array(amount)/10000).tolist(), cmap='Reds', alpha=1)
        plt.title('Top transactors hotspots')
        plt.show()
        
    
    def attr_plot(self, data, samples, discrete = None, numeric = None):
        sns.set(style='whitegrid')
        if discrete is None:
            for col in numeric:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                sns.distplot(data[col], bins=100, hist_kws={'alpha': 0.7})
                plt.subplot(122)
                sns.distplot(samples[col], bins=100, hist_kws={'alpha': 0.7})
        else:
            for col in discrete:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                sns.countplot(x=col, data=data[discrete])
                plt.xticks(rotation=45)
                plt.subplot(122)
                sns.countplot(x=col, data=samples[discrete])
                plt.xticks(rotation=45)
    
    def corrplot(self, data, samples):
        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        sns.heatmap(data.loc[:, :"target_lat"].corr(), annot=True, cmap="Blues", vmin = -0.5, vmax = 0.5)

        plt.subplot(122)
        sns.heatmap(samples.loc[:, :"target_lat"].corr(), annot=True, cmap="Blues", vmin = -0.5, vmax = 0.5)