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
                                      bins = 10, top_n = 1000,
                                      value_col = 'usd_amt', 
                                      category_col = '33b_cur'):
        
        orig_data = self.original_data.sort_values(by=[value_col, category_col], ascending = False)
        synthetic_data = self.synthetic_data.sort_values(by=[value_col, category_col], ascending = False)
        
        bucket = 1/bins
        for quant in range(bins):
            
            # quant = np.around(quant/bins, decimals=1)
            # np.around([0.37, 1.64], decimals=1)
            high = np.round(1 - quant * bucket, decimals=2)
            low = (high - bucket)
            print(f"low : {low}; high: {high}")
            
            orig = orig_data[(orig_data[value_col] > np.quantile(orig_data[value_col], low)) & 
               (orig_data[value_col] <= np.quantile(orig_data[value_col], high ))][:top_n]
            #.sort_values(by=['usd_amt', '33b_cur'])[:top_n]
            
            synthetic = synthetic_data[(synthetic_data[value_col] > np.quantile(synthetic_data[value_col], low)) & 
              (synthetic_data[value_col] <= np.quantile(synthetic_data[value_col], high))][:top_n]
            
            
            # synthetic = synthetic_data[(synthetic_data[value_col] > np.quantile(synthetic_data[value_col], 0.1 * quant)) & 
            #   (synthetic_data[value_col] <= np.quantile(synthetic_data[value_col], 0.1 * (quant + 1)))][:top_n]
            #.sort_values(by=['usd_amt', '33b_cur'])[:top_n]
            sns.set(style='whitegrid')
            fig, axes = plt.subplots(1, 2, figsize=(17, 3))
            fig.suptitle('Box Plots for transactions')
            sns.boxplot(ax=axes[0], data=orig, x=category_col, y=value_col)
            sns.boxplot(ax=axes[1], data=synthetic, x=category_col, y=value_col)
            plt.tight_layout()
            plt.show()
        
            #         sns.set(style="whitegrid")
            #         fig, axes = plt.subplots(bins, 2, figsize=(20, 20))
            #         fig.suptitle('Box Plots for transactions')

            #         for row in range(bins):
            #             df = orig_data
            #             orig = df[(df[value_col] > np.quantile(df[value_col], 0.1 * row)) & 
            #                    (df[value_col] <= np.quantile(df[value_col], 0.1 * (row + 1)))][:100]
            #             #.sort_values(by=[category_col, value_col], ascending = False)

            #             df = synthetic_data
            #             synthetic = df[(df[value_col] > np.quantile(df[value_col], 0.1 * row)) & 
            #                         (df[value_col] <= np.quantile(df[value_col], 0.1 * (row + 1)))][:100]
            #             #.sort_values(by=[category_col, value_col], ascending = False)

            #             sns.boxplot(ax=axes[row, 0], data=orig, x = category_col, y = value_col)
            #             sns.boxplot(ax=axes[row, 1], data=synthetic, x = category_col, y = value_col)
            #             plt.tight_layout()
            #             plt.show()
            
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
                # m = Basemap(projection='lcc', resolution='c', 
                #             lat_0=31, lon_0=-130,
                #             width=6E6, height=3.2E6)

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
                plt.show()