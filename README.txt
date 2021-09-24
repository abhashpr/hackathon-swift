The submission has four notebooks and supporting files:
    
    NOTEBOOKS
    ---------
    => Data_Preparation_v3.ipynb [Should be run first]
       This notebook splits different fields in source data into features that can be fed into the model
      
    => Data_Generation_gmm-v1.ipynb [Should be run second]
       This notebook trains the model and generates the final data in MT103 format
    
    ___________________________________________________________________________________________________
    ???????????????????????????????????????????????????????????????????????????????????????????????????
    Folloing notebooks could not be run due to time and environment constraints
    => Data_Generation_gan-v1.ipynb [This need not run]
       This notebook has an alternative GAN model based taken from CTGAN framework. We change the model
       to include 1-Dimensional CNN layers as first set of layers in Discriminator. We got good results 
       from this but is not complete and cannot be run in this state. The GAN model is based on 
       pac GAN and conditional GAN
    => Rule_gen.ipynb
       This notebook uses KModes clustering to cluster data based on discrete features and provide 
       high probabaility ranges for the numerical attributes. These rules are autamatically created 
       to be applied to the synthetic data to filter sampled records that do not mimic the actual 
       records.
       
    FILES
    -----
    => base.py
       Creates the base synthesizer for the neural network model
      
    => sampler.py
       Data sampler to create condition vector and generate samples
    
    => geo_handler.py
       Code to handle location to geocodes conversion. Calculate nearest location for 
       given geo coordinates
    
    => nn.py
       GAN model with all layers and methods to fit data and generate samples
    
    => transformer.py
       Used by GAN model as well as the GMM model to transform data to be fed into 
       the model.
       
    => data_analyser.py
       Set of visualisation for the actual data and synthetic data
    
    => reverse_xfm.py
       Set of code to transform data from one format to another for anaysis, modeling,
       final MT103 format.
       

-------
Step 1:
-------
    ==> Ensure the setup.sh script is run through the first cell in notebook
    ==> Please run these from shell if it does not run in the notebook:
        COMMAND: `yes | . ./setup.sh` 
        [or ensure that these packages are run into the session where the notebook runs]


-------
Step 2:
-------
   ==> Run notebook: Data_Preparation_v3.ipynb


-------
Step 3:
-------
   ==> Run notebook: Data_Generation_gmm-v1.ipynb
   
   [
       The data has been split up into two groups:
       Transferrer type = 'A'
       Transferrer type = 'F' or 'K'
   ]
   There are two Gaussian Mixture models to simulate the latent 
   space for correlations between attributes
   
   Pre-requisites and Changes:
   ===========================
       The following cells should be edited to run the simulation on whole set of data or smaller set.
       Cell 4:
           run_sample = True
           If set `True` the simulation will take only 1000 records to feed to the simulation model
       
       Cell 9:
           reload_data = True
               If set to true then it will retrigger the data transformation pipeline. 
               Else loads the existing pickled transformer model
               Set to true for input data change or first time run.
           retrain = True
               If set to true then it will retrigger the `fit` method for model. 
               Else loads the existing pickled training model
               Set to true for input data change or first time run.
           n_components = 15
               Change number of clusters model has to work with.
               
       Cell 13:
           num_samples = 20000
           Change the above variable to select more number of samples from the model
       
       Cell 23:
           `datan.plot_transaction_distribution(bins = 5, value_col = 'usd_amt', top_n = 100)`
           Shows distribution of transaction currencies in 5 equal percentiles based on amount
       Cell 24:
           `datan.plot_analysis(lower = lower_bound_percentile, transactor = 'target', value_col = 'usd_amt')`
           Used to plot the top 100 transactions by value to show on the world map.
       
       