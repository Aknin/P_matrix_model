Gaussian-ISP-model
===

Code estimating the parameters of a reverberation model based on the image source principle, as described in [Achille Aknin and Roland Badeau, _"Evaluation of a stochastic reverberation model based of the image source principle"_, DAFX 2020].

Requirements
-------------
  - **Python 2.7 or python 3**
  - **numpy** version >= 1.16
  - **scipy** version >= 1.2.1

How to use
-------------
  See notebook **example.ipynb** for an example run of the EM algorithm or **help(EM)** for documentation of the module.

Example datasets
-------------
  In **Gaussian-ISP-model/examples**, example RIRs can be found in pickle format. Note that these were exported with python 2.7's **cPickle** module, so please use the option **encoding="latin1"** of the **\_pickle.load** function if you are using python 3, as demonstrated in **example.ipynb**.
  
  Loading the pickle will return a dictionnary with the following keys and values:
  - h: the RIR, including the measure error,
  - fs: the sampling frequency,
  - T60: the reverberation time, computed using Eyring's formula,
  - a: the corresponding true exponential decrease parameter,
  - V: the room's volume,
  - rif_g: the temporal impulse response of the filter g,
  - sigma2: the white noise variance
  - w: the random white noise that was added to the RIR, allowing us to get the latent parameter b = h-w,
  - SNR: the corresponding Signal to Noise Ratio.
  
  Two datasets are available:
  - absorption: a dataset with 6 different wall absorption levels ranging from 0.3 to 0.8, each with 10 different random white noise seed,
  - SNR: a dataset with 9 different white noise variances, of which the decimal logarithms are uniformly spaced from -6 to -2, each with 10 different random white noise seed (sigma_0 corresponds to 10^(-6) and sigma_9 to 10^(-2)).
  
  

Contact
-------------
  Contact Achille Aknin (achille [dot] aknin [at] telecom-paris [dot] fr) if you have any question or issue.
