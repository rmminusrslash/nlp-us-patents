# US Patents - 99% of the result with a mininum solution

The goal of this repo is to take the kaggle challenge [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) and the great [solution repository](https://github.com/Gladiator07/U.S.-Patent-Phrase-to-Phrase-Matching-Kaggle/tree/main) and strip the full solution down to a mininum one.

Useful Resources:
- [Kaggle Challenge](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching)
- [Kaggle Result Discussion](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332355)
- [Solution repository](https://github.com/Gladiator07/U.S.-Patent-Phrase-to-Phrase-Matching-Kaggle/tree/main)
- [Weights and Biases Experiment Dashboard](https://wandb.ai/gladiator/USPPPM-Kaggle)

## Type of problem
The goal is to score how relevant a search result is to a user query. A score of 0 means not related and 1 means "really close match".

Example:<br>
TV set;television set;1.0</br>
pushing pin,pushing syndrome;0.0


## Real-life ML vs Kaggle Solution
Questions I have:
1. Can I create a mininum version of a single model code that still produces a good result? Kaggle winners often have to work hard to achieve the last tiny uplifts. In real-world scenarios the marginal gains don't justify the addtional time spent.
2. What is the performance difference between a good single model and the ensemble solution?
3. What is the performance difference between different models?
4. Error Analysis- Is it worth doing in kaggle competitions?
5. How much effort is needed to set up a model training on a gpu-for-rent service?

Answers
1. See code
2. Based on the [Kaggle Result Discussion](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332355) the best single model achieves a pearson score of 0.833 while the ensemble achieves 0.85. In practice this means ensembles are only worth it for use cases with a huge leverage where other avenues like more or improved data are already done
3. Based on the [Weights and Biases Experiment Dashboard](https://wandb.ai/gladiator/USPPPM-Kaggle) Bert-Large has a pearson score of ~0.78 vs microsoft-deberta-v3-large ~0.83. The second model is 3x as large. However a patent bert (based on bert-large) performs at ~0.82. So specialized models can be very useful.
4. TODO
5. See "How to run" Section

## Insights
- hydra is a nice library for config managment when running many experiments. It also ensures that each run is logged into a seperate output directory
- a "debug" option that samples a tiny amount of training data is as always useful to confirm the end-2-end flow works




## How to run 

### Locally
1. Download data:
  ````
  $ export KAGGLE_API_KEY=yourkey
 $ export KAGGLE_USER=youruser
 $ bash use_patents/download_data.sh  
 ````
2. Install the env: 
  ````
  $ poetry install --no-root
 ````
If you have a gpu, you can change pyproject.toml to the gpu based version of pytorch for a faster run.

## Use the GPU-for-rent service jarvislabs.ai
If one has basic developer knowledge, these services are much nicer and faster to use than kaggle, colab or big players like aws.

#### Start a GPU instance
You need a credit card and 10$ min spend to sign up on https://cloud.jarvislabs.ai/. Upload your ssh key under Profile-> API keys. To start an instance, select instance type xxxx, drop down the section "advanced settings" and upload jarvislabs_setup.sh as a startup script. This means it will be executed when the instance is started.

#### Connect to it
Copy the ssh command by clicking the ssh icon next to the started instance in jarvis dashboard. Paste it in your terminal and log into the instance.
Go into the downloaded repository
`$ python -m us_patents.train debug=False data.input_dir=$PWD/data`

#### Pause instance
After training you can pause the instance to pay only a basic fee for a paused instance. Append this to your training script
`python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause[]" `

TODOS
- traing done, but not the same performance as in weights and biases -> 74.8 todo: check worst bert performance in w+b

- next: improve performance
- in jarvislab script, remove the branch and use master
