# AI-SOCO 2020 Final Project

This project encapsulates my attempt at the Authorship Identification of SOurce COde (AI-SOCO) 2020 task as part of the Forum for Information Retrieval Evaluation (FIRE) 2020 conference.

This project also makes up my final project for the university of birmingham MSc Computer Science course starting 2019.



### Challenge Description

Taken from [AI-SOCO 2020 website](https://sites.google.com/view/ai-soco-2020/task-description):

General authorship identification is essential to the detection of undesirable deception of others' content misuse or exposing the owners of some anonymous hurtful content. This is done by revealing the author of that content. Authorship Identification of SOurce COde (AI-SOCO) focuses on uncovering the author who wrote some piece of code. This facilitates solving issues related to cheating in academic, work and open source environments. Also, it can be helpful in detecting the authors of malware softwares over the world.

The detection of cheating in academic communities is significant to properly address the contribution of each researcher. Also, in work environments, credit sometimes goes to people that did not deserve it. Such issues of plagiarism could arise in open source projects that are available on public platforms. Similarly, this could be used in public or private online coding contests whether done in coding interviews or in official coding training contests to detect the cheating of applicants or contestants. A system like this could also play a big role in detecting the source of anonymous malicious softwares.

The dataset is composed of source codes collected from the open submissions in the Codeforces online judge. Codeforces is an online judge for hosting competitive programming contests such that each contest consists of multiple problems to be solved by the participants. A Codeforces participant can solve a problem by writing a solution for it using any of the available programming languages on the website, and then submitting the solution through the website. The solution's result can be correct (accepted) or incorrect (wrong answer, time limit exceeded, etc.).

In our dataset, we selected 1,000 users and collected 100 source codes from each one. So, the total number of source codes is 100,000. All collected source codes are correct, bug-free, compile-ready and written using the C++ programming language using different versions. For each user, all collected source codes are from unique problems.

Given the pre-defined set of source codes and their authors, the task is to build a system to determine which one of these authors wrote a given unseen before source code.
### Objectives

My objective in this project is to provide a solution that scores highly in the leaderboard by using and evaluating promising Natural Language Processing (NLP) methodologies and machine learning techniques, both shallow and deep in order to identify the best ensemble method to provide a competitive solution.

### Dependencies

* numpy~=1.18.5
* scikit-learn~=0.23.1
* tensorflow~=2.2.0
* transformers~=3.0.2
* torch~=1.5.1
* joblib~=0.15.1
* matplotlib~=3.2.1
* scipy~=1.4.1
* lime~=0.2.0.1
* tqdm~=4.46.1
* regex~=2020.6.8

### Baselines

Baselines were provided by AI-SOCO 2020 and have been modified to work with our project structure.

### Usage
 
Before usage, the data.zip file in the data_dir directory must be unzipped in its current location. This contains the raw, pre-processed and AST datasets required by the models. It also contains the pre-trained CodeBERT model used for fine-tuning.
 
Each model is contained within its own directory (except for initial_models.py which is in the root directory) based on vectorisation technique used. Most models utilise argparse. Arguments can be seen by running a script with the '-h' argument where relevant. Example args.txt files are also included which can be ultilised with the argument '@args.txt'.
Each directory also contains accessory python scripts used in the model, and some example files used for debugging and demonstration purposes.

Before ensembling can be run the following files must be run atleast once to completion to generate the prediction probability vector files used for ensembling:
* ast_model.py
* bert_model.py
* ngram_model.py
* stylometry_model.py
* bow_model.py

Probability files and trained models are not included in this repository due to size limitations but are available on request.

#### CodeBERT

The CodeBERT model is provided in three seperate scripts to allow use on Amazon Web Services during the fine-tuning process but avoid cost associated with running the tokeniser and precidtion processed, which can be carried out locally. bert_tokenize.py generates tokens for each source code to be used for fine-tuning the CodeBERT model and making predictions.
bert_model.py fine-tunes the CodeBERT model.
bert_predict.py uses the fine-tuned model to generate predictions on the tokenised source codes.
