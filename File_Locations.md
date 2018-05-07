## File Locations

### Important Files in `/ds/hohsiangwu/projects/semantic_search`

1. `{train, valid, test}.function`:   text file, each line is a tokenized function 

         - train + valid function rows: 4,978,625
         - test function rows: 50,290
         - total function rows: 5,028,915


2. `{train, valid, test}.docstring`:  text file, each line is a tokenized docstring

         - train + valid comment rows: 4,978,625
         - test comment rows: 50,290
         - total comment rows: 5,028,915


### Important Files in `/ds/hamel/CodeML/Get_Python_From_BigQuery/`

1.  `use_emb.npy`            :          Google Universal Sentence Encoder.  shape: (4978625, 512)

2. `concat_train_avg_emb.npy`:          language model average pooling.     shape: (4978625, 400)

3. `concat_train_max_emb.npy`:          language model max pooling.         shape: (4978625, 400)

4. `concat_train_last_emb.npy`:         language model last hidden state.   shape: (4978625, 400)

5. `combined_fastailm_emb`    :         Horizontal concat of [2, 3, 4].     shape: (4978625, 1200)

6.  `codeSearch_Model_frozen.hdf5`:     My best keras model model that maps code to vector with val_loss = -0.8061 cosine proximity loss.   

7. `lm_fastai_codecomment_model.pytorch`:  This is the language model trained with fastai, you will have to have the latest version of the fast.ai library

8.  `lm_fastai_codecomment_model_state_dict`: This is the state dict of the language model, a more lightweight way of re-instantiating the model's parameters.

9.  `fitlam_index.nmslib`:  This is the nmslib index that has all the code after it has been vectorized by the language model.  

### Important Notebooks

1.  `hamel/CodeML/Get_Python_From_BigQuery/Parse_Ho_Hsian_Files.ipynb`  this notebook where I do things that are CPU intense:
 - preprocess the {training, validation, lineage} files. 
 - run all the comments through the vectorizers (did a version for both Google and My own languagel model).
 - Loaded all the vectors into an NMS Lib
 - **This is the notebook where the actual demo lives**

2. `hamel/fastai/courses/dl1/lang-model-code-comments.ipynb`:  this notebook is where i trained the fastai language model, and also where I then used the trained model to vectorize all the comments.

3. `/hamel/CodeML/projects/function_summarizer/keras-code-search.ipynb` this is the notebook where I
 - train a function summarizer 
 - fine tune the function summarizer to predict embedding instead of docstring
 - make predictions for all the training data to vectorize all the code (happens on GPU).  This is the code that is loaded into the index where you want to do nearest neighbor search from.

4. `hamel/CodeML/Get_Python_From_BigQuery/Get%20Data%20For%20Python_Code_Search.ipynb` - this notebook I was using to build my own training set, but I abandoned this in favor of Ho-Hsiang's Training dataset which he gave me.
