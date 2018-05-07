
import os
from pathlib import Path
from more_itertools import chunked
from fastai.text import *
import torch

DATA_PATH = Path('/ds/hohsiangwu/projects/semantic_search/')
MODEL_PATH = Path('/ds/hamel/fastai/courses/dl1/code_comment_lm')
os.environ["CUDA_VISIBLE_DEVICES"]="2"
BOS = '_xbos_ '

def list_flatten(l):
    "List[List] --> List"
    return [item for sublist in l for item in sublist]


def read_files(PATH=DATA_PATH, max_vocab = 30000, min_freq = 25):
    "Read in files for language model."
    
    with open(PATH/'train.docstring', 'r') as f:
        t_comment = f.readlines()

    with open(PATH/'valid.docstring', 'r') as f:
        v_comment = f.readlines()

    with open(PATH/'test.docstring', 'r') as f:
        holdout_comment = f.readlines()

    tok_trn = list_flatten([(BOS + x).split() for x in t_comment])
    tok_val = list_flatten([(BOS + x).split() for x in v_comment])

    # index to string
    freq = Counter(tok_trn) # on training set, then applied to val
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    
    trn_lm = np.array([stoi[s] for s in tok_trn])
    val_lm = np.array([stoi[s] for s in tok_val])

    # string to index
    print(f'Size of vocabulary: {len(itos):,}')
    print(f'Size of flattened training set: {len(trn_lm):,}')
    print(f'Size of flattened validation set: {len(val_lm):,}')
    
    
    #tokenize original sequences
    tok_trn_list = [(BOS + x).split() for x in t_comment]
    tok_val_list = [(BOS + x).split() for x in v_comment]
    tok_hld_list = [(BOS + x).split() for x in holdout_comment]
    idx_trn_list = [[stoi[a] for a in x] for x in tok_trn_list]
    idx_val_list = [[stoi[a] for a in x] for x in tok_val_list]
    idx_hld_list = [[stoi[a] for a in x] for x in tok_hld_list]
    return itos, stoi, trn_lm, val_lm, idx_val_list, idx_hld_list


def train_model(PATH=MODEL_PATH, DATA_PATH=DATA_PATH):
    PATH.mkdir(exist_ok=True)
    itos, stoi, trn_lm, val_lm, _, _ = read_files(DATA_PATH)
    PATH.mkdir(exist_ok=True)
    em_sz,nh,nl = 400,400,3
    wd=1e-7
    bptt=20
    bs=32
    vs=len(itos)
    torch.cuda.empty_cache()
    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs, bptt)
    md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7
    learner= md.get_model(opt_fn, em_sz, nh, nl, 
                            dropouti=drops[0], 
                            dropout=drops[1], 
                            wdrop=drops[2], 
                            dropoute=drops[3], 
                            dropouth=drops[4])
    lrs = 1e-3 / 2
    
    learner.fit(lrs, 1, wds=wd, cycle_len=3, use_clr=(32,10))
    lm_fastai_codecomment_model = learner.model.cpu()
    return lm_fastai_codecomment_model.eval()    


def build_google_emb_index(PATH=DATA_PATH):
    """
    Vectorize training set comments with Google's Universal Sentence Encoder.
    
    Note - Cached version saved at: PATH/'use_emb.npy
    """
    
    with open(PATH/'train.docstring', 'r') as f:
        t_comment = f.readlines()

    with open(PATH/'valid.docstring', 'r') as f:
        v_comment = f.readlines()
    
    #Tensorflow Hub
    import tensorflow_hub as hub
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    with tf.Session() as session:
        def get_embeddings(text_blob_list):
                emb = session.run(embed(text_blob_list))
                return emb

    train_chunked = list(chunked(train_comment, 500000))
    i = 1
    use_emb_list = []

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        for x in train_chunked:
            print(i)
            i+=1
            #75th percentile of comment length is 89 characters, so I made limit 200.
            use_emb_list.append(get_embeddings([s[:200] for s in x]))
    
    use_emb = np.concatenate(use_emb_list)
    
    return use_emb