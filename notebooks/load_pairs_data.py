from pathlib import Path
PATH = Path('/ds/hohsiangwu/projects/semantic_search')

def read_data(PATH=PATH):
    """
    Reads data from PATH and produces the following variables:
    
    train_code, train_comment, holdout_code, holdout_comment, train_lineage
    
    """
    with open(PATH/'train.function', 'r') as f:
        t_code = f.readlines()

    with open(PATH/'train.function', 'r') as f:
        t_code = f.readlines()

    with open(PATH/'valid.function', 'r') as f:
        v_code = f.readlines()

    with open(PATH/'test.function', 'r') as f:
        holdout_code = f.readlines()
        
    train_code = t_code + v_code
    print(f'train_code rows: {len(train_code):,}')
    print(f'holdout_code rows: {len(holdout_code):,}')
    print(f'total code rows: {len(holdout_code + train_code):,}')
        
    
    with open(PATH/'train.docstring', 'r') as f:
        t_comment = f.readlines()

    with open(PATH/'valid.docstring', 'r') as f:
        v_comment = f.readlines()

    with open(PATH/'test.docstring', 'r') as f:
        holdout_comment = f.readlines()
       
    train_comment = t_comment + v_comment
    print(f'\ntrain_comment rows: {len(train_comment):,}')
    print(f'holdout_comment rows: {len(holdout_comment):,}')
    print(f'total comment rows: {len(holdout_comment + train_comment):,}')
    
    with open(PATH/'train.lineage', 'r') as f:
        t_lineage = f.readlines()
    
    with open(PATH/'valid.lineage', 'r') as f:
        v_lineage = f.readlines()
    
    with open(PATH/'test.lineage', 'r') as f:
        holdout_lineage = f.readlines()
    
    train_lineage = t_lineage + v_lineage
    print(f'\ntrain_lineage rows: {len(train_lineage):,}')
    print(f'holdout_comment rows: {len(holdout_lineage):,}')
    print(f'total lineage rows: {len(holdout_lineage + train_lineage):,}')
    
    return train_code, train_comment, holdout_code, holdout_comment, train_lineage, holdout_lineage