import random

def train_test_split(X, y, test_size=0.2, seed=42, shuffle=True):
    assert X.shape[0] == y.shape[0], 'X and y must be the same size'
    random.seed(seed)
  
    terminator = int(y.shape[0] * (1 - test_size))
    
    if shuffle:
      all_date = list(zip(X, y))
      random.shuffle(all_date)
      X, y = zip(*all_date)  
    
    return X[:terminator], X[terminator:], y[:terminator], y[terminator:]
