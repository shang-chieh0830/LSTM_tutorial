
def split_time_series(df, variable, trn = 0.8, val = 0, tst = 0.2):
    
    '''
    Split time series into train set, validation set, and test set
    Parameters
    ---------
    df: data frame
    variable:  must come with " "
    trn: train set ratio
    val: validation set ratio
    tst: test ratio
    
    trn + val + tst must equal 1
    '''

    if trn + val + tst != 1:
        raise ValueError("Incorrect splitting")
    timesteps = df.index.to_numpy()
    variable = df[variable].to_numpy()
    
    num_train_samples = int(trn * len(df)) 
    num_val_samples = int(val * len(df)) 
    num_test_samples = len(df) - num_train_samples - num_val_samples 
    print("num_train_samples:", num_train_samples)
    print("num_val_samples:", num_val_samples)
    print("num_test_samples:", num_test_samples)
    
    # Create train data splits
    X_train, y_train = timesteps[:num_train_samples], variable[:num_train_samples]

    # Create validation data splits
    X_val, y_val = timesteps[num_train_samples:num_train_samples+num_val_samples], variable[num_train_samples:num_train_samples+num_val_samples]

    # Create test data splits
    X_test, y_test = timesteps[num_train_samples+num_val_samples:], variable[num_train_samples+num_val_samples:]

    print(len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test



########################################################################################

# Create a function to plot time series data
def plot_time_series(timesteps, values, x_axis = "Time" ,y_axis = "None", format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  x_axis,y_axis: specifies the x-axis label and y-axis label
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  from matplotlib import pyplot as plt
   
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel(x_axis)
  plt.ylabel(y_axis)
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)



########################################################################################



# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:, :-horizon], x[:, -horizon:]


# Create function to view NumPy arrays as windows 
def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels


# Make the train/test splits
def make_trn_val_tst_splits(windows, labels, trn = 0.8, val = 0, tst=0.2):
  """
  Splits matching pairs of windows and labels into train, validation, and test splits.
  """
  num_trn_samples = int(len(windows) * trn) 
  num_val_samples = int(len(windows) * val)
  num_tst_samples = int(len(windows) * tst)

  train_windows = windows[:num_trn_samples]
  train_labels = labels[:num_trn_samples]

  validation_windows = windows[num_trn_samples:num_trn_samples+num_val_samples]
  validation_labels = labels[num_trn_samples:num_trn_samples+num_val_samples]

  test_windows = windows[num_trn_samples+num_val_samples:]
  test_labels = labels[num_trn_samples+num_val_samples:]

  return train_windows, validation_windows , test_windows, train_labels, validation_labels , test_labels
