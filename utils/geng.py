import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def standardize_data(X):
    """
    Standardize the data using StandardScaler
    """
    
    # X is a 3D array with shape (n_cases, n_timepoints, n_channels)
    # For demonstration, let's create a sample array with this shape
    # X = np.random.rand(360, 45, 2)

    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Reshape the data for StandardScaler
    X_reshaped = X.reshape(-1, X.shape[-1])  # Reshaping to (360*45, 2)

    # Fit the scaler on the data and transform
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to original shape (360, 45, 2)
    X_scaled = X_scaled.reshape(X.shape)

    return X_scaled


def standardize_data_individual(X_sample):

    # Initialize an empty array to store the scaled data
    X_scaled_individual = np.zeros_like(X_sample)

    # Iterate over each case/sample in the dataset
    for i in range(X_sample.shape[0]):
        # Reshape the data for StandardScaler (flattening timepoints and channels)
        sample_reshaped = X_sample[i].reshape(-1, X_sample.shape[-1])

        # Initialize a new scaler for each sample
        scaler_individual = StandardScaler()

        # Fit the scaler on the data and transform
        sample_scaled = scaler_individual.fit_transform(sample_reshaped)

        # Reshape back to the original shape and store in the scaled array
        X_scaled_individual[i] = sample_scaled.reshape(X_sample[i].shape)

    return X_scaled_individual  # Verifying the shape after scaling


def fit_split(classifier, x,y,epoch=300):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    input_shape = x.shape[1:]
    nb_classes = len(np.unique(y, axis=0))
    
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 重置模型（非常重要）
        self.model = self.build_model(input_shape, nb_classes)
        self.model.load_weights(self.output_directory+'model_init.hdf5')

        # 使用选定的数据分割进行训练
        self.fit(x_train, y_train, x_val, y_val, ...)

    