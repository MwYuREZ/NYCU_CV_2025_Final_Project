import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import warnings

warnings.filterwarnings('ignore')

# Set constants
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = [512, 512]
RESIZED_IMAGE = [224, 224]
NUM_CLASSES = 5
RANDOM_SEED = 42


def load_data(path):
    """Load data from the path and prepare training metadata."""
    tfrec_fnames = tf.io.gfile.glob(path + '/train_tfrecords/ld_train*.tfrec')
    label_to_disease = pd.read_json(os.path.join(
        path, 'label_num_to_disease_map.json'), typ='series')
    train_csv = pd.read_csv(os.path.join(path, 'train.csv'))
    train_csv['disease'] = train_csv['label'].map(label_to_disease)
    
    # Split into train and validation (75:25)
    train_fnames = tfrec_fnames[:12]
    valid_fnames = tfrec_fnames[12:]
    
    return train_fnames, valid_fnames, train_csv, label_to_disease


def count_data_items(filenames):
    """Count number of samples in TFRecord files."""
    n = [int(re.compile(r"-([0-9]*)\.").search(fname).group(1))
         for fname in filenames]
    return np.sum(n)


def _parse_function(proto):
    """Parse TFRecord examples."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }

    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    target = tf.one_hot(parsed_features['target'], depth=NUM_CLASSES)
    return image, target


def load_dataset(tfrecords_fnames):
    """Load dataset from TFRecord files."""
    raw_ds = tf.data.TFRecordDataset(tfrecords_fnames, num_parallel_reads=AUTO)
    parsed_ds = raw_ds.map(_parse_function, num_parallel_calls=AUTO)
    return parsed_ds


def preprocess_fn(image, label):
    """Preprocess image and label."""
    image = image / 255.0
    image = tf.image.resize(image, RESIZED_IMAGE)
    # extended the training labels from 5 to 6 as to match the cassava model's output which included a "background" label
    label = tf.concat([label, [0]], axis=0)
    return image, label


def data_augment(image, target):
    modified = tf.image.random_flip_left_right(image)
    modified = tf.image.random_flip_up_down(modified)
    modified = tf.image.random_brightness(modified, 0.1)
    modified = tf.image.random_saturation(modified, 0.8, 1.2)
    modified = tf.image.rot90(modified, k=tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    modified = tf.image.random_crop(
        tf.image.resize_with_crop_or_pad(modified, 240, 240), [224, 224, 3])
    modified = tf.image.random_hue(modified, 0.1)
    modified = tf.clip_by_value(modified, 0.0, 1.0)
    return modified, target


def build_datasets(train_fnames, valid_fnames, with_aug):
    """Build training and validation datasets."""
    # Training dataset
    train_ds = load_dataset(train_fnames)
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=AUTO)
    
    if with_aug:
        train_ds = train_ds.map(data_augment, num_parallel_calls=AUTO)
    
    train_ds = train_ds.repeat().shuffle(2048, seed=RANDOM_SEED).batch(
        BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    # Validation dataset
    valid_ds = load_dataset(valid_fnames)
    valid_ds = valid_ds.map(preprocess_fn, num_parallel_calls=AUTO)
    valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    return train_ds, valid_ds


def build_model():
    """Build and compile the model."""
    os.environ["TFHUB_CACHE_DIR"] = "/kaggle/working"
    
    # Load pre-trained model from TFHub
    hub_layer = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2',
                              trainable=True)
    
    # Build model
    model = tf_keras.Sequential([
        tf_keras.Input(shape=(*RESIZED_IMAGE, 3)),
        hub_layer
    ])
    
    # Compile model
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=5e-6, weight_decay=1e-5),
        loss=tf_keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model


def get_callbacks():
    """Get training callbacks."""
    early_stop = tf_keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0,
        patience=5, 
        mode='max', 
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = tf_keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5,
        patience=2, 
        min_delta=0,
        mode='max', 
        verbose=1
    )
    
    # Add ModelCheckpoint callback to save the best model
    checkpoint = tf_keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Add TensorBoard callback for visualization
    tensorboard = tf_keras.callbacks.TensorBoard(
        log_dir='/kaggle/working',
        histogram_freq=1,
        write_graph=True
    )
    
    # Create callback to store history for learning curve
    class StoreHistory(tf_keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.history = {}
            
    history_callback = StoreHistory()
    
    return [early_stop, reduce_lr, checkpoint, tensorboard, history_callback]


def plot_learning_curve(history):
    """Plot learning curves for training and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


def main():
    """Main function to run the pipeline."""
    
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
        
    # Set path to data
    PATH = '/kaggle/input/cassava-leaf-disease-classification'
    
    # Load data
    print("Loading data...")
    train_fnames, valid_fnames, train_csv, label_to_disease = load_data(PATH)
    
    # Print dataset information
    print(f"\nDiseases in dataset:")
    for label, disease in label_to_disease.items():
        print(f"  Class {label}: {disease}")
    
    print(f"\nSample training data:")
    print(train_csv.head())
    
    # Count data samples
    n_train = count_data_items(train_fnames)
    n_valid = count_data_items(valid_fnames)
    train_steps = n_train // BATCH_SIZE
    
    # Build datasets
    print("\nBuilding datasets...")
    train_ds, valid_ds = build_datasets(train_fnames, valid_fnames, with_aug=True)
    
    # Build and display model
    print("\nBuilding model...")
    model = build_model()
    
    # Train model
    print("\nTraining model...")
    callbacks = get_callbacks()
    
    # Train the model
    history = model.fit(
        train_ds, 
        validation_data=valid_ds,
        epochs=30, 
        steps_per_epoch=train_steps,
        callbacks=callbacks
    )
    
    # Plot learning curves
    print("\nPlotting learning curves...")
    plot_learning_curve(history)
    
    # Save the final model
    model.save('best_model.h5')
    print("\nTraining complete! Model saved as 'best_model.h5'")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    main()