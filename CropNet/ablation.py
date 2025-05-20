import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Set constants
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = [512, 512]
RESIZED_IMAGE = [224, 224]
NUM_CLASSES = 5
RANDOM_SEED = 42
EPOCHS = 10


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
    """Apply data augmentation to images."""
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


def build_datasets(train_fnames, valid_fnames, with_aug, batch_size=BATCH_SIZE):
    """Build training and validation datasets."""
    # Training dataset
    train_ds = load_dataset(train_fnames)
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=AUTO)
    
    if with_aug:
        train_ds = train_ds.map(data_augment, num_parallel_calls=AUTO)
    
    train_ds = train_ds.repeat().shuffle(2048, seed=RANDOM_SEED).batch(
        batch_size, drop_remainder=True).prefetch(AUTO)
    
    # Validation dataset
    valid_ds = load_dataset(valid_fnames)
    valid_ds = valid_ds.map(preprocess_fn, num_parallel_calls=AUTO)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(AUTO)
    
    return train_ds, valid_ds


def build_model(learning_rate=2e-5, weight_decay=1e-5):
    """Build and compile the model."""
    os.environ["TFHUB_CACHE_DIR"] = "/kaggle/working"
    
    # Input layer
    inputs = tf_keras.Input(shape=(*RESIZED_IMAGE, 3))
    
    # Load pre-trained model from TFHub
    hub_layer = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2',
                                trainable=True)
    outputs = hub_layer(inputs)
    
    # Build model
    model = tf_keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf_keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(run_name):
    """Get training callbacks."""
    early_stop = tf_keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0,
        patience=3,
        mode='max', 
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = tf_keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5,
        patience=1, 
        min_delta=0,
        mode='max', 
        verbose=1
    )
    
    # Add ModelCheckpoint callback to save the best model
    checkpoint = tf_keras.callbacks.ModelCheckpoint(
        f'best_model_{run_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    return [early_stop, reduce_lr, checkpoint]


def run_experiment(config):
    """Run a single experiment with the given configuration."""
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Set path to data
    PATH = '/kaggle/input/cassava-leaf-disease-classification'
    
    # Load data
    print(f"Running experiment: {config['name']}")
    train_fnames, valid_fnames, _, _ = load_data(PATH)
    
    # Count data samples
    n_train = count_data_items(train_fnames)
    train_steps = n_train // config['batch_size']
    
    # Build datasets
    train_ds, valid_ds = build_datasets(
        train_fnames, 
        valid_fnames, 
        with_aug=config['use_augmentation'],
        batch_size=config['batch_size']
    )
    
    # Build model
    model = build_model(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # Get callbacks
    callbacks = get_callbacks(config['name'])
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        train_ds, 
        validation_data=valid_ds,
        epochs=EPOCHS, 
        steps_per_epoch=train_steps,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    # Get best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    
    # Return results
    return {
        'name': config['name'],
        'history': history.history,
        'best_val_accuracy': best_val_acc,
        'training_time': training_time
    }


def ablation_study():
    """Run ablation study with different model configurations."""
    # Base configuration (full model)
    base_config = {
        'name': 'baseline',
        'learning_rate': 2e-5,
        'weight_decay': 1e-5,
        'batch_size': BATCH_SIZE,
        'use_augmentation': True,
    }
    
    # Define ablation configurations
    configs = [
        base_config,
        # Ablate data augmentation
        {**base_config, 'name': 'no_augmentation', 'use_augmentation': False},
        # Different batch size
        {**base_config, 'name': 'batch_size_16', 'batch_size': 16},
        # No weight decay
        {**base_config, 'name': 'no_weight_decay', 'weight_decay': 0.0},
    ]
    
    # Run all experiments
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
        
        # Print results so far
        print(f"\nResults for {result['name']}:")
        print(f"  Best validation accuracy: {result['best_val_accuracy']:.4f}")
        print(f"  Training time: {result['training_time']:.2f} seconds")
    
    return results


def plot_ablation_results(results):
    """Plot ablation study results."""
    # Extract results for plotting
    names = [r['name'] for r in results]
    accuracies = [r['best_val_accuracy'] for r in results]
    times = [r['training_time'] / 60 for r in results]  # Convert to minutes
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot validation accuracies
    bars1 = ax1.bar(names, accuracies, color='skyblue')
    ax1.set_title('Best Validation Accuracy by Configuration')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Validation Accuracy')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot training times
    bars2 = ax2.bar(names, times, color='salmon')
    ax2.set_title('Training Time by Configuration')
    ax2.set_ylabel('Training Time (minutes)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.show()
    
    # Also plot learning curves for each configuration
    plt.figure(figsize=(20, 15))
    
    for i, result in enumerate(results):
        # Plot training & validation accuracy
        plt.subplot(3, 2, i+1)
        plt.plot(result['history']['accuracy'], label='Training')
        plt.plot(result['history']['val_accuracy'], label='Validation')
        plt.title(f'Model Accuracy: {result["name"]}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ablation_learning_curves.png')
    plt.show()


def plot_comparison_chart(results):
    """Create a comparison chart showing relative performance impact."""
    # Get baseline result
    baseline = next(r for r in results if r['name'] == 'baseline')
    baseline_acc = baseline['best_val_accuracy']
    
    # Calculate relative performance differences
    names = [r['name'] for r in results if r['name'] != 'baseline']
    rel_diffs = [(r['best_val_accuracy'] - baseline_acc) * 100 for r in results if r['name'] != 'baseline']
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, rel_diffs)
    
    # Color bars based on positive/negative impact
    for i, bar in enumerate(bars):
        if rel_diffs[i] < 0:
            bar.set_color('salmon')
        else:
            bar.set_color('lightgreen')
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlabel('Change in Validation Accuracy (%)')
    plt.title('Impact of Each Ablation Relative to Baseline')
    
    # Add values at the end of bars
    for i, v in enumerate(rel_diffs):
        plt.text(v + 0.1 if v >= 0 else v - 0.6, i, f'{v:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('ablation_comparison.png')
    plt.show()


def generate_report(results):
    """Generate a comprehensive report of the ablation study."""
    # Create a DataFrame with results
    df = pd.DataFrame([
        {
            'Configuration': r['name'],
            'Best Val Accuracy': r['best_val_accuracy'],
            'Training Time (min)': r['training_time'] / 60,
            'Accuracy Change (%)': (r['best_val_accuracy'] - results[0]['best_val_accuracy']) * 100 
                                    if r['name'] != 'baseline' else 0.0
        }
        for r in results
    ])
    
    # Sort by validation accuracy (descending)
    df = df.sort_values('Best Val Accuracy', ascending=False)
    
    # Print the report
    print("\n===== ABLATION STUDY REPORT =====")
    print("\nConfigurations ranked by validation accuracy:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Analysis of each component's impact
    print("\nComponent Impact Analysis:")
    for r in results[1:]:  # Skip baseline
        impact = (r['best_val_accuracy'] - results[0]['best_val_accuracy']) * 100
        print(f"  {r['name']}: {impact:.2f}% impact on accuracy")
    
    # Save results to CSV
    df.to_csv('ablation_results.csv', index=False)
    print("\nResults saved to 'ablation_results.csv'")


if __name__ == "__main__":
    # Enable mixed precision for faster training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Run ablation study
    print(f"Starting ablation study at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = ablation_study()
    
    # Plot results
    plot_ablation_results(results)
    plot_comparison_chart(results)
    
    # Generate report
    generate_report(results)
    
    print(f"Ablation study completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")