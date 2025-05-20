import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tf_keras

# Constants
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1
IMAGE_SIZE = [512, 512]
TARGET_SIZE = [224, 224]
NUM_CLASSES = 5
TEST_PATH = '../input/cassava-leaf-disease-classification/test_tfrecords/'


def create_model():
    """Creates and loads the pre-trained cassava disease classification model."""
    # Initialize the hub layer
    hub_layer = hub.KerasLayer(
        '/kaggle/input/cropnet/keras/pretrained/1')

    # Create the model
    model = tf_keras.Sequential()
    model.add(tf_keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)))
    model.add(hub_layer)

    # Load pre-trained weights
    model.load_weights(
        "/kaggle/input/final/keras/best/1/best_model.h5")

    return model


def _parse_function(proto):
    """Parse TFRecord example into image, target, and image_name."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }

    parsed_features = tf.io.parse_single_example(proto, feature_description)

    # Decode and reshape image
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.cast(image, tf.float32)  # :: [0.0, 255.0]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    # One-hot encode the target
    target = tf.one_hot(parsed_features['target'], depth=NUM_CLASSES)
    image_id = parsed_features['image_name']

    return image, target, image_id


def _preprocess_fn(image, label, image_id):
    """Preprocess image and label for model input."""
    image = image / 255.0
    image = tf.image.resize(image, TARGET_SIZE)
    # extended the training labels from 5 to 6 as to match the cassava model's output which included a "background" label
    label = tf.concat([label, [0]], axis=0)
    return image, label, image_id


def load_dataset(tfrecords_fnames):
    """Load dataset from TFRecord files."""
    raw_ds = tf.data.TFRecordDataset(tfrecords_fnames, num_parallel_reads=AUTO)
    parsed_ds = raw_ds.map(_parse_function, num_parallel_calls=AUTO)
    return parsed_ds


def build_dataset(fnames, is_training=False):
    """Build dataset with appropriate preprocessing."""
    ds = load_dataset(fnames)
    ds = ds.map(_preprocess_fn, num_parallel_calls=AUTO)

    # Add batching and prefetching
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO)

    return ds


def get_predictions(model, dataset):
    """Run prediction on the dataset and return results."""
    # Get model predictions
    preds = model.predict(dataset)
    labels = tf.argmax(preds, axis=-1).numpy()

    # Extract image names from dataset
    names = []
    for item in dataset:
        names.append(item[2].numpy())

    names = np.concatenate(names)
    names = [name.decode() for name in names]

    return names, labels


def create_submission(image_ids, predicted_labels, output_file="submission.csv"):
    """Create and save submission CSV file."""
    submission_df = pd.DataFrame(
        {'image_id': image_ids, 'label': predicted_labels})
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


def main():
    """Main execution function."""
    # 1. Create and load model
    model = create_model()
    print("Model loaded successfully.")

    # 2. Prepare test dataset
    test_fnames = [TEST_PATH + fname for fname in os.listdir(TEST_PATH)]
    test_ds = build_dataset(test_fnames)
    print(f"Test dataset prepared with {len(test_fnames)} files.")

    # 3. Make predictions
    image_ids, predicted_labels = get_predictions(model, test_ds)
    print(f"Predictions completed for {len(image_ids)} images.")

    # 4. Create submission file
    create_submission(image_ids, predicted_labels)
    print("Done!")


if __name__ == "__main__":
    main()
