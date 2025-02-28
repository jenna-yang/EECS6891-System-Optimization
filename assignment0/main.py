import pdb
import matplotlib.pyplot as plt
            
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


colors = plt.cm.tab10.colors
def plot_results(histories):
    # Training Accuracy
    plt.subplot(1, 2, 1)
    for i, (name, history) in enumerate(histories):
        plt.plot(history.history['sparse_categorical_accuracy'], 
                    color=colors[i], linestyle='-', linewidth=5, alpha=0.7, label=name)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', fontsize=14)

    # Validation Accuracy
    plt.subplot(1, 2, 2)
    for i, (name, history) in enumerate(histories):
        plt.plot(history.history['val_sparse_categorical_accuracy'], 
                    color=colors[i], linestyle='--', linewidth=5, alpha=0.7, label=name)
    plt.title('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', fontsize=14)

    plt.tight_layout()
    plt.show()        

'''
# naive model implementation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)

# for i in range(5):
#     plt.imshow(test_images[i], cmap='gray')
#     plt.title(f"Predicted: {predictions[i].argmax()} | True: {test_labels[i]}")
#     plt.show()

# plt.plot(history.history['sparse_categorical_accuracy'], linewidth=4, color="tab:purple")
# plt.plot(history.history['val_sparse_categorical_accuracy'], linewidth=4, color="tab:green")
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test', fontsize=14], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'], linewidth=4, color="tab:purple")
# plt.plot(history.history['val_loss'], linewidth=4, color="tab:green")
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test', fontsize=14], loc='upper left')
# plt.show()

'''

def ActivationEffect():
    # 1. Effect of activation function
    activation_list = ['relu', 'sigmoid', 'gelu', 'elu', 'leaky_relu']
    histories = []
    colors = plt.cm.tab10.colors[:len(activation_list)]

    for activation in activation_list:
        print(f"Training with activation: {activation}")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
        ])
        if activation == 'leaky_relu':
            model.add(tf.keras.layers.Dense(128))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        else:
            model.add(tf.keras.layers.Dense(128, activation=activation))
        model.add(tf.keras.layers.Dense(10))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_test,
            verbose=1,
        )
        histories.append((activation, history))

    test_images, test_labels = next(iter(ds_test))
    predictions = model.predict(test_images)
    for i, (activation, history) in enumerate(histories):
        color = colors[i]
        plt.plot(history.history['loss'], 
                color=color, linestyle='-', linewidth=5, alpha=0.5, label=f'{activation}')
    plt.title('Model Training Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()
    for i, (activation, history) in enumerate(histories):
        color = colors[i]
        plt.plot(history.history['val_loss'], 
                color=color, linestyle='--', linewidth=5, alpha=0.5, label=f'{activation}')
    plt.title('Model Test Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()
    for i, (activation, history) in enumerate(histories):
        color = colors[i]
        plt.plot(history.history['sparse_categorical_accuracy'], 
                color=color, linestyle='-', linewidth=5, alpha=0.5, label=f'{activation}')
    plt.title('Model Training Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.show()
    for i, (activation, history) in enumerate(histories):
        color = colors[i]
        plt.plot(history.history['val_sparse_categorical_accuracy'], 
                color=color, linestyle='--', linewidth=5, alpha=0.5, label=f'{activation}')
    plt.title('Model Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.show()


def ArchitectureEffect():
    # 1. Effect of feature size
    feature_sizes = [64, 128, 256]
    feature_histories = []
    
    # 2. Effect of layer depth
    layer_counts = [1, 2, 3]
    layer_histories = []
    
    # 3. Effect of layer type (Dense vs Conv)
    layer_types = ['dense', 'conv']
    conv_histories = []
    
    
    print("\n=== Feature Size Experiment ===")
    for units in feature_sizes:
        print(f"Training with {units} units per layer")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_test,
            verbose=1
        )
        feature_histories.append((f'{units} units', history))

    print("\n=== Layer Depth Experiment ===")
    for layers in layer_counts:
        print(f"Training with {layers} hidden layers")
        model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28))])
        
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            
        model.add(tf.keras.layers.Dense(10))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_test,
            verbose=1
        )
        layer_histories.append((f'{layers} layers', history))

    print("\n=== Layer Type Experiment ===")
    for layer_type in layer_types:
        print(f"Training with {layer_type} layers")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
        
        if layer_type == 'conv':
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            
        model.add(tf.keras.layers.Dense(10))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_test,
            verbose=1
        )
        conv_histories.append((layer_type, history))


    def plot_training_comparison(histories, title, ylabel):
        for i, (name, history) in enumerate(histories):
            plt.plot(history.history['sparse_categorical_accuracy'], 
                    color=colors[i], linestyle='-', linewidth=5, alpha=0.7, label=name)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', fontsize=14)
        plt.show()

    def plot_val_comparison(histories, title, ylabel):
        for i, (name, history) in enumerate(histories):
            plt.plot(history.history['val_sparse_categorical_accuracy'], 
                    color=colors[i], linestyle='--', linewidth=5, alpha=0.7, label=name)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', fontsize=14)
        plt.show()

    # plot_training_comparison(feature_histories, 
    #                'Feature Size: Training Accuracy', 
    #                'Accuracy')
    # plot_val_comparison(feature_histories, 
    #                'Feature Size: Validation Accuracy', 
    #                'Accuracy')
    # plot_training_comparison(layer_histories, 
    #                'Feature Size: Training Accuracy', 
    #                'Accuracy')
    # plot_val_comparison(layer_histories,
    #             'Layer Depth: Validation Accuracy',
    #             'Accuracy')
    # plot_training_comparison(conv_histories, 
    #                'Feature Size: Training Accuracy', 
    #                'Accuracy')
    # plot_val_comparison(conv_histories,
    #             'Layer Type: Validation Accuracy',
    #             'Accuracy')


def HyperparameterEffect():
    # 1. Learning rate experiment
    learning_rates = [0.1, 0.01, 0.001, 0.0005]
    lr_histories = []
    
    # 2. Batch size experiment
    batch_sizes = [32, 64, 128, 256, 512]
    batch_histories = []
    
    colors = plt.cm.tab10.colors
    
    # Experiment 1: Learning rates
    print("\n=== Learning Rate Experiment ===")
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_test,
            verbose=1
        )
        lr_histories.append((f'LR={lr}', history))

    # Experiment 2: Batch sizes
    print("\n=== Batch Size Experiment ===")
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        
        # Reload and reprocess dataset with current batch size
        (ds_train_raw, ds_test_raw), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=False,
            as_supervised=True,
            with_info=True,
        )
        
        train_ds = ds_train_raw.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples, seed=42)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        test_ds = ds_test_raw.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        history = model.fit(
            train_ds,
            epochs=15,
            validation_data=test_ds,
            verbose=1
        )
        batch_histories.append((f'Batch={batch_size}', history))

    plot_results(lr_histories)
    plot_results(batch_histories)


def EarlyEffect():
    def call_early_stopping():
        return EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

    histories = []
    layer_types = ['dense', 'conv', 'dense-early', 'conv-early']

    for layer_type in layer_types:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
        
        if layer_type == 'conv':
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            
        model.add(tf.keras.layers.Dense(10))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        
        if 'early' in layer_type:
            history = model.fit(
                ds_train,
                epochs=15,
                validation_data=ds_test,
                verbose=1,
                callbacks=[call_early_stopping()]
            )
        else:
            history = model.fit(
                ds_train,
                epochs=15,
                validation_data=ds_test,
                verbose=1
            )
        histories.append((layer_type, history))
    plot_results(histories)


if __name__ == "__main__":
    # ActivationEffect()
    # ArchitectureEffect()
    # HyperparameterEffect()
    EarlyEffect()