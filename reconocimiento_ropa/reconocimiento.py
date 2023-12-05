import tensorflow as tf
import tensorflow_datasets as tfds

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)


