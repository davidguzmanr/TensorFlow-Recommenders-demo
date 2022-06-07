import tensorflow_recommenders as tfrs
import tensorflow as tf

from typing import Dict, Text

class ECommerceModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(
        self,
        user_model: tf.keras.Model,
        item_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval,
    ):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model

        # Set up a retrieval task.
        self.task = task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features['user_id'])
        item_embeddings = self.item_model(features['item_id'])

        return self.task(user_embeddings, item_embeddings)