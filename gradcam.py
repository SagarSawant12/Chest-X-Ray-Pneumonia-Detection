import numpy as np
import tensorflow as tf

def get_gradcam_heatmap(model, img_array, layer_name):
    """
    Computes a smoothed, normalized Grad-CAM heatmap for a target convolutional layer.
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Target score for binary sigmoid output
        loss = predictions[:, 0]

    # Calculate gradients of the predicted score w.r.t. the conv feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight the channels by the pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to focus only on positive contributors
    heatmap = np.maximum(heatmap.numpy(), 0)
    
    # Safe normalization to prevent division by zero
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val

    return heatmap