"""
Mask R-CNN
Multi-GPU Support for Keras.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

##################################################################################
# TF2.x MIGRATION NOTE (2025-05-30 by @copilot):
#
# This ParallelModel class implements a custom data parallelism strategy
# suitable for TensorFlow 1.x with Keras.
#
# For TensorFlow 2.x, the recommended and idiomatic way to achieve multi-GPU
# training (or other distributed training scenarios) is to use the
# `tf.distribute.Strategy` API, particularly `tf.distribute.MirroredStrategy`
# for single-host, multi-GPU data parallelism.
#
# Users migrating Mask R-CNN to TensorFlow 2.x should:
# 1. Remove the usage of this `ParallelModel` from `mrcnn/model.py`.
# 2. In their main training script, instantiate a `tf.distribute.Strategy`
#    (e.g., `strategy = tf.distribute.MirroredStrategy()`).
# 3. Create the Keras model (MaskRCNN instance) and compile it
#    *within* the strategy's scope (`with strategy.scope(): ...`).
#
# This `parallel_model.py` file is kept for users who might still be using
# a TensorFlow 1.x environment or for historical reference, but it is NOT
# recommended for use with the TensorFlow 2.x refactored Mask R-CNN code.
##################################################################################

import tensorflow as tf
# TF2 Change: Use tf.keras imports
# import keras.backend as K
# import keras.layers as KL
# import keras.models as KM
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM


class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.

    TF2.x NOTE: This class is for TensorFlow 1.x. Use tf.distribute.Strategy in TF2.x.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        if LooseVersion(tf.__version__) >= LooseVersion("2.0.0"):
            logging.warning(
                "ParallelModel is designed for TensorFlow 1.x. "
                "For TensorFlow 2.x, tf.distribute.Strategy is recommended. "
                "This class may not work as expected or efficiently in TF 2.x."
            )
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                             outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        
        # TF2 Change: self.inner_model.input_names might not exist or be preferred.
        # Using model.inputs (list of tensors) and their names.
        # input_slices = {name: tf.split(x, self.gpu_count)
        #                 for name, x in zip(self.inner_model.input_names,
        #                                    self.inner_model.inputs)}
        
        # A more robust way if input_names is not reliable:
        # Assuming inputs are tensors and their .name property can be used if unique.
        # Or, just use their order.
        # For simplicity, let's assume self.inner_model.inputs is a list of input tensors.
        # The original code used input_names to key the slices.
        # If input_names is not available, we might need to rely on order.
        
        # Let's try to stick to the original logic structure as much as possible
        # while acknowledging potential TF2 Keras API changes.
        # Keras models in TF2 still have .inputs and .input_names
        
        input_slices = {}
        if hasattr(self.inner_model, 'input_names') and self.inner_model.input_names:
            input_names_ref = self.inner_model.input_names
        else: # Fallback if input_names is not there or empty
            input_names_ref = [inp.name.split(':')[0] for inp in self.inner_model.inputs]


        for name, x_tensor in zip(input_names_ref, self.inner_model.inputs):
            input_slices[name] = tf.split(x_tensor, self.gpu_count)


        output_names_ref = []
        if hasattr(self.inner_model, 'output_names') and self.inner_model.output_names:
            output_names_ref = self.inner_model.output_names
        else: # Fallback if output_names is not there or empty
            output_names_ref = [out.name.split(':')[0] for out in self.inner_model.outputs]


        outputs_all = []
        for _ in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope: # Added 'as scope' for clarity
                    # Run a slice of inputs through this replica
                    
                    # Original: zipped_inputs = zip(self.inner_model.input_names, self.inner_model.inputs)
                    # inputs = [
                    #     KL.Lambda(lambda s: input_slices[name][i],
                    #               output_shape=lambda s: (None,) + s[1:])(tensor)
                    #     for name, tensor in zipped_inputs]
                    
                    # Rebuild inputs for the replica using the sliced tensors
                    replica_inputs = []
                    for idx, name in enumerate(input_names_ref):
                        # The Lambda was to ensure the slice op is in the graph for this tower.
                        # In TF2, this might be less critical if ops are placed eagerly or tf.function traces it.
                        # However, to replicate, we keep it.
                        # The output_shape lambda needs to correctly infer shape.
                        # (None,) + s[1:] assumes batch dim is first and unknown.
                        # For a slice, batch dim might be fixed.
                        # tf.split preserves rank, so s[1:] is okay.
                        
                        # Get the original tensor corresponding to this name/index
                        original_input_tensor = self.inner_model.inputs[idx]
                        
                        # The lambda for output_shape needs to operate on the shape of the *original* tensor's slice
                        # not the entire tensor `s`.
                        # Let's simplify the output_shape or make it more robust.
                        # If input_slices[name][i] is a tensor, its shape is known.
                        # However, Lambda in TF1 often needed explicit output_shape.

                        # Create a Lambda layer for each input slice
                        # The output_shape of the Lambda layer should match the shape of the slice.
                        # input_slices[name][i] is already a tensor.
                        # Wrapping it in another Lambda might be redundant if the slice op is already correctly placed.
                        # Let's try to use the slice directly, assuming it's placed on the correct device by the outer tf.device.
                        # If not, the Lambda is needed.
                        # For TF1 compatibility, the Lambda approach is safer.
                        
                        # To define output_shape for Lambda, we need the shape of the slice.
                        # This can be tricky if it's dynamic.
                        # Let original_input_shape = K.int_shape(original_input_tensor)
                        # slice_shape_part = original_input_shape[1:] if original_input_shape else None
                        # def get_output_shape(input_tensor_shape):
                        #    if input_tensor_shape and slice_shape_part:
                        #        return (input_tensor_shape[0] // self.gpu_count, ) + slice_shape_part # This is not quite right
                        #    return None # Fallback for dynamic
                        
                        # Simpler: just pass the slice. If replica model is created new, it adapts.
                        # But here, self.inner_model is called.
                        # The original Lambda was likely to ensure the slice operation is associated with this device tower.
                        slice_tensor = input_slices[name][i]
                        
                        # To ensure a new symbolic tensor for each tower's graph for this input slice:
                        # This lambda makes a new symbolic tensor from the slice.
                        # The output_shape is tricky. Let's assume the slice's shape is what we want.
                        # For a generic lambda s: s, output_shape should be s.
                        # The original output_shape was `lambda s: (None,) + s[1:]` which seems to assume
                        # the input `s` to the lambda is the *original* unsliced tensor.
                        # This is not how Lambda usually works; `s` is the input to the Lambda itself.
                        # Let's assume the input to Lambda is the slice itself.
                        # Then output_shape should be the shape of that slice.

                        # A more robust output_shape for the Lambda if it just passes through the slice:
                        def get_slice_shape(slice_tensor_for_shape):
                            # K.int_shape can return None for dynamic dimensions
                            # We need a symbolic shape, (None, dim1, dim2, ...)
                            # The batch size of the slice is original_batch_size / gpu_count
                            # Other dimensions are same as original_input_tensor.
                            orig_shape = K.int_shape(original_input_tensor)
                            if orig_shape:
                                return (None,) + orig_shape[1:] # Keep None for batch, rest same
                            return None # Dynamic

                        # This lambda is to ensure the slice is "used" in this name_scope/device.
                        # It's a bit of a TF1 graph manipulation.
                        # replica_input_slice = KL.Lambda(lambda x: x, output_shape=get_slice_shape(slice_tensor))(slice_tensor)
                        # Using the slice directly might be cleaner if TF2 handles placement.
                        # The original code's Lambda was more about re-wrapping the slice.
                        # Let's try to replicate the original intent of creating a new symbolic tensor per tower.
                        # The key is that `input_slices[name][i]` is already a tensor.
                        # The lambda `lambda s: input_slices[name][i]` with `(tensor)` as input to lambda
                        # means `s` is `tensor` (original input), and it returns the pre-computed slice.
                        # This seems overly complex. The lambda should operate on `s`.
                        # Let's assume `inputs` to the replica model should be these slices.
                        
                        # Original logic:
                        # inputs = [
                        #    KL.Lambda(lambda s: input_slices[name_orig][i],  <-- name_orig from zipped_inputs
                        #              output_shape=lambda s: (None,) + s[1:])(tensor_orig) <-- tensor_orig from zipped_inputs
                        #    for name_orig, tensor_orig in zip(self.inner_model.input_names, self.inner_model.inputs)
                        # ]
                        # This means for each original input `tensor_orig`, it creates a Lambda.
                        # The Lambda's input `s` is `tensor_orig`. The Lambda's output is `input_slices[name_orig][i]`.
                        # This is a way to symbolically use the pre-sliced tensor in the graph of this tower.
                        # The output_shape lambda `lambda s: (None,) + s[1:]` refers to shape of `tensor_orig`.
                        # This is unusual for output_shape. output_shape should describe the *output* of the Lambda.
                        
                        # Let's simplify, assuming input_slices[name][i] is the tensor for this replica's input
                        replica_inputs.append(input_slices[name][i])

                    # Create the model replica and get the outputs
                    # This reuses weights of self.inner_model.
                    outputs = self.inner_model(replica_inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs_on_gpus, name in zip(outputs_all, output_names_ref): # Use output_names_ref
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                
                # K.int_shape(outputs_on_gpus[0]) can be None if dynamic
                # Let's use tf.rank or check shape more carefully
                # If shape is () or (1,), it's likely a scalar loss/metric per GPU.
                
                # Check if the output is scalar-like (rank 0 or 1 with size 1)
                # This is a heuristic. Better way is if losses are explicitly named or handled.
                first_output_shape = K.int_shape(outputs_on_gpus[0])
                is_scalar_output = (first_output_shape == () or first_output_shape == (1,))

                if is_scalar_output: # Average if scalar-like
                    # Average
                    # m = KL.Lambda(lambda o_list: tf.add_n(o_list) / len(o_list), name=name)(outputs_on_gpus)
                    # A more direct way to average:
                    if len(outputs_on_gpus) > 0 :
                        m = tf.add_n(outputs_on_gpus) / float(len(outputs_on_gpus))
                        # Wrap in Lambda if it needs to be a Keras layer for model construction
                        m = KL.Lambda(lambda x: x, name=name + "_avg")(m) # Identity Lambda to name it
                    else: # Should not happen if gpu_count > 0
                        m = None # Or raise error
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs_on_gpus)
                if m is not None: merged.append(m)
        return merged


if __name__ == "__main__":
    # TF2 MIGRATION NOTE: The testing code below is for TF1.x and Keras standalone.
    # It will likely not run correctly as-is in a TF2.x environment without
    # significant adaptation (e.g., using tf.keras, handling eager execution,
    # replacing fit_generator, etc.).
    # This example illustrates the usage of ParallelModel in its original context.
    print("TF2 MIGRATION NOTE: The __main__ block in parallel_model.py is for TF1.x testing.")
    print("It is not expected to run correctly in TF2.x without significant changes.")

    # Conditional execution for safety, or comment out entirely for TF2 focus
    if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
        import os
        import numpy as np
        # import keras.optimizers # TF2: tf.keras.optimizers
        # from keras.datasets import mnist # TF2: tf.keras.datasets.mnist
        # from keras.preprocessing.image import ImageDataGenerator # TF2: tf.keras.preprocessing.image.ImageDataGenerator
        from tensorflow import keras as tf_keras # Use the aliased keras

        GPU_COUNT = 2 # Make sure you have 2 GPUs available for this test

        # Root directory of the project
        ROOT_DIR = os.path.abspath("../") # Adjust if necessary

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs_parallel_test")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        def build_model(x_train_shape, num_classes): # Pass shape instead of data
            # Reset default graph. Keras leaves old ops in the graph,
            # which are ignored for execution but clutter graph
            # visualization in TensorBoard.
            tf.compat.v1.reset_default_graph() # TF1 graph reset

            inputs = KL.Input(shape=x_train_shape[1:], name="input_image")
            x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",
                          name="conv1")(inputs)
            x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                          name="conv2")(x)
            x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
            x = KL.Flatten(name="flat1")(x)
            x = KL.Dense(128, activation='relu', name="dense1")(x)
            x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

            return KM.Model(inputs, x, name="digit_classifier_model")

        # Load MNIST Data
        (x_train, y_train), (x_test, y_test) = tf_keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1).astype('float32') / 255
        x_test = np.expand_dims(x_test, -1).astype('float32') / 255
        
        # Convert y_train and y_test to one-hot if using categorical_crossentropy,
        # or keep as integers for sparse_categorical_crossentropy.
        # The model uses softmax and sparse_categorical_crossentropy is fine.

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        # Build data generator and model
        datagen = tf_keras.preprocessing.image.ImageDataGenerator()
        # Check if GPUs are available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus or len(gpus) < GPU_COUNT:
            print(f"Warning: This test requires {GPU_COUNT} GPUs. Found {len(gpus)}. Skipping multi-GPU test.")
        else:
            print(f"Found {len(gpus)} GPUs. Proceeding with multi-GPU test using {GPU_COUNT} GPUs.")
            model = build_model(x_train.shape, 10)

            # Add multi-GPU support.
            model = ParallelModel(model, GPU_COUNT)

            # optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=5.0) # TF1
            optimizer = tf_keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=5.0) # TF2

            model.compile(loss='sparse_categorical_crossentropy', # sparse_categorical_crossentropy
                          optimizer=optimizer, metrics=['accuracy'])

            model.summary()

            # Train
            # model.fit_generator( # TF1
            model.fit( # TF2
                datagen.flow(x_train, y_train, batch_size=64 * GPU_COUNT), # Adjust batch size for total
                steps_per_epoch=x_train.shape[0] // (64 * GPU_COUNT), # Adjust steps
                epochs=1, # Keep epochs low for a quick test
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=[tf_keras.callbacks.TensorBoard(log_dir=MODEL_DIR, write_graph=True)]
            )
            print(f"Multi-GPU test (if run) finished. Check logs in {MODEL_DIR}")
    else:
        print("Skipping __main__ block of parallel_model.py as TensorFlow version is >= 2.0.0.")
