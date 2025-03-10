#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     ToyResNet Description
    /// </summary>

    public class ToyResNet : absSciSharpBase, IJWTensorBase
    {
        IModel model;
        ICallback result;
        NDArray x_test, y_test;
        NDArray x_train, y_train;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Toy ResNet",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.enable_eager_execution();

            BuildModel();
            PrepareData(req);
            Train();

            return result.history["accuracy"].Last() > 0.22;
        }

        // ------------------------------------------------

        public override void BuildModel()
        {
            var inputs = keras.Input(shape: (32, 32, 3), name: "img");
            var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
            x = layers.Conv2D(64, 3, activation: "relu").Apply(x);

            x = layers.BatchNormalization().Apply(x);
            var block_1_output = layers.MaxPooling2D(3).Apply(x);

            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_2_output = layers.Add().Apply(new Tensors(x, block_1_output));

            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_3_output = layers.Add().Apply(new Tensors(x, block_2_output));

            x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
            x = layers.GlobalAveragePooling2D().Apply(x);
            x = layers.Dense(256, activation: "relu").Apply(x);
            x = layers.Dropout(0.5f).Apply(x);
            var outputs = layers.Dense(10).Apply(x);

            model = keras.Model(inputs, outputs, name: "toy_resnet");
            model.summary();

            model.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
                loss: keras.losses.CategoricalCrossentropy(from_logits: true),
                metrics: new[] { "accuracy" });
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();

            x_train = x_train / 255.0f;
            x_test = x_test / 255.0f;

            y_train = np_utils.to_categorical(y_train, 10);
            y_test = np_utils.to_categorical(y_test, 10);
        }

        // ------------------------------------------------

        public override void Train()
        {
            result = model.fit(x_train[new Slice(0, 2000)], 
                               y_train[new Slice(0, 2000)],
                               batch_size: 64,
                               epochs: 5,
                               validation_split: 0.2f);
        }
    }
}
