#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.NeuralNetworks
{
    // ----------------------------------------------------
    /// <summary>
    ///     NeuralNetXorKeras Description
    /// </summary>

    public class NeuralNetXorKeras : absSciSharpBase, IJWTensorBase
    {
        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "NN XOR in Keras",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            var y = np.array(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });
            var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });

            var model = keras.Sequential();

            model.add(keras.Input(2));
            model.add(keras.layers.Dense(32, keras.activations.Relu));
            model.add(keras.layers.Dense(1, keras.activations.Sigmoid));

            model.compile(optimizer: keras.optimizers.Adam(),
                          loss: keras.losses.MeanSquaredError(),
                          new[] { "accuracy" });

            model.fit(x, y, batch_size: 4, epochs: 200);
            model.evaluate(x, y);
            Tensor result = model.predict(x, 4);
            return result.ToArray<float>() is [< 0.5f, > 0.5f, > 0.5f, < 0.5f];
        }
    }
}
