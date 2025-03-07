#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     ImageClassificationKeras Description
    /// </summary>

    public class ImageClassificationKeras : absSciSharpBase, IJWTensorBase
    {
        Model model;
        int epochs = 10;
        int batch_size = 32;
        Shape img_dim = (64, 64);
        IDatasetV2 train_ds, val_ds;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Image Classification (Keras)",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();
            Train();

            return true;
        }

        // ------------------------------------------------

        public override void BuildModel()
        {
            var num_classes = 5;
            var layers = keras.layers;

            var myLayers = new List<ILayer>
            {

                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            };

            model = keras.Sequential(myLayers);

            model.compile(optimizer: keras.optimizers.Adam(),
                          loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                          metrics: new[] { "accuracy" });

            model.summary();
        }

        // ------------------------------------------------

        public override void Train()
        {
            model.fit(train_ds, validation_data: val_ds, epochs: epochs);
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            var fileName = "flower_photos.tgz";
            var url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz";
            var data_dir = Path.Combine(Path.GetTempPath(), "flower_photos");

            Web.Download(url, data_dir, fileName);
            Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir);
            data_dir = Path.Combine(data_dir, "flower_photos");

            // -----------------
            // convert to tensor

            train_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                        validation_split: 0.2f,
                                                                        subset: "training",
                                                                        seed: 123,
                                                                        image_size: img_dim,
                                                                        batch_size: batch_size);

            val_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                      validation_split: 0.2f,
                                                                      subset: "validation",
                                                                      seed: 123,
                                                                      image_size: img_dim,
                                                                      batch_size: batch_size);

            train_ds = train_ds.shuffle(1000).prefetch(buffer_size: -1);
            val_ds = val_ds.prefetch(buffer_size: -1);
        }
    }
}
