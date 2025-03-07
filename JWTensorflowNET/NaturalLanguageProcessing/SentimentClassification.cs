#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using static Tensorflow.KerasApi;

using SciSharp.Models;
using SciSharp.Models.TimeSeries;

namespace JWTensorflowNET.NaturalLanguageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     SentimentClassification Description
    /// </summary>

    public class SentimentClassification : absSciSharpBase, IJWTensorBase
    {
        ITimeSeriesTask task;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Text Sentiment Classification",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run()
        {
            var wizard = new ModelWizard();

            task = wizard.AddTimeSeriesTask<ConvolutionalModel>(new TaskOptions
            {
                WeightsPath = @"timeseries_linear_v1\saved_weights.h5"
            });

            task.SetModelArgs(new TimeSeriesModelArgs { });

            return true;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            // --------------------------------------------
            // tf.debugging.set_log_device_placement(true);

            string url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
            
            var dataset = keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                               untar: true,
                                               cache_dir: Path.GetTempPath(),
                                               cache_subdir: "aclImdb_v1");

            var data_dir = Path.Combine(dataset, "aclImdb");
            var train_dir = Path.Combine(data_dir, "train");

            int batch_size = 32;
            int seed = 42;
            var raw_train_ds = keras.preprocessing.text_dataset_from_directory(train_dir,
                                                                               batch_size: batch_size,
                                                                               validation_split: 0.2f,
                                                                               subset: "training",
                                                                               seed: seed);
        }
    }
}
