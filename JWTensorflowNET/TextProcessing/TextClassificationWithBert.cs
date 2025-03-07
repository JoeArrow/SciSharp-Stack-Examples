#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Hub;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using JWTensorflowNET.ReqResp;

namespace JWTensorflowNET.TextProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     TextClassificationWithBert Description
    /// </summary>

    public class TextClassificationWithBert : absSciSharpBase, IJWTensorBase
    {
        NDArray train_data, train_labels, test_data, test_labels;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = false,
            Name = "Classify text with BERT",
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData();
            return false;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
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

            foreach(var (text_batch, label_batch) in raw_train_ds.take(1))
            {
                foreach(var i in range(3))
                {
                    print("Review", text_batch.StringData()[i]);
                    print("Label", label_batch.numpy()[i]);
                }
            }

            print("Label 0 corresponds to", raw_train_ds.class_names[0]);
            print("Label 1 corresponds to", raw_train_ds.class_names[1]);

            var raw_val_ds = keras.preprocessing.text_dataset_from_directory(train_dir,
                                                                             batch_size: batch_size,
                                                                             validation_split: 0.2f,
                                                                             subset: "validation",
                                                                             seed: seed);

            var test_dir = Path.Combine(data_dir, "test");
            var raw_test_ds = keras.preprocessing.text_dataset_from_directory(test_dir, batch_size: batch_size);

            var max_features = 10000;
            var sequence_length = 250;

            Func<Tensor, Tensor> custom_standardization = input_data =>
            {
                var lowercase = tf.strings.lower(input_data);
                var stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ");

                return tf.strings.regex_replace(stripped_html,
                                                "'[!\"\\#\\$%\\&\'\\(\\)\\*\\+,\\-\\./:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]'",
                                                "");
            };

            var tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3";
            var bert_model = hub.KerasLayer(tfhub_handle_encoder);
        }
    }
}
