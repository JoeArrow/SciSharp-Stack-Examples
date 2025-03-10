#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Drawing;
using System.Diagnostics;

using Tensorflow.NumPy;
using Tensorflow.Keras.Utils;

using static Tensorflow.Binding;

using Console = Colorful.Console;
using JWTensorflowNET.ReqResp;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     ImageRecognitionInception Description
    /// </summary>

    public class ImageRecognitionInception : absSciSharpBase, IJWTensorBase
    {
        string dir = "ImageRecognitionInception";
        string pbFile = "tensorflow_inception_graph.pb";
        string labelFile = "imagenet_comp_graph_label_strings.txt";

        List<NDArray> file_ndarrays = new List<NDArray>();

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Image Recognition Inception",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.compat.v1.disable_eager_execution();

            PrepareData(req);

            var graph = tf.Graph().as_default();

            // ----------------------------
            // import GraphDef from pb file

            graph.Import(Path.Join(dir, pbFile));

            var input_name = "input";
            var output_name = "output";

            var input_operation = graph.OperationByName(input_name);
            var output_operation = graph.OperationByName(output_name);

            var labels = File.ReadAllLines(Path.Join(dir, labelFile));
            var result_labels = new List<string>();
            var sw = new Stopwatch();

            var sess = tf.Session(graph);

            foreach(var nd in file_ndarrays)
            {
                sw.Restart();

                var results = sess.run(output_operation.outputs[0], (input_operation.outputs[0], nd));
                results = np.squeeze(results);
                int idx = np.argmax(results);

                Console.WriteLine($"{labels[idx]} {results[idx]} in {sw.ElapsedMilliseconds}ms", Color.Tan);
                result_labels.Add(labels[idx]);
            }

            return result_labels.Contains("military uniform");
        }

        // ------------------------------------------------

        private NDArray ReadTensorFromImageFile(string file_name,
                                                int input_height = 224,
                                                int input_width = 224,
                                                int input_mean = 117,
                                                int input_std = 1)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.io.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");

            var cast = tf.cast(decodeJpeg, tf.float32);
            var dims_expander = tf.expand_dims(cast, 0);

            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);

            var sub = tf.subtract(bilinear, new float[] { input_mean });
            var normalized = tf.divide(sub, new float[] { input_std });

            var sess = tf.Session(graph);
            return sess.run(normalized);
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            Directory.CreateDirectory(dir);

            // --------------
            // get model file

            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";

            Web.Download(url, dir, "inception5h.zip");

            Compress.UnZip(Path.Join(dir, "inception5h.zip"), dir);

            // -----------------------
            // download sample picture

            Directory.CreateDirectory(Path.Join(dir, "img"));
            url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg";
            Web.Download(url, Path.Join(dir, "img"), "grace_hopper.jpg");

            url = $"https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/shasta-daisy.jpg";
            Web.Download(url, Path.Join(dir, "img"), "shasta-daisy.jpg");

            // ---------------
            // load image file

            var files = Directory.GetFiles(Path.Join(dir, "img"));

            for(int i = 0; i < files.Length; i++)
            {
                var nd = ReadTensorFromImageFile(files[i]);
                file_ndarrays.Add(nd);
            }
        }
    }
}
