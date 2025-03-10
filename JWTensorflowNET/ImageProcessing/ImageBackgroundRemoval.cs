#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow;
using Tensorflow.Keras.Utils;

using static Tensorflow.Binding;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     ImageBackgroundRemoval Description
    /// </summary>

    public class ImageBackgroundRemoval : absSciSharpBase, IJWTensorBase
    {
        string dataDir = "deeplabv3";
        string modelName = "frozen_inference_graph.pb";
        string modelDir = "deeplabv3_mnv2_pascal_train_aug";

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Image Background Removal",
            Enabled = false,
            IsImportingGraph = true
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData(req);

            // ----------------------------
            // import GraphDef from pb file

            var graph = new Graph().as_default();
            graph.Import(Path.Join(dataDir, modelDir, modelName));

            Tensor output = graph.OperationByName("SemanticPredictions");

            var sess = tf.Session(graph);

            // ---------------------------------
            // Runs inference on a single image.

            sess.run(output, new FeedItem(output, "[np.asarray(resized_image)]"));

            return false;
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            // -------------------------
            // Get mobile_net_model file

            string fileName = "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz";
            string url = $"http://download.tensorflow.org/models/{fileName}";

            Web.Download(url, dataDir, fileName);
            Compress.ExtractTGZ(Path.Join(dataDir, fileName), dataDir);
        }
    }
}
