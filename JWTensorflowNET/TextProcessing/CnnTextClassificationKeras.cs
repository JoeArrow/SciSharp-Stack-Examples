#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow.Keras.Utils;

namespace JWTensorflowNET.TextProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     CnnTextClassificationKeras Description
    /// </summary>

    public class CnnTextClassificationKeras : absSciSharpBase, IJWTensorBase
    {
        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "CNN Text Classification (Keras)",
            Enabled = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            return true;
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            var fileName = "aclImdb_v1.tar.gz";
            var url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz";
            var data_dir = Path.GetTempPath();

            Web.Download(url, data_dir, fileName);
            Compress.ExtractGZip(Path.Join(data_dir, fileName), data_dir);
            data_dir = Path.Combine(data_dir, "aclImdb_v1");
        }
    }
}
