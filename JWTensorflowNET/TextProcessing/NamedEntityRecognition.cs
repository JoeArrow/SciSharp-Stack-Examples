#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

namespace JWTensorflowNET.TextProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     NamedEntityRecognition Description
    /// </summary>

    public class NamedEntityRecognition : absSciSharpBase, IJWTensorBase
    {
        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "NER",
            Enabled = false,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            throw new NotImplementedException();
        }
    }
}
