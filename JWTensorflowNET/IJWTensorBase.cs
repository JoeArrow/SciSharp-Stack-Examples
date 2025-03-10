#region © 2025 Joe Arrowood (JoeWare).
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow;

namespace JWTensorflowNET
{
    public interface IJWTensorBase
    {
        BaseConfig Config { get; set; }

        BaseConfig InitConfig();

        bool Run(IReq? req = null);

        void BuildModel();

        /// -----------------------------------------------
        /// <summary>
        ///     Build dataflow graph, train and predict
        /// </summary>
        /// <returns></returns>

        void Test();
        void Train();
        string FreezeModel();

        void Predict();

        Graph BuildGraph();
        Graph ImportGraph();

        /// -----------------------------------------------
        /// <summary>
        ///     Prepare dataset
        /// </summary>

        void PrepareData(IReq req);
    }
}
