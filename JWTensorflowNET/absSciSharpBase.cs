#region © 2025 Joe Arrowood (JoeWare).
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Diagnostics.CodeAnalysis;

using Tensorflow;
using Tensorflow.Keras.Layers;

using JWTensorflowNET.ReqResp;

namespace JWTensorflowNET
{
    // ----------------------------------------------------
    /// <summary>
    ///     SciSharpBase Description
    /// </summary>

    [ExcludeFromCodeCoverage]
    public abstract class absSciSharpBase
    {
        public BaseConfig Config { get; set; }
        protected LayersApi layers = new LayersApi();

        // ----------------------------------------------------

        public virtual void BuildModel() { }

        // ----------------------------------------------------

        public virtual Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        // ----------------------------------------------------

        public virtual Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        // ----------------------------------------------------

        public virtual void PrepareData(IReq req)
        {
            throw new NotImplementedException();
        }

        // ----------------------------------------------------

        public virtual void Train() { }

        // ----------------------------------------------------

        public virtual void Test() { }

        // ----------------------------------------------------

        public virtual void Predict() { }

        // ----------------------------------------------------

        public virtual string FreezeModel()
        {
            throw new NotImplementedException();
        }
    }
}
