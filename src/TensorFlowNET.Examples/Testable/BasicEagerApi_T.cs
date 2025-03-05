#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System;

using Tensorflow;

using TensorFlowNET.Examples.ReqResp;

using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.Testable
{
    // ----------------------------------------------------
    /// <summary>
    ///     BasicEagerApi_T Description
    /// </summary>

    public class BasicEagerApi_T : SciSharpExample//, IExample
    {
        private Tensor _a;
        private Tensor _b;

        // ------------------------------------------------

        public ExampleConfig InitConfig() => Config = new ExampleConfig
        {
            Name = "Basic Eager"
        };

        // ----------------------------------------------------

        public bool Run(IReq req)
        {
            // -------------
            // Set Eager API

            _a = tf.constant(req.GetValue<int>("a"));
            _b = tf.constant(req.GetValue<int>("b"));

            Console.WriteLine("Setting Eager mode...");
            tf.enable_eager_execution();

            // -----------------------
            // Define constant tensors

            Console.WriteLine("Define constant tensors");

            Console.WriteLine($"a = {_a}");
            Console.WriteLine($"b = {_b}");

            // -------------------------------------------------
            // Run the operation without the need for tf.Session

            Console.WriteLine("Running operations, without tf.Session");

            Tensor sum = _a + _b;
            Console.WriteLine($"a + b = {sum}");

            Tensor product = _a * _b;
            Console.WriteLine($"a * b = {product}");

            // -----------------------------
            // Full compatibility with Numpy

            return true;
        }
    }
}
