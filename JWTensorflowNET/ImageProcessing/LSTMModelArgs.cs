#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow.Keras.ArgsDefinition;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     LSTMModelArgs Description
    /// </summary>

    internal class LSTMModelArgs : ModelArgs
    {
        public int NumUnits { get; set; }
        public int NumClasses { get; set; }
        public float LearningRate { get; set; }
    }
}
