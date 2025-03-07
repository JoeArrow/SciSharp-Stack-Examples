#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;

namespace JWTensorflowNET.NeuralNetworks
{
    public partial class FullyConnectedKeras
    {
        /// <summary>
        ///     Network parameters.
        /// </summary>
        
        public class NeuralNetArgs : ModelArgs
        {
            /// <summary>
            ///     1st layer number of neurons.
            /// </summary>
            
            public int NeuronOfHidden1 { get; set; }
            public Activation Activation1 { get; set; }

            /// <summary>
            /// 2nd layer number of neurons.
            /// </summary>
            public int NeuronOfHidden2 { get; set; }
            public Activation Activation2 { get; set; }

            public int NumClasses { get; set; }
        }
    }
}
