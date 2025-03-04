#region © 2025 Joe Arrowood (JoeWare).
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

namespace JWTensorflowNET
{
    // ----------------------------------------------------
    /// <summary>
    ///     BaseConfig Description
    /// </summary>

    public class BaseConfig
    {
        /// -----------------------------------------------
        /// <summary>
        ///     Example name
        /// </summary>

        public string Name { get; set; }

        /// -----------------------------------------------
        /// <summary>
        ///     True to run example
        /// </summary>

        public bool Enabled { get; set; } = true;

        /// -----------------------------------------------
        /// <summary>
        ///     Set true to import the computation 
        ///     graph instead of building it.
        /// </summary>

        public bool IsImportingGraph { get; set; } = false;
    }
}
