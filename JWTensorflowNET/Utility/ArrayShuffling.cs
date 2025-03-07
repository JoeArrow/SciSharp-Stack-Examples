#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

namespace JWTensorflowNET.Utility
{
    // ----------------------------------------------------
    /// <summary>
    ///     ArrayShuffling Description
    /// </summary>

    public static class ArrayShuffling
    {
        public static T[] Shuffle<T>(this Random rng, T[] array)
        {
            int n = array.Length;

            while(n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }

            return array;
        }
    }
}
