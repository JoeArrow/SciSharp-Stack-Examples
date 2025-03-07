#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;

namespace JWTensorflowNET.Utility
{
    // ----------------------------------------------------
    /// <summary>
    ///     CoNLLDataset Description
    /// </summary>

    public class CoNLLDataset
    {
        static Dictionary<string, int> vocab_chars;
        static Dictionary<string, int> vocab_words;
        static Dictionary<string, int> vocab_tags;

        string _path;

        // ------------------------------------------------

        public CoNLLDataset(string path)
        {
            if(vocab_chars == null)
            {
                vocab_chars = load_vocab("filepath_chars");
            }

            if(vocab_words == null)
            {
                vocab_words = load_vocab("filepath_words");
            }

            if(vocab_tags == null)
            {
                vocab_tags = load_vocab("filepath_tags");
            }

            _path = path;
        }

        // ------------------------------------------------

        private (int[], int) processing_word(string word)
        {
            var char_ids = word.ToCharArray().Select(x => vocab_chars[x.ToString()]).ToArray();

            // ------------------
            // 1. preprocess word

            if(true)            // lowercase
            {
                word = word.ToLower();
            }

            // -----------------
            // 2. get id of word

            int id = vocab_words.GetValueOrDefault(word, vocab_words["$UNK$"]);

            return (char_ids, id);
        }

        private int processing_tag(string word)
        {
            // --------------
            // get id of word

            int id = vocab_tags.GetValueOrDefault(word, -1);
            return id;
        }

        // ------------------------------------------------

        private Dictionary<string, int> load_vocab(string filename)
        {
            var dict = new Dictionary<string, int>();
            int i = 0;

            File.ReadAllLines(filename)
                .Select(x => dict[x] = i++)
                .Count();

            return dict;
        }

        // ------------------------------------------------

        public IEnumerable<((int[], int)[], int[])> GetItems()
        {
            var lines = File.ReadAllLines(_path);

            int niter = 0;
            var words = new List<(int[], int)>();
            var tags = new List<int>();

            foreach(var l in lines)
            {
                string line = l.Trim();

                if(string.IsNullOrEmpty(line) || line.StartsWith("-DOCSTART-"))
                {
                    if(words.Count > 0)
                    {
                        niter++;
                        yield return (words.ToArray(), tags.ToArray());
                        words.Clear();
                        tags.Clear();
                    }
                }
                else
                {
                    var ls = line.Split(' ');

                    // ------------
                    // process word

                    var word = processing_word(ls[0]);
                    var tag = processing_tag(ls[1]);

                    words.Add(word);
                    tags.Add(tag);
                }
            }
        }
    }
}
