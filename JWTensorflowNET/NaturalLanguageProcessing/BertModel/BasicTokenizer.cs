#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Globalization;

using Tensorflow;

namespace BERT
{
    // ----------------------------------------------------
    /// <summary>
    ///     BasicTokenizer Description
    /// </summary>

    public class BasicTokenizer
    {
        bool do_lower_case;
        bool strip_accents;
        public List<string> never_split;

        // ------------------------------------------------

        public BasicTokenizer(bool do_lower_case, bool strip_accents = true, List<string> never_split = null)
        {
            this.do_lower_case = do_lower_case;
            this.strip_accents = strip_accents;
            this.never_split = never_split == null ? new List<string>() : never_split;
        }

        // ------------------------------------------------

        public static bool _is_control(char c)
        {
            // -------------------------------------------------
            // Checks whether or not `c` is a control character.

            if(c == '\t' || c == '\n' || c == '\r') { return false; }

            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);

            if(cat.ToString().StartsWith("Other")) { return true; }

            return false;
        }

        // ------------------------------------------------

        public static bool _is_whitespace(char c)
        {
            // ----------------------------------------------------
            // Checks whether or not `c` is a whitespace character.

            if(c == '\t' || c == '\n' || c == '\r') { return false; }

            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);

            if(cat.ToString() == "SpaceSeparator") { return true; }

            return false;
        }

        // ------------------------------------------------

        public static bool _is_punctuation(char c)
        {
            int cp = (int)c;

            if((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) { return true; }

            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);

            if(cat.ToString().Contains("Punctuation")) { return true; }

            return false;
        }

        // ------------------------------------------------

        public static List<string> whitespace_tokenize(string text)
        {
            // ----------------------------------------------------------------
            // Runs basic whitespace cleaning and splitting on a piece of text.

            text = text.Trim();
            var tokens = text.Split(' ');
            return new List<string>(tokens);
        }

        // ------------------------------------------------

        public string _run_strip_accents(string text)
        {
            // ------------------------------------
            // Strips accents from a piece of text.
            // Normalize(text); // Not importance.

            List<char> output = new List<char>();

            foreach(var c in text)
            {
                var cat = CharUnicodeInfo.GetUnicodeCategory(c);

                if(cat == UnicodeCategory.NonSpacingMark) { continue; }

                output.Add(c);
            }

            return string.Join("", output);
        }

        // ------------------------------------------------

        public List<string> _run_split_on_punc(string text, List<string> never_split = null)
        {
            // --------------------------------------
            // Splits punctuation on a piece of text.

            if(never_split != null && never_split.Contains(text))
            {
                return new List<string>(new string[] { text });
            }

            char[] chars = text.ToArray();
            int i = 0;
            bool start_new_word = true;
            List<List<char>> output = new List<List<char>>();

            while(i < chars.Length)
            {
                char chr = chars[i];

                if(_is_punctuation(chr))
                {
                    var tmp = new List<char>();
                    tmp.Add(chr);
                    output.Add(new List<char>(tmp));
                    start_new_word = true;
                }
                else
                {
                    if(start_new_word) output.Add(new List<char>());
                    start_new_word = false;
                    output[output.Count - 1].Add(chr);
                }

                i++;
            }

            var res = new List<string>();

            foreach(var x in output) { res.Add(string.Join("", x)); }

            return res;
        }

        // ------------------------------------------------

        public List<string> tokenize(string text, List<string> never_split = null)
        {
            text = _clean_text(text);
            var orig_tokens = whitespace_tokenize(text);
            List<string> split_tokens = new List<string>();

            foreach(string token in orig_tokens)
            {
                var _token = token;

                if(!never_split.Contains(token))
                {
                    if(do_lower_case)
                    {
                        _token = _token.ToLower();

                        if(this.strip_accents != false)
                        {
                            _token = this._run_strip_accents(_token);
                        }
                    }
                    else if(strip_accents)
                    {
                        _token = this._run_strip_accents(_token);
                    }

                    split_tokens.extend(this._run_split_on_punc(_token, never_split));
                }
            }

            var output_tokens = whitespace_tokenize(string.Join(" ", split_tokens));
            return output_tokens;
        }

        // ------------------------------------------------

        protected string _clean_text(string text)
        {
            // ------------------------------------------------------------------
            // Performs invalid character removal and whitespace cleanup on text.

            var output = new List<char>();

            foreach(char c in text)
            {
                int cp = (int)c;
                if(cp == 0 || cp == 0xFFFD || _is_control(c)) { continue; }
                if(_is_whitespace(c)) { output.Add(' '); }
                else { output.Add(c); }
            }

            return string.Join("", output);
        }
    }
}
