import unittest

from xquik_export import XquikExportError, load_xquik_texts


class XquikExportTests(unittest.TestCase):
    def test_loads_documented_json_export_field(self):
        payload = b'[{"tweetText":"Great launch"},{"tweetText":"Buggy update"}]'

        self.assertEqual(load_xquik_texts(payload), ["Great launch", "Buggy update"])

    def test_loads_paginated_api_results(self):
        payload = b'{"results":[{"tweetText":"Fast reply"}],"hasMore":false}'

        self.assertEqual(load_xquik_texts(payload), ["Fast reply"])

    def test_loads_cli_jsonl(self):
        payload = b'{"tweetText":"First row"}\n{"full_text":"Second row"}'

        self.assertEqual(load_xquik_texts(payload), ["First row", "Second row"])

    def test_loads_documented_csv_header_with_bom(self):
        payload = "\ufeffTweet ID,Tweet Text\n1,Clean interface\n".encode()

        self.assertEqual(load_xquik_texts(payload), ["Clean interface"])

    def test_normalizes_compatible_text_headers(self):
        payload = b"tweet_text,tweet\nUseful feature,Compatible fallback\n"

        self.assertEqual(load_xquik_texts(payload), ["Useful feature"])

    def test_preserves_compatible_tweet_field(self):
        payload = b'[{"tweet":"Compatible fallback"}]'

        self.assertEqual(load_xquik_texts(payload), ["Compatible fallback"])

    def test_skips_blank_and_user_only_rows(self):
        payload = (
            b'[{"tweetText":" "},{"xUsername":"example"},{"body":"Useful feature"}]'
        )

        self.assertEqual(load_xquik_texts(payload), ["Useful feature"])

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(XquikExportError, "Invalid JSON on line 2"):
            load_xquik_texts(b'{"tweetText":"Good"}\n{"tweetText":')

    def test_rejects_non_utf8_files(self):
        with self.assertRaisesRegex(XquikExportError, "UTF-8"):
            load_xquik_texts(b"\xff\xfe")

    def test_rejects_oversized_files(self):
        with self.assertRaisesRegex(XquikExportError, "exceeds"):
            load_xquik_texts(b"1234", max_bytes=3)

    def test_rejects_too_many_text_rows(self):
        payload = b'[{"tweetText":"One"},{"tweetText":"Two"}]'

        with self.assertRaisesRegex(XquikExportError, "1-row limit"):
            load_xquik_texts(payload, max_rows=1)

    def test_row_limit_counts_records_without_text(self):
        payload = b'[{"xUsername":"one"},{"xUsername":"two"}]'

        with self.assertRaisesRegex(XquikExportError, "1-row limit"):
            load_xquik_texts(payload, max_rows=1)


if __name__ == "__main__":
    unittest.main()
