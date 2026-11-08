"""Offline tests: reject fabricated completion and corrupted official inventory."""
import copy
import unittest
from cvpr_inventory import parse_listing
from build_paper_index import validate, ROOT
import json

FIXTURE = '''<dt class="ptitle"><br><a href="/content/CVPR2026/html/A_paper.html">A &amp; B</a></dt>
<dd><input name="query_author" value="A Author"></dd>
<dd>[<a href="/content/CVPR2026/papers/A_paper.pdf">pdf</a>]</dd>'''


class PaperTests(unittest.TestCase):
    def test_official_metadata_and_entities(self):
        r = parse_listing(FIXTURE)[0]
        self.assertEqual(r['title'], 'A & B')
        self.assertEqual(r['authors'], ['A Author'])
        self.assertEqual(r['status'], 'queued')

    def test_duplicates_and_wrong_conference(self):
        for data in [FIXTURE+FIXTURE, FIXTURE.replace('/CVPR2026/', '/CVPR2026W/')]:
            with self.assertRaises(ValueError):
                parse_listing(data)

    def test_missing_source(self):
        with self.assertRaises(ValueError):
            parse_listing(FIXTURE.replace('>pdf</a>', '>missing</a>'))

    def test_real_inventory(self):
        m = json.loads((ROOT/'papers/CVPR2026/manifest.json').read_text(encoding='utf-8'))
        counts = validate(m)
        self.assertEqual(sum(counts.values()), m['listed_count'])

    def test_completion_cannot_be_just_a_flag(self):
        m = {'listed_count': 1, 'papers': [parse_listing(FIXTURE)[0]]}
        m['papers'][0]['status'] = 'reviewed'
        with self.assertRaises(ValueError):
            validate(m)
        bad = copy.deepcopy(m)
        bad['papers'][0]['note_path'] = '../../README.md'
        with self.assertRaises(ValueError):
            validate(bad)


if __name__ == '__main__':
    unittest.main()
