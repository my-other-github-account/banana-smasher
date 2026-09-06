#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

class ResumePlanTest(unittest.TestCase):
    def test_failed_boundary_keeps_accepted_and_excludes_empty_pair(self):
        path=Path(__file__).with_name('s4_missing_resume.py')
        self.assertTrue(path.exists(), 'missing-only failed-boundary resume is absent')
        spec=importlib.util.spec_from_file_location('resume',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        accepted=[{'cell':[41,229,'down']}]
        configs=[{'layer':41,'expert':e,'projection':p} for e in (229,230,231) for p in ('fused13','down')]
        self.assertEqual(mod.missing_cells(configs,accepted,[[41,230,'fused13'],[41,230,'down']]),[[41,229,'fused13'],[41,231,'fused13'],[41,231,'down']])
        self.assertEqual(accepted,[{'cell':[41,229,'down']}])
        with self.assertRaises(AssertionError):mod.missing_cells(configs,accepted+accepted,[])
        with self.assertRaises(AssertionError):mod.missing_cells(configs+[configs[0]],accepted,[])

if __name__=='__main__':unittest.main()
