"""Portable tooling contracts; no ML dependencies or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('research',ROOT/'scripts/research.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class ResearchContracts(unittest.TestCase):
    def test_all_forty_frozen_configs_match_and_parse(self):
        total=0
        for ds in R.CAMPAIGNS:
            m,_=R.manifest(ROOT,ds)
            for run in m['runs'].values():
                p=R.local_path(ROOT,run['resolved_config'])
                self.assertEqual(R.sha(p),run['resolved_config_sha256'])
                cfg=R.read_config(p)
                self.assertFalse(cfg['resume']);self.assertIsNone(cfg['load_from'])
                self.assertEqual(cfg['train_dataloader']['batch_size'],4 if ds=='pixray' else 2)
                self.assertEqual(cfg['train_cfg']['max_epochs'],50 if ds=='pixray' else 10)
                self.assertEqual(cfg['randomness']['seed'],run['seed'])
                total+=1
        self.assertEqual(total,40)

    def test_rebase_only_project_paths_preserves_model_settings(self):
        value={'model':{'width':128,'heads':4},'paths':(R.OLD_ROOT+'/data/x','torchvision://resnet101')}
        got=R.rebase(value,ROOT)
        self.assertEqual(got['model'],value['model'])
        self.assertEqual(got['paths'][0],str(ROOT/'data/x'))
        self.assertEqual(got['paths'][1],value['paths'][1])
        self.assertEqual(value['paths'][0],R.OLD_ROOT+'/data/x')

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):R.local_path(ROOT,'../outside')

    def test_config_execution_rejected(self):
        import ast
        with self.assertRaises(ValueError):R.literal(ast.parse("__import__('os').system('bad')",mode='eval').body)

    def test_all_sources_match_historical_records(self):
        for ds in R.CAMPAIGNS:
            m,_=R.manifest(ROOT,ds)
            source=json.loads(R.local_path(ROOT,m['source_snapshot']['per_file']).read_text())['files']
            for name,h in source.items():self.assertEqual(R.sha(R.local_path(ROOT,name)),h,name)


if __name__=='__main__':unittest.main()