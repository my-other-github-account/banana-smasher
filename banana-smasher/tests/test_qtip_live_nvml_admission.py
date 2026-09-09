"""No subprocesses or cached occupancy in the GB10 admission hot path."""
import ast
from functools import lru_cache
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'src/banana_smasher'

class LiveAdmission(unittest.TestCase):
    def test_live_nvml_without_process_spawn(self):
        tree = ast.parse((ROOT/'qtip_rings.py').read_text())
        names = {'effective_cuda_free_bytes','qtip_admission_memory','_qtip_nvml','_qtip_compute_pids'}
        ns = {'Path': Path, 'lru_cache': lru_cache}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'qtip_rings.py','exec'),ns)
        calls = []; occupants = [[os.getpid()], []]; initialized=[]
        nvml = SimpleNamespace(nvmlInit=lambda:initialized.append(True),nvmlDeviceGetCount=lambda:2,
            nvmlDeviceGetHandleByIndex=lambda i:i,
            nvmlDeviceGetComputeRunningProcesses=lambda i:(calls.append(i) or [SimpleNamespace(pid=p) for p in occupants[i]]))
        cuda = SimpleNamespace(mem_get_info=lambda d:(100,1000), memory_reserved=lambda d:40,
            memory_allocated=lambda d:10,get_device_properties=lambda d:SimpleNamespace(name='NVIDIA GB10'))
        torch=SimpleNamespace(cuda=cuda)
        with patch.dict('sys.modules',pynvml=nvml), patch('subprocess.check_output',side_effect=AssertionError('hot path spawned nvidia-smi')), patch.object(Path,'read_text',return_value='MemAvailable: 20 kB\n'):
            probe=ns['qtip_admission_memory']
            self.assertEqual(probe(torch)['available_bytes'],20480)
            self.assertEqual(probe(torch)['available_bytes'],20480)
            self.assertEqual(calls,[0,1,0,1]); self.assertEqual(len(initialized),1)
            occupants[1]=[os.getpid()+100000]
            with self.assertRaisesRegex(RuntimeError,'foreign GPU'):probe(torch)
            nvml.nvmlDeviceGetComputeRunningProcesses=lambda i:(_ for _ in ()).throw(RuntimeError('NVML unavailable'))
            with self.assertRaisesRegex(RuntimeError,'NVML unavailable'):probe(torch)
            nvml.nvmlDeviceGetCount=lambda:0
            with self.assertRaisesRegex(RuntimeError,'no devices'):probe(torch)
            cuda.get_device_properties=lambda d:SimpleNamespace(name='NVIDIA H100')
            self.assertEqual(probe(torch)['available_bytes'],130)

if __name__=='__main__':unittest.main()
