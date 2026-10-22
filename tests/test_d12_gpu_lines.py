"""Memory batching must partition full spatial axes without gaps or cropped context."""
import ast
from pathlib import Path
import math
import unittest

class FullAxisBatching(unittest.TestCase):
    def test_rectangular_tiles_preserve_axes_and_cover_each_voxel_once(self):
        path = Path(__file__).resolve().parents[1] / 'src/d12_seeds/gpu_scoring.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'axis_tiles')
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        shape = (7, 11, 19)
        for axis in range(3):
            counts = {}
            for tile in scope['axis_tiles'](shape, axis, 5000):
                self.assertEqual((tile[axis].start,tile[axis].stop), (0,shape[axis]))
                self.assertLessEqual(math.prod(s.stop-s.start for s in tile)*28,5000)
                import itertools
                for point in itertools.product(*(range(s.start,s.stop) for s in tile)):
                    counts[point] = counts.get(point,0)+1
            self.assertEqual(len(counts),math.prod(shape))
            self.assertTrue(all(v==1 for v in counts.values()))

    def test_global_axis_coverage_and_workspace_bound(self):
        path = Path(__file__).resolve().parents[1] / 'src/d12_seeds/gpu_scoring.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'line_batches')
        scope = {'math': math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        for shape in [(7, 11, 19), (100, 4096, 4096)]:
            for axis, d in enumerate(shape):
                batches = list(scope['line_batches'](shape, axis, 256 << 20))
                self.assertEqual(batches[0][0], 0)
                self.assertEqual(batches[-1][1], math.prod(shape)//d)
                self.assertTrue(all(left[1] == right[0] for left,right in zip(batches,batches[1:])))
                self.assertTrue(all((stop-first)*(28*d+8) <= 256 << 20 for first,stop in batches))

if __name__ == '__main__':
    unittest.main()
