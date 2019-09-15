from setuptools import setup
from torch.utils import cpp_extension

setup(name="rasterize_triangles_cpp",
      ext_modules=[
         cpp_extension.CppExtension(
            "rasterize_triangles_cpp", ["rasterize_triangles.cpp"]),
      ],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
