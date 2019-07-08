from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name="rasterize_triangles_cpp",
      ext_modules=[
        CppExtension(
            "rasterize_triangles_cpp",
            [
                "rasterize_triangles_grad.cpp",
                "rasterize_triangles_impl.h",
                "rasterize_triangles_impl.cpp",
                "rasterize_triangles.cpp"
            ]),
      ],
      cmdclass={"build_ext": BuildExtension})
