from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("LUTbeacontest2",["/home/pi/ip/LUTbeacontest2.pyx"],library_dirs=['.'],extra_compile_args=["-Ofast"])]

setup(
    name="LUTbeacontest2",
    cmdclass={"build_ext":build_ext},
    ext_modules= ext_modules
)
