# Troubleshooting

## Installation of MacOSX

If you encounter problem with openmp during installation of MacOSX.

1. Install LLVM via brew
```console
brew install llvm libomp
```

2. Export variables to define paths to Clang and related libraries installed
```console
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include"
```
