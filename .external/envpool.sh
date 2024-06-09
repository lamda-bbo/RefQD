cd .external
rm -rf envpool
set -e
git clone --branch v0.8.4 https://github.com/sail-sg/envpool.git
cd envpool
git apply ../envpool.patch
cp third_party/pip_requirements/requirements-release.txt third_party/pip_requirements/requirements.txt
USE_BAZEL_VERSION=6.4.0 bazel run --config=release //:setup -- bdist_wheel
cp bazel-bin/setup.runfiles/envpool/dist/envpool-0.8.4.1-cp311-cp311-linux_x86_64.whl ../envpool-0.8.4.1-cp311-cp311-linux_x86_64.whl
cd ../..
