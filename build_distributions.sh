# build the distributions.

for package_dir in libs/*
do
  cd $package_dir
  python -m build -o ./../../dist/ -n -x
  cd -
done