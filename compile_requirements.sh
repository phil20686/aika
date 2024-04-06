#set -e
#set pipefail

> packages.in
python --version
# first, build wheels for all the packages, but exclude dependencies so that the
# packages don't need to know where to find each other.
for package_dir in libs/*
do
  cd $package_dir
  python -m pip wheel --no-deps . -w ../../wheels
  python setup.py --name >> ../../packages.in
  cd -
done

sed -i "s/$/[test]/g" packages.in

# now that we have populated `wheels` with wheels for each package, we can point to it
# with --find-links to ensure that each package can resolve all of its local
# dependencies.

# I actually think that this step is unnecessary for our needs? But I think it might be
# useful for deployments, so I'll keep it commented out for now.

#for package_dir in libs/*
#do
#  cd $package_dir
#  pip wheel . -w ../../wheels --find-links ../../wheels
#  cd -
#done


#VERSION=$(python -c 'import sys; print(".".join([str(x) for x in sys.version_info[:2]]))')
#REQUIREMENTS="requirements$VERSION.txt"
REQUIREMENTS="requirements.txt"


pip-compile packages.in -o $REQUIREMENTS --find-links ./wheels --upgrade --rebuild -q

# replace all references to the local packages with editable path versions
sed -i -E "s|^aika-([^=]+)==.*$|-e libs/\1|g" $REQUIREMENTS