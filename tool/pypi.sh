rm -rf dist/*
rm -rf build/*

python tool/bumpversion.py
pip install twine
rm -rf dist/*
rm -rf build/*
python setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*.tar.gz
twine upload dist/*.tar.gz
