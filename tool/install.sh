# conda env remove --name test
conda create -n test python=3.9 -y
conda activate test

pip install --upgrade pip
rm -rf dist/
rm -rf build/
python setup.py sdist bdist_wheel
pip install --upgrade "dist/paddlelabel_ml-$(cat paddlelabel_ml/version).tar.gz"
cd
paddlelabel_ml