run:
	python3 scenario1.py
	python3 scenario2.py
	python3 scenario3.py

install:
	pip install -r requirements.txt


build:
	python3 setup.py build bdist_wheel


clean:
	if exist "./build" rm -rf build
	if exist "./dist" rm -rf dist
	if exist "./mymapackage.egg-info" rm -rf mypackage.egg-info