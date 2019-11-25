all:
	rm -rf build
	env CC=gcc-9 CXX=g++-9 pip3 install -vvv -e .
