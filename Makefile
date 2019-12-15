all:
	rm -rf build
	env CC=gcc-9 CXX=g++-9 pip3 install -vvv -e .

cmake:
	rm -rf build
	rm -f cppplasmaopt.cpython-37m-darwin.so
	mkdir build
	cd build ; cmake \
		-DCMAKE_CXX_COMPILER=g++-9 \
		-DCMAKE_C_COMPILER=gcc-9 \
		-DCMAKE_BUILD_TYPE=Debug \
		..
	make -C build -j 4
	ln -s build/cppplasmaopt.cpython-37m-darwin.so .
