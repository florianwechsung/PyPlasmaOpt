pip-mac:
	rm -rf build
	env CC=gcc-10 CXX=g++-10 pip3 install -vvv -e .

pip-linux:
	rm -rf build
	pip3 install -vvv -e .

cmake-mac:
	rm -rf build-mac
	rm -f cppplasmaopt.cpython-38m-darwin.so
	mkdir build-mac
	cd build-mac ; cmake \
		-DCMAKE_CXX_COMPILER=g++-10 \
		-DCMAKE_C_COMPILER=gcc-10 \
		-DCMAKE_BUILD_TYPE=Debug \
		..
	make -C build-mac -j 8
	ln -s build-mac/cppplasmaopt.cpython-38m-darwin.so .

cmake-linux:
	rm -rf build-linux
	rm -f cppplasmaopt.cpython-36m-x86_64-linux-gnu.so
	mkdir build-linux
	cd build-linux ; cmake \
		-DCMAKE_CXX_COMPILER=g++ \
		-DCMAKE_C_COMPILER=gcc \
		-DCMAKE_BUILD_TYPE=Debug \
		..
	make -C build-linux -j 4
	ln -s build-linux/cppplasmaopt.cpython-36m-x86_64-linux-gnu.so
