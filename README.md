# PlasmaOpt

## Requirements

On a recent linux (e.g. Ubuntu > 18.04), most requirements should be met.
First install an MPI library of your choice via
    
    sudo apt install mpich

or
    
    brew install mpich

On mac, install python via

    brew install python3

and make sure to follow the instructions under _Caveats_ when installing python.

## Installation

To install run

    git clone --recursive git@github.com:florianwechsung/PyPlasmaOpt.git

or if you don't have SSH keys for GitHub set up

    git clone --recursive https://github.com/florianwechsung/PyPlasmaOpt.git

change into the directory

    cd PyPlasmaOpt/

and then run

    pip3 install -e -vvv .

To check the installation

    pytest tests/


## Common issues:

On macOS, sometimes the wrong python gets picked up. To make sure that you are using Homebrew installed python, run

    /usr/local/Cellar/python@3.8/3.8.5/bin/python3 -m pip install -e -vvv .

where you replace `3.8` and `3.8.5` by the appropriate version number.
