# VItA Python Utils

Shared utility library for the UIUC CISL Head and Mice Phantom projects.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/kevinh0718/vita_python_utils.git
```

or locally after cloning this repo:

```bash
pip install . # or "pip install -e ." for the need of actively updating the src script
```

## Usage

It's assumed that the user is familiar with the basic vita operation\
or is using this repo following instructions in some of the phantom generation repo

If none of the above conditions applied, please follow the `example/demo_usage.ipynb`.

Before running the notebook, please setup the vita docker container.

1. Make sure the `vita_setup.sh` and `vita_test.sh` are executable. If not, run:
```bash
chmod u+x vita_setup.sh vita_test.sh
```

2. Execute `vita_setup.sh`. You can examine the setup using `vita_test.sh`.\
If all goes well, you can see `demo_sphere.cco` and `demo_sphere.vtp` being generated.

Note that these two commands should be execute in this repo.\
If not, please update the path to the `example/demo` folder