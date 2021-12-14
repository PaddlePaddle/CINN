.. python-docs documentation master file, created by
   sphinx-quickstart on Tue Nov 10 10:13:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CINN's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

CINN(Compiler Infrusture for Neural Networks) is a union of several sub-projects:

- :code:`cinn`, a domain specific language for kernel construction,
- :code:`cinnrt`, an efficient runtime framework for static graph execution.

Install
-------
.. toctree::
   :maxdepth: 1

   ./install.md
   ./guide.md

CINN
------

CINN Tutorials
~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   tutorials/index


Paddle-CINN Tutorials
~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   tutorial_paddle/index


C++ APIs
~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   matmul.md
   load_paddle_model.md
   cinn_builder_api.md
   cpp/library_root.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
