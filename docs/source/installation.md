(installation)=

# Installation

Assuming you have Python installed on your machine, start by creating a virtual
environment inside your project directory. For example, if you are a venv user, run

```{eval-rst}
.. tab-set::

    .. tab-item:: Linux/macOS
        :sync: linux

        .. code-block:: bash

            python3 -m venv myenv

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: bat

            python -m venv myenv
```

This command will create a new directory `myenv` (choose any name you like).
To activate the virtual environment, run

```{eval-rst}
.. tab-set::

    .. tab-item:: Linux/macOS
        :sync: linux

        .. code-block:: bash

            source myenv/bin/activate

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: bat

            myvenv\Scripts\activate.bat
```

If you are a conda user, run

```bash
conda create --name myenv python=3.10
conda activate myenv
```

Again, you can choose any name you like for the virtual environment, and any
supported version of Python.

Next, navigate to the `latents` directory. Install the package locally using the
following command:

```{eval-rst}
.. tab-set::

    .. tab-item:: Linux/macOS
        :sync: linux

        .. code-block:: bash

            python3 -m pip install .

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: bat

            python -m pip install .
```

Alternatively, you can install the package in editable mode, which allows you
to modify the source code and have the changes reflected immediately, without
having to reinstall the package. To do so, run

```{eval-rst}
.. tab-set::

    .. tab-item:: Linux/macOS
        :sync: linux

        .. code-block:: bash

            python3 -m pip install -e .

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: bat

            python -m pip install -e .
```

:::{note}
Depending on your system, you may need to explicitly provide the path to
the `latents` package. In that case, replace `.` with the path to the
`latents` directory.
:::

You can now import and use the `latents` package wherever your project is located.
