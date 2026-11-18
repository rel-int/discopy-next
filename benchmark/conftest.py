""" ``geometric_modal.py`` is a Modal app run with ``modal run``, not a
benchmark: it imports ``modal``, which the workflow does not install. """

collect_ignore = ["geometric_modal.py"]
