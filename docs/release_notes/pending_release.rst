Pending Release Notes
=====================

Updates / New Features
----------------------

Misc.

* Update .gitignore away from the old monorepo version.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

Dependency Versions

* Update the locked version of urllib3 to address a security vulnerability.

* Update the developer dependency and locked version of ipython to address a
  security vulnerability.

* Update the required and locked version of pillow to address a security
  vulnerability.

* Removed `jedi = "^0.17"` requirement and update to `ipython = "^7.17.3"`
  since recent ipython update appropriately addresses the dependency.
