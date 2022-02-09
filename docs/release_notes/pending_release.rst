Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated CI unittests workflow to include codecov reporting and to run
  nightly.

Misc.

* Updated .gitignore away from the old monorepo version.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

Dependency Versions

* Updated the locked version of urllib3 to address a security vulnerability.

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Updated the required and locked version of pillow to address a security
  vulnerability.

* Removed `jedi = "^0.17"` requirement and updated to `ipython = "^7.17.3"`
  since recent ipython update appropriately addresses the dependency.
