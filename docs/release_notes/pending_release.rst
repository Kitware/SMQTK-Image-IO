Pending Release Notes
=====================


Updates / New Features
----------------------

CI

* Add workflow to inherit the smqtk-core publish workflow.

Misc.

* Update .gitignore away from the old monorepo version.

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.


Fixes
-----

CI

* Also run CI unittests for PRs targetting branches that match the `release*`
  glob.

Misc.

 * Add smqtk_plugins section to pyproject.toml so that ImageReader
   implementations are discoverable
