## Makefile
```
This Makefile rules are designed for Neat-EO.pink devs and power-users.
For plain user installation follow README.md instructions, instead.


 make install     To install, few Python dev tools and Neat-EO.pink in editable mode.
                  So any further Neat-EO.pink Python code modification will be usable at once,
                  throught either neo tools commands or neat_eo.* modules.

 make check       Launchs code tests, and tools doc updating.
                  Do it, at least, before sending a Pull Request.

 make check_tuto  Launchs neo commands embeded in tutorials, to be sure everything still up to date.
                  Do it, at least, on each CLI modifications, and before a release.
                  NOTA: It takes a while.

 make pink        Python code beautifier,
                  as Pink is the new Black ^^
```
