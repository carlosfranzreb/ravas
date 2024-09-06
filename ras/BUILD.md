# Compile EXECUTABLE for `stream_processor` GUI

create a packed binary (executable) for the `stream_processor` GUI

## Prerequisites

install [`pyinstaller`][1]

```bash
pyhton -m pip install pyinstaller
```

## Bundle Binary

run `pyinstaller` with spec file `build_exec.spec` from this directory

```bash
pyinstaller build_exec.spec
```

the bundled application will be in sub-directory `dist/`


------

[1]: https://pyinstaller.org/en/stable/
