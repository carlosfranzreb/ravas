# Compile EXECUTABLE for `stream_processor` GUI

create a packed binary (executable) for the `stream_processor` GUI

## Prerequisites

install [`pyinstaller`][1]

```bash
pyhton -m pip install pyinstaller
```

## Prepare Resources

the build process includes the following, additional resources in the bundled executable:

 * `ravas/configs`: the configuration files (`*.yaml`)
 * `ravas/onnx` and `ravas/target_feats`: the models & resources ("voices") for anonymizing the audio
   * see the GitHub page for [releases][2] for downloading the corresponding `*.onnx` and `*.pt` files, 
     specifically the releases
     * [v0.1][3]: models & default voice (`*.onnx` and `*.pt`)
     * [v0.2][4]: additional voices (`*.pt`)
 * `rpm/dist`: the web-app for the browser-based avatar renderer and the avatar model files
   * see [rpm/README.md][5] for build the Chrome Extension (`*.crx`) for the browser-based avatar renderer
   * in any case, do include the directory `rpm/dist/chrome-extension/*` which will also contain the avatar model files  
     (NOTE: these avatar model files are also required by the non-browser-based avatar renderer, i.e. the `python`-based renderer!)


Make sure these resources are present at the expected locations before starting the build / bundle process!

## Bundle Binary

run `pyinstaller` with spec file `build_exec.spec` from this directory

```bash
pyinstaller build_exec.spec
```

the bundled application will be in sub-directory `dist/`


------

[1]: https://pyinstaller.org/en/stable/
[2]: https://github.com/carlosfranzreb/stream_processing/releases
[3]: https://github.com/carlosfranzreb/stream_processing/releases/tag/v0.1
[4]: https://github.com/carlosfranzreb/stream_processing/releases/tag/v0.2
[5]: ../rpm/README.md
