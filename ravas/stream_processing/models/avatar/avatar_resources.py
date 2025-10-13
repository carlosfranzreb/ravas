import os


PROJECT_BASE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
""" root directory of project """

# HACK "detect" if module is run from compiled EXEC (vs. run via normal python interpreter/script):
#      compiled EXEC's "root directory" is only three dirs up (see also build configuration in `build_exec.spec`)
if not os.path.exists(os.path.join(PROJECT_BASE_DIR, "rpm")):
    PROJECT_BASE_DIR = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )


def get_web_extension_path() -> str:
    return os.path.join(PROJECT_BASE_DIR, "rpm", "dist", "chrome-extension")


def get_web_extension_file() -> str:
    return os.path.join(PROJECT_BASE_DIR, "rpm", "dist", "chrome-extension.crx")


def get_avatar_models_dir() -> str:
    # try the (unpacked) extension's directory first:
    # this directory is included in the compiled/packaged app (see build_exec.spec), but
    # it may not exist in the source/repo, if the extension was not build yet
    # ... so use the rpm/public/ directory as a fallback, since that in included in the git repo
    unpacked_ext_dir_path = get_web_extension_path()
    if os.path.exists(unpacked_ext_dir_path):
        return unpacked_ext_dir_path
    return os.path.join(PROJECT_BASE_DIR, "rpm", "public")
